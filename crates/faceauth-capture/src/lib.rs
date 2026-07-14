//! Bounded V4L2 capture and monotonic IR/RGB frame pairing.

use std::{
    io,
    path::{Path, PathBuf},
    time::Duration,
};

use faceauth_core::{CaptureModality, CapturePair};
use serde::Serialize;
use thiserror::Error;
use v4l::{
    Device, Format, FourCC,
    buffer::{Flags, Type},
    fraction::Fraction,
    io::{mmap::Stream as MmapStream, traits::CaptureStream},
    video::Capture,
};
use zeroize::Zeroizing;

/// Pixel encoding accepted by the capture boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum PixelFormat {
    /// One byte of grayscale intensity per pixel.
    Gray8,
    /// Packed YUYV 4:2:2.
    Yuyv,
    /// Motion JPEG. Decoding must occur in a separately bounded preprocessing step.
    Mjpeg,
}

impl PixelFormat {
    fn fourcc(self) -> FourCC {
        match self {
            Self::Gray8 => FourCC::new(b"GREY"),
            Self::Yuyv => FourCC::new(b"YUYV"),
            Self::Mjpeg => FourCC::new(b"MJPG"),
        }
    }
}

/// Exact capture requirements for one camera stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct CaptureSpec {
    /// Required frame width.
    pub width: u32,
    /// Required frame height.
    pub height: u32,
    /// Required pixel encoding.
    pub pixel_format: PixelFormat,
    /// Required frames per second.
    pub frames_per_second: u32,
    /// Number of kernel mmap buffers.
    pub buffer_count: u32,
    /// Frames discarded after stream start.
    pub warmup_frames: u8,
    /// Per-frame poll timeout.
    pub frame_timeout_millis: u32,
    /// Maximum accepted `bytesused` value.
    pub max_frame_bytes: usize,
}

impl CaptureSpec {
    /// Validate bounded capture settings for a camera modality.
    ///
    /// # Errors
    ///
    /// Returns [`CaptureError::InvalidSpec`] when dimensions, timing, buffers, frame bounds, or
    /// modality-specific pixel format are invalid.
    pub fn validate(self, modality: CaptureModality) -> Result<(), CaptureError> {
        let valid_format = match modality {
            CaptureModality::Infrared => self.pixel_format == PixelFormat::Gray8,
            CaptureModality::Visible => {
                matches!(self.pixel_format, PixelFormat::Yuyv | PixelFormat::Mjpeg)
            }
        };
        if self.width == 0
            || self.height == 0
            || self.frames_per_second == 0
            || !(2..=16).contains(&self.buffer_count)
            || self.warmup_frames > 30
            || self.frame_timeout_millis == 0
            || self.max_frame_bytes == 0
            || !valid_format
        {
            return Err(CaptureError::InvalidSpec);
        }
        Ok(())
    }
}

/// Negotiated V4L2 stream format.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct NegotiatedFormat {
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Pixel encoding.
    pub pixel_format: PixelFormat,
    /// Frames per second.
    pub frames_per_second: u32,
}

/// Metadata safe to log or return from diagnostics.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct FrameSummary {
    /// Camera modality.
    pub modality: CaptureModality,
    /// Monotonic kernel timestamp in microseconds.
    pub timestamp_micros: u64,
    /// Driver frame sequence.
    pub sequence: u32,
    /// Number of ephemeral frame bytes.
    pub bytes: usize,
    /// Negotiated format.
    pub format: NegotiatedFormat,
}

/// Ephemeral frame whose bytes are zeroized on drop.
pub struct CapturedFrame {
    summary: FrameSummary,
    data: Zeroizing<Vec<u8>>,
}

impl CapturedFrame {
    /// Borrow frame bytes for bounded preprocessing or inference.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.data
    }

    /// Return metadata that does not expose frame content.
    #[must_use]
    pub const fn summary(&self) -> &FrameSummary {
        &self.summary
    }
}

/// Paired ephemeral IR and visible-light observations.
pub struct PairedFrames {
    /// Near-infrared frame.
    pub infrared: CapturedFrame,
    /// Visible-light frame.
    pub visible: CapturedFrame,
}

impl PairedFrames {
    /// Return timestamp metadata for the core capture policy.
    #[must_use]
    pub const fn timing(&self) -> CapturePair {
        CapturePair {
            infrared_timestamp_micros: self.infrared.summary.timestamp_micros,
            visible_timestamp_micros: Some(self.visible.summary.timestamp_micros),
        }
    }
}

/// Bounded frame-pairing policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct PairingPolicy {
    /// Maximum timestamp difference accepted as one observation.
    pub max_skew_micros: u64,
    /// Maximum frame replacements attempted before failure.
    pub max_replacements: u8,
}

impl PairingPolicy {
    /// Validate pairing bounds.
    ///
    /// # Errors
    ///
    /// Returns [`CaptureError::InvalidSpec`] when no skew or replacement budget is configured.
    pub const fn validate(self) -> Result<(), CaptureError> {
        if self.max_skew_micros == 0 || self.max_replacements == 0 {
            Err(CaptureError::InvalidSpec)
        } else {
            Ok(())
        }
    }
}

/// Frame source used by pairing and test doubles.
pub trait FrameSource {
    /// Capture one fresh frame.
    ///
    /// # Errors
    ///
    /// Returns [`CaptureError`] on timeout, malformed metadata, device loss, or an invalid frame.
    fn next_frame(&mut self) -> Result<CapturedFrame, CaptureError>;
}

/// Configured V4L2 capture device. A stream borrows this device for its lifetime.
pub struct V4l2CaptureDevice {
    device: Device,
    node: PathBuf,
    modality: CaptureModality,
    spec: CaptureSpec,
    negotiated: NegotiatedFormat,
}

impl V4l2CaptureDevice {
    /// Open and require an exact V4L2 format and frame rate.
    ///
    /// # Errors
    ///
    /// Returns [`CaptureError`] when the spec is invalid, the device cannot be opened, or V4L2
    /// negotiates a different format or frame rate.
    pub fn open(
        node: impl AsRef<Path>,
        modality: CaptureModality,
        spec: CaptureSpec,
    ) -> Result<Self, CaptureError> {
        spec.validate(modality)?;
        let node = node.as_ref().to_owned();
        let device = Device::with_path(&node)?;
        let requested = Format::new(spec.width, spec.height, spec.pixel_format.fourcc());
        let actual = device.set_format(&requested)?;
        if actual.width != spec.width
            || actual.height != spec.height
            || actual.fourcc != spec.pixel_format.fourcc()
        {
            return Err(CaptureError::NegotiatedFormatMismatch);
        }
        let mut parameters = device.params()?;
        parameters.interval = Fraction::new(1, spec.frames_per_second);
        let actual_parameters = device.set_params(&parameters)?;
        let numerator = u64::from(actual_parameters.interval.numerator);
        let denominator = u64::from(actual_parameters.interval.denominator);
        if numerator == 0
            || denominator
                != numerator
                    .checked_mul(u64::from(spec.frames_per_second))
                    .ok_or(CaptureError::InvalidSpec)?
        {
            return Err(CaptureError::NegotiatedFrameRateMismatch);
        }
        Ok(Self {
            device,
            node,
            modality,
            spec,
            negotiated: NegotiatedFormat {
                width: actual.width,
                height: actual.height,
                pixel_format: spec.pixel_format,
                frames_per_second: spec.frames_per_second,
            },
        })
    }

    /// Start an mmap stream and discard configured warmup frames.
    ///
    /// # Errors
    ///
    /// Returns [`CaptureError`] when mmap setup or a warmup capture fails.
    pub fn stream(&self) -> Result<V4l2FrameStream<'_>, CaptureError> {
        let stream =
            MmapStream::with_buffers(&self.device, Type::VideoCapture, self.spec.buffer_count)?;
        let mut stream = V4l2FrameStream {
            stream,
            node: &self.node,
            modality: self.modality,
            negotiated: self.negotiated,
            max_frame_bytes: self.spec.max_frame_bytes,
        };
        stream.stream.set_timeout(Duration::from_millis(u64::from(self.spec.frame_timeout_millis)));
        for _ in 0..self.spec.warmup_frames {
            let _frame = stream.next_frame()?;
        }
        Ok(stream)
    }

    /// Return the configured V4L2 node.
    #[must_use]
    pub fn node(&self) -> &Path {
        &self.node
    }

    /// Return the exact negotiated stream format.
    #[must_use]
    pub const fn negotiated(&self) -> NegotiatedFormat {
        self.negotiated
    }
}

/// Active mmap capture stream.
pub struct V4l2FrameStream<'a> {
    stream: MmapStream<'a>,
    node: &'a Path,
    modality: CaptureModality,
    negotiated: NegotiatedFormat,
    max_frame_bytes: usize,
}

impl FrameSource for V4l2FrameStream<'_> {
    fn next_frame(&mut self) -> Result<CapturedFrame, CaptureError> {
        let (buffer, metadata) = self.stream.next().map_err(|error| {
            if error.kind() == io::ErrorKind::TimedOut {
                CaptureError::FrameTimeout { node: self.node.to_owned() }
            } else {
                CaptureError::Io(error)
            }
        })?;
        if metadata.flags.contains(Flags::ERROR) {
            return Err(CaptureError::CorruptFrame { node: self.node.to_owned() });
        }
        if !metadata.flags.contains(Flags::TIMESTAMP_MONOTONIC) {
            return Err(CaptureError::NonMonotonicTimestamp { node: self.node.to_owned() });
        }
        let bytes_used = usize::try_from(metadata.bytesused)
            .map_err(|_| CaptureError::FrameSizeInvalid { node: self.node.to_owned() })?;
        if bytes_used == 0 || bytes_used > buffer.len() || bytes_used > self.max_frame_bytes {
            return Err(CaptureError::FrameSizeInvalid { node: self.node.to_owned() });
        }
        let timestamp_micros = timestamp_micros(metadata.timestamp.sec, metadata.timestamp.usec)?;
        Ok(CapturedFrame {
            summary: FrameSummary {
                modality: self.modality,
                timestamp_micros,
                sequence: metadata.sequence,
                bytes: bytes_used,
                format: self.negotiated,
            },
            data: Zeroizing::new(buffer[..bytes_used].to_vec()),
        })
    }
}

/// Capture a bounded, timestamp-aligned IR/RGB observation.
///
/// The older stream advances until the frames satisfy the configured skew. Frames discarded by
/// this algorithm are zeroized when replaced.
///
/// # Errors
///
/// Returns [`CaptureError`] when either source fails, reports the wrong modality, or cannot produce
/// an aligned pair within the replacement budget.
pub fn capture_pair(
    infrared: &mut impl FrameSource,
    visible: &mut impl FrameSource,
    policy: PairingPolicy,
) -> Result<PairedFrames, CaptureError> {
    policy.validate()?;
    let mut infrared_frame = infrared.next_frame()?;
    let mut visible_frame = visible.next_frame()?;
    ensure_modality(&infrared_frame, CaptureModality::Infrared)?;
    ensure_modality(&visible_frame, CaptureModality::Visible)?;

    for replacement in 0..=policy.max_replacements {
        let skew = infrared_frame
            .summary
            .timestamp_micros
            .abs_diff(visible_frame.summary.timestamp_micros);
        if skew <= policy.max_skew_micros {
            return Ok(PairedFrames { infrared: infrared_frame, visible: visible_frame });
        }
        if replacement == policy.max_replacements {
            return Err(CaptureError::PairingBudgetExhausted { last_skew_micros: skew });
        }
        if infrared_frame.summary.timestamp_micros < visible_frame.summary.timestamp_micros {
            infrared_frame = infrared.next_frame()?;
            ensure_modality(&infrared_frame, CaptureModality::Infrared)?;
        } else {
            visible_frame = visible.next_frame()?;
            ensure_modality(&visible_frame, CaptureModality::Visible)?;
        }
    }
    Err(CaptureError::InvalidSpec)
}

fn ensure_modality(frame: &CapturedFrame, expected: CaptureModality) -> Result<(), CaptureError> {
    if frame.summary.modality == expected {
        Ok(())
    } else {
        Err(CaptureError::WrongModality { expected, actual: frame.summary.modality })
    }
}

fn timestamp_micros(seconds: i64, microseconds: i64) -> Result<u64, CaptureError> {
    let seconds = u64::try_from(seconds).map_err(|_| CaptureError::InvalidTimestamp)?;
    let microseconds = u64::try_from(microseconds).map_err(|_| CaptureError::InvalidTimestamp)?;
    if microseconds >= 1_000_000 {
        return Err(CaptureError::InvalidTimestamp);
    }
    seconds
        .checked_mul(1_000_000)
        .and_then(|value| value.checked_add(microseconds))
        .ok_or(CaptureError::InvalidTimestamp)
}

/// Capture or pairing failure.
#[derive(Debug, Error)]
pub enum CaptureError {
    /// Capture or pairing configuration is invalid.
    #[error("invalid capture specification")]
    InvalidSpec,
    /// V4L2 selected a different image format or size.
    #[error("V4L2 did not accept the exact requested format")]
    NegotiatedFormatMismatch,
    /// V4L2 selected a different frame rate.
    #[error("V4L2 did not accept the exact requested frame rate")]
    NegotiatedFrameRateMismatch,
    /// A frame was not ready before the bounded deadline.
    #[error("frame capture timed out on {node}")]
    FrameTimeout {
        /// V4L2 device node.
        node: PathBuf,
    },
    /// Driver marked the frame as corrupted.
    #[error("driver reported a corrupt frame on {node}")]
    CorruptFrame {
        /// V4L2 device node.
        node: PathBuf,
    },
    /// Driver did not identify its timestamp as monotonic.
    #[error("frame on {node} lacks a monotonic kernel timestamp")]
    NonMonotonicTimestamp {
        /// V4L2 device node.
        node: PathBuf,
    },
    /// Frame byte count was empty, out of bounds, or larger than policy.
    #[error("frame size is invalid on {node}")]
    FrameSizeInvalid {
        /// V4L2 device node.
        node: PathBuf,
    },
    /// Kernel timestamp was negative, malformed, or overflowed.
    #[error("frame timestamp is invalid")]
    InvalidTimestamp,
    /// A source supplied a frame for another modality.
    #[error("frame source returned {actual:?}; expected {expected:?}")]
    WrongModality {
        /// Required modality.
        expected: CaptureModality,
        /// Actual modality.
        actual: CaptureModality,
    },
    /// Streams could not produce one aligned pair within policy.
    #[error("frame pairing budget exhausted; last skew was {last_skew_micros}us")]
    PairingBudgetExhausted {
        /// Last observed timestamp difference.
        last_skew_micros: u64,
    },
    /// V4L2 or file-descriptor operation failed.
    #[error("V4L2 operation failed: {0}")]
    Io(#[from] io::Error),
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::*;

    struct FakeSource {
        frames: VecDeque<CapturedFrame>,
    }

    impl FrameSource for FakeSource {
        fn next_frame(&mut self) -> Result<CapturedFrame, CaptureError> {
            self.frames.pop_front().ok_or(CaptureError::InvalidSpec)
        }
    }

    fn frame(modality: CaptureModality, timestamp_micros: u64) -> CapturedFrame {
        CapturedFrame {
            summary: FrameSummary {
                modality,
                timestamp_micros,
                sequence: 1,
                bytes: 3,
                format: NegotiatedFormat {
                    width: 1,
                    height: 1,
                    pixel_format: if modality == CaptureModality::Infrared {
                        PixelFormat::Gray8
                    } else {
                        PixelFormat::Yuyv
                    },
                    frames_per_second: 15,
                },
            },
            data: Zeroizing::new(vec![1, 2, 3]),
        }
    }

    #[test]
    fn older_frames_advance_until_pair_is_aligned() -> Result<(), CaptureError> {
        let mut infrared = FakeSource {
            frames: VecDeque::from([
                frame(CaptureModality::Infrared, 1_000_000),
                frame(CaptureModality::Infrared, 1_200_000),
            ]),
        };
        let mut visible =
            FakeSource { frames: VecDeque::from([frame(CaptureModality::Visible, 1_190_000)]) };

        let pair = capture_pair(
            &mut infrared,
            &mut visible,
            PairingPolicy { max_skew_micros: 20_000, max_replacements: 2 },
        )?;

        assert_eq!(pair.timing().infrared_timestamp_micros, 1_200_000);
        assert_eq!(pair.timing().visible_timestamp_micros, Some(1_190_000));
        Ok(())
    }

    #[test]
    fn wrong_modality_fails_closed() {
        let mut infrared =
            FakeSource { frames: VecDeque::from([frame(CaptureModality::Visible, 1_000_000)]) };
        let mut visible =
            FakeSource { frames: VecDeque::from([frame(CaptureModality::Visible, 1_000_000)]) };

        assert!(matches!(
            capture_pair(
                &mut infrared,
                &mut visible,
                PairingPolicy { max_skew_micros: 20_000, max_replacements: 1 }
            ),
            Err(CaptureError::WrongModality { .. })
        ));
    }

    #[test]
    fn timestamp_conversion_is_checked() {
        assert!(matches!(timestamp_micros(12, 345), Ok(12_000_345)));
        assert!(timestamp_micros(-1, 0).is_err());
        assert!(timestamp_micros(1, 1_000_000).is_err());
    }

    #[test]
    fn modality_restricts_accepted_pixel_formats() {
        let base = CaptureSpec {
            width: 640,
            height: 360,
            pixel_format: PixelFormat::Gray8,
            frames_per_second: 15,
            buffer_count: 4,
            warmup_frames: 2,
            frame_timeout_millis: 1_500,
            max_frame_bytes: 1024 * 1024,
        };

        assert!(base.validate(CaptureModality::Infrared).is_ok());
        assert!(base.validate(CaptureModality::Visible).is_err());
        assert!(
            CaptureSpec { pixel_format: PixelFormat::Mjpeg, ..base }
                .validate(CaptureModality::Visible)
                .is_ok()
        );
    }

    #[test]
    fn pairing_budget_exhaustion_reports_last_skew() {
        let mut infrared = FakeSource {
            frames: VecDeque::from([
                frame(CaptureModality::Infrared, 1_000_000),
                frame(CaptureModality::Infrared, 1_010_000),
            ]),
        };
        let mut visible =
            FakeSource { frames: VecDeque::from([frame(CaptureModality::Visible, 2_000_000)]) };

        assert!(matches!(
            capture_pair(
                &mut infrared,
                &mut visible,
                PairingPolicy { max_skew_micros: 20_000, max_replacements: 1 }
            ),
            Err(CaptureError::PairingBudgetExhausted { last_skew_micros: 990_000 })
        ));
    }
}
