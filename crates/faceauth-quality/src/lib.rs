//! Bounded, desktop-independent face image quality assessment.
//!
//! This crate consumes only borrowed, tightly packed image views. It does not own, retain, copy,
//! serialize, or return image pixels. Callers provide a normalized face box and already-derived
//! pose/visibility measurements; the result contains only bounded calibration measurements and
//! scores.

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

/// Hard ceiling applied before any image scan, independent of runtime configuration.
pub const ABSOLUTE_MAX_IMAGE_PIXELS: usize = 16_777_216;
/// Largest accepted width or height, independent of runtime configuration.
pub const ABSOLUTE_MAX_IMAGE_DIMENSION: u32 = 8_192;

const LUMA_LEVELS: usize = 256;
const CANCEL_CHECK_PIXELS: usize = 4_096;

/// A borrowed, tightly packed 8-bit grayscale image.
#[derive(Clone, Copy)]
pub struct Gray8View<'a> {
    width: u32,
    height: u32,
    pixels: &'a [u8],
}

impl fmt::Debug for Gray8View<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Gray8View")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("pixels", &"<redacted>")
            .finish()
    }
}

impl<'a> Gray8View<'a> {
    /// Construct a view after checking dimensions, arithmetic, the absolute resource ceiling, and
    /// exact buffer length.
    ///
    /// # Errors
    ///
    /// Returns [`QualityError`] when the image is empty, oversized, or not tightly packed.
    pub fn new(width: u32, height: u32, pixels: &'a [u8]) -> Result<Self, QualityError> {
        validate_image_layout(width, height, pixels.len(), 1)?;
        Ok(Self { width, height, pixels })
    }

    /// Image width in pixels.
    #[must_use]
    pub const fn width(self) -> u32 {
        self.width
    }

    /// Image height in pixels.
    #[must_use]
    pub const fn height(self) -> u32 {
        self.height
    }
}

/// A borrowed, tightly packed interleaved RGB8 image.
#[derive(Clone, Copy)]
pub struct Rgb8View<'a> {
    width: u32,
    height: u32,
    pixels: &'a [u8],
}

impl fmt::Debug for Rgb8View<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Rgb8View")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("pixels", &"<redacted>")
            .finish()
    }
}

impl<'a> Rgb8View<'a> {
    /// Construct a view after checking dimensions, arithmetic, the absolute resource ceiling, and
    /// exact buffer length.
    ///
    /// # Errors
    ///
    /// Returns [`QualityError`] when the image is empty, oversized, or not tightly packed RGB8.
    pub fn new(width: u32, height: u32, pixels: &'a [u8]) -> Result<Self, QualityError> {
        validate_image_layout(width, height, pixels.len(), 3)?;
        Ok(Self { width, height, pixels })
    }

    /// Image width in pixels.
    #[must_use]
    pub const fn width(self) -> u32 {
        self.width
    }

    /// Image height in pixels.
    #[must_use]
    pub const fn height(self) -> u32 {
        self.height
    }
}

/// Supported borrowed image encodings.
#[derive(Clone, Copy, Debug)]
pub enum ImageView<'a> {
    /// One luma byte per pixel.
    Gray8(Gray8View<'a>),
    /// Interleaved red, green, and blue bytes per pixel.
    Rgb8(Rgb8View<'a>),
}

impl ImageView<'_> {
    const fn dimensions(self) -> (u32, u32) {
        match self {
            Self::Gray8(view) => (view.width, view.height),
            Self::Rgb8(view) => (view.width, view.height),
        }
    }

    fn luma(self, x: usize, y: usize) -> u8 {
        match self {
            Self::Gray8(view) => {
                let index = y * view.width as usize + x;
                view.pixels[index]
            }
            Self::Rgb8(view) => {
                let index = (y * view.width as usize + x) * 3;
                let red = u32::from(view.pixels[index]);
                let green = u32::from(view.pixels[index + 1]);
                let blue = u32::from(view.pixels[index + 2]);
                let luma = (77 * red + 150 * green + 29 * blue + 128) >> 8;
                u8::try_from(luma).unwrap_or(u8::MAX)
            }
        }
    }
}

impl<'a> From<Gray8View<'a>> for ImageView<'a> {
    fn from(value: Gray8View<'a>) -> Self {
        Self::Gray8(value)
    }
}

impl<'a> From<Rgb8View<'a>> for ImageView<'a> {
    fn from(value: Rgb8View<'a>) -> Self {
        Self::Rgb8(value)
    }
}

/// Normalized face geometry and bounded model-derived confidence measurements.
///
/// Box coordinates use the full image as 0..=1. Pose angles are degrees. `visible_fraction`
/// represents the estimated unoccluded fraction of the face, not a raw occlusion mask.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FaceGeometry {
    /// Horizontal center of the face box in 0..=1.
    pub center_x: f32,
    /// Vertical center of the face box in 0..=1.
    pub center_y: f32,
    /// Face box width relative to image width in 0..=1.
    pub width: f32,
    /// Face box height relative to image height in 0..=1.
    pub height: f32,
    /// Head yaw in degrees.
    pub yaw_degrees: f32,
    /// Head pitch in degrees.
    pub pitch_degrees: f32,
    /// In-plane head roll in degrees.
    pub roll_degrees: f32,
    /// Aggregate confidence of the admitted landmark set in 0..=1.
    pub landmark_confidence: f32,
    /// Estimated unoccluded face fraction in 0..=1.
    pub visible_fraction: f32,
}

impl FaceGeometry {
    fn validate(self) -> Result<(), QualityError> {
        let values = [
            self.center_x,
            self.center_y,
            self.width,
            self.height,
            self.yaw_degrees,
            self.pitch_degrees,
            self.roll_degrees,
            self.landmark_confidence,
            self.visible_fraction,
        ];
        if values.iter().any(|value| !value.is_finite())
            || !(0.0..=1.0).contains(&self.center_x)
            || !(0.0..=1.0).contains(&self.center_y)
            || !(0.0..=1.0).contains(&self.width)
            || !(0.0..=1.0).contains(&self.height)
            || self.width == 0.0
            || self.height == 0.0
            || !(0.0..=1.0).contains(&self.landmark_confidence)
            || !(0.0..=1.0).contains(&self.visible_fraction)
            || !(-90.0..=90.0).contains(&self.yaw_degrees)
            || !(-90.0..=90.0).contains(&self.pitch_degrees)
            || !(-180.0..=180.0).contains(&self.roll_degrees)
        {
            return Err(QualityError::InvalidGeometry);
        }
        let half_width = self.width / 2.0;
        let half_height = self.height / 2.0;
        if self.center_x < half_width
            || self.center_x + half_width > 1.0
            || self.center_y < half_height
            || self.center_y + half_height > 1.0
        {
            return Err(QualityError::FaceBoxOutsideImage);
        }
        Ok(())
    }
}

/// Four-point calibration for a measurement with an acceptable middle band.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BandCalibration {
    /// Values at or below this point score zero.
    pub reject_below: f32,
    /// Values at or above this point enter the ideal band.
    pub ideal_min: f32,
    /// Highest value in the ideal band.
    pub ideal_max: f32,
    /// Values at or above this point score zero.
    pub reject_above: f32,
}

impl BandCalibration {
    fn validate(self) -> bool {
        [self.reject_below, self.ideal_min, self.ideal_max, self.reject_above]
            .iter()
            .all(|value| value.is_finite())
            && self.reject_below < self.ideal_min
            && self.ideal_min <= self.ideal_max
            && self.ideal_max < self.reject_above
    }

    fn score(self, value: f32) -> f32 {
        if value <= self.reject_below || value >= self.reject_above {
            0.0
        } else if value < self.ideal_min {
            (value - self.reject_below) / (self.ideal_min - self.reject_below)
        } else if value <= self.ideal_max {
            1.0
        } else {
            (self.reject_above - value) / (self.reject_above - self.ideal_max)
        }
    }
}

/// Calibration for a measurement where larger is better.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LowerBoundCalibration {
    /// Values at or below this point score zero.
    pub reject_at_or_below: f32,
    /// Values at or above this point score one.
    pub ideal_at_or_above: f32,
}

impl LowerBoundCalibration {
    fn validate(self) -> bool {
        self.reject_at_or_below.is_finite()
            && self.ideal_at_or_above.is_finite()
            && self.reject_at_or_below < self.ideal_at_or_above
    }

    fn score(self, value: f32) -> f32 {
        ((value - self.reject_at_or_below) / (self.ideal_at_or_above - self.reject_at_or_below))
            .clamp(0.0, 1.0)
    }
}

/// Calibration for an absolute deviation where smaller is better.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeviationCalibration {
    /// Deviations at or below this point score one.
    pub ideal_at_or_below: f32,
    /// Deviations at or above this point score zero.
    pub reject_at_or_above: f32,
}

impl DeviationCalibration {
    fn validate(self) -> bool {
        self.ideal_at_or_below.is_finite()
            && self.reject_at_or_above.is_finite()
            && self.ideal_at_or_below >= 0.0
            && self.ideal_at_or_below < self.reject_at_or_above
    }

    fn score(self, absolute_deviation: f32) -> f32 {
        ((self.reject_at_or_above - absolute_deviation)
            / (self.reject_at_or_above - self.ideal_at_or_below))
            .clamp(0.0, 1.0)
    }
}

/// Relative weights for the conservative aggregate. Every component must remain enabled.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualityWeights {
    /// Exposure weight.
    pub exposure: f32,
    /// Dynamic-range weight.
    pub dynamic_range: f32,
    /// Sharpness weight.
    pub sharpness: f32,
    /// Face-scale weight.
    pub face_scale: f32,
    /// Centering weight.
    pub centering: f32,
    /// Pose weight.
    pub pose: f32,
    /// Visibility/occlusion weight.
    pub occlusion: f32,
    /// Landmark-confidence weight.
    pub landmark_confidence: f32,
}

impl QualityWeights {
    const fn values(self) -> [f32; 8] {
        [
            self.exposure,
            self.dynamic_range,
            self.sharpness,
            self.face_scale,
            self.centering,
            self.pose,
            self.occlusion,
            self.landmark_confidence,
        ]
    }

    fn validate(self) -> bool {
        self.values().iter().all(|weight| weight.is_finite() && (0.01..=100.0).contains(weight))
    }
}

/// Calibrated quality policy and resource limits.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualityConfig {
    /// Per-deployment image pixel ceiling, bounded by [`ABSOLUTE_MAX_IMAGE_PIXELS`].
    pub max_image_pixels: usize,
    /// Minimum face-region width required for stable spatial measurements.
    pub minimum_face_width_pixels: u32,
    /// Minimum face-region height required for stable spatial measurements.
    pub minimum_face_height_pixels: u32,
    /// Lower histogram percentile used for dynamic range, from 1 through 49.
    pub low_percentile: u8,
    /// Upper histogram percentile used for dynamic range, from 51 through 99.
    pub high_percentile: u8,
    /// Mean luma calibration in the native 0..=255 range.
    pub exposure: BandCalibration,
    /// Luma values at or below this level count as dark clipping.
    pub dark_clip_at_or_below: u8,
    /// Luma values at or above this level count as highlight clipping.
    pub bright_clip_at_or_above: u8,
    /// Combined dark/highlight clipped-pixel fraction calibration.
    pub clipped_fraction: DeviationCalibration,
    /// Percentile luma span calibration in the native 0..=255 range.
    pub dynamic_range: LowerBoundCalibration,
    /// Normalized Laplacian variance calibration. The raw variance is divided by 255 squared.
    pub sharpness: LowerBoundCalibration,
    /// Face box area as a fraction of full-frame area.
    pub face_fraction: BandCalibration,
    /// Horizontal normalized center deviation calibration.
    pub horizontal_centering: DeviationCalibration,
    /// Vertical normalized center deviation calibration.
    pub vertical_centering: DeviationCalibration,
    /// Absolute yaw calibration in degrees.
    pub yaw: DeviationCalibration,
    /// Absolute pitch calibration in degrees.
    pub pitch: DeviationCalibration,
    /// Absolute roll calibration in degrees.
    pub roll: DeviationCalibration,
    /// Estimated unoccluded face fraction calibration.
    pub visibility: LowerBoundCalibration,
    /// Aggregate landmark confidence calibration.
    pub landmark_confidence: LowerBoundCalibration,
    /// Component weights for the harmonic aggregate.
    pub weights: QualityWeights,
    /// Maximum amount by which the aggregate may exceed its weakest component.
    pub weakest_component_headroom: f32,
}

impl QualityConfig {
    /// Return an explicit engineering baseline for tests, diagnostics, and calibration tooling.
    ///
    /// This is not a production biometric default. A deployment must serialize reviewed values in
    /// its strict configuration and bind them to immutable calibration evidence.
    #[must_use]
    pub const fn engineering_baseline() -> Self {
        Self {
            max_image_pixels: 8_294_400,
            minimum_face_width_pixels: 32,
            minimum_face_height_pixels: 32,
            low_percentile: 5,
            high_percentile: 95,
            exposure: BandCalibration {
                reject_below: 24.0,
                ideal_min: 72.0,
                ideal_max: 184.0,
                reject_above: 240.0,
            },
            dark_clip_at_or_below: 8,
            bright_clip_at_or_above: 247,
            clipped_fraction: DeviationCalibration {
                ideal_at_or_below: 0.02,
                reject_at_or_above: 0.25,
            },
            dynamic_range: LowerBoundCalibration {
                reject_at_or_below: 12.0,
                ideal_at_or_above: 96.0,
            },
            sharpness: LowerBoundCalibration { reject_at_or_below: 0.002, ideal_at_or_above: 0.08 },
            face_fraction: BandCalibration {
                reject_below: 0.02,
                ideal_min: 0.12,
                ideal_max: 0.45,
                reject_above: 0.75,
            },
            horizontal_centering: DeviationCalibration {
                ideal_at_or_below: 0.05,
                reject_at_or_above: 0.35,
            },
            vertical_centering: DeviationCalibration {
                ideal_at_or_below: 0.08,
                reject_at_or_above: 0.40,
            },
            yaw: DeviationCalibration { ideal_at_or_below: 8.0, reject_at_or_above: 35.0 },
            pitch: DeviationCalibration { ideal_at_or_below: 8.0, reject_at_or_above: 30.0 },
            roll: DeviationCalibration { ideal_at_or_below: 8.0, reject_at_or_above: 30.0 },
            visibility: LowerBoundCalibration { reject_at_or_below: 0.45, ideal_at_or_above: 0.92 },
            landmark_confidence: LowerBoundCalibration {
                reject_at_or_below: 0.40,
                ideal_at_or_above: 0.90,
            },
            weights: QualityWeights {
                exposure: 1.0,
                dynamic_range: 1.0,
                sharpness: 1.5,
                face_scale: 1.0,
                centering: 0.75,
                pose: 1.25,
                occlusion: 1.5,
                landmark_confidence: 1.0,
            },
            weakest_component_headroom: 0.12,
        }
    }

    /// Validate all calibrations and resource bounds.
    ///
    /// Defaults are conservative engineering starting points, not production biometric
    /// calibration. Deployments must validate them on the exact camera and population.
    ///
    /// # Errors
    ///
    /// Returns [`QualityError::InvalidConfig`] for contradictory, non-finite, or excessive values.
    pub fn validate(self) -> Result<(), QualityError> {
        let minimum_face_pixels =
            usize::try_from(self.minimum_face_width_pixels).ok().and_then(|width| {
                usize::try_from(self.minimum_face_height_pixels)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            });
        if !(9..=ABSOLUTE_MAX_IMAGE_PIXELS).contains(&self.max_image_pixels)
            || !(3..=ABSOLUTE_MAX_IMAGE_DIMENSION).contains(&self.minimum_face_width_pixels)
            || !(3..=ABSOLUTE_MAX_IMAGE_DIMENSION).contains(&self.minimum_face_height_pixels)
            || minimum_face_pixels.is_none_or(|pixels| pixels > self.max_image_pixels)
            || !(1..=49).contains(&self.low_percentile)
            || !(51..=99).contains(&self.high_percentile)
            || self.low_percentile >= self.high_percentile
            || !self.exposure.validate()
            || self.exposure.reject_below < 0.0
            || self.exposure.reject_above > 255.0
            || self.dark_clip_at_or_below >= self.bright_clip_at_or_above
            || !self.clipped_fraction.validate()
            || self.clipped_fraction.reject_at_or_above > 1.0
            || !self.dynamic_range.validate()
            || self.dynamic_range.reject_at_or_below < 0.0
            || self.dynamic_range.ideal_at_or_above > 255.0
            || !self.sharpness.validate()
            || self.sharpness.reject_at_or_below < 0.0
            || self.sharpness.ideal_at_or_above > 16.0
            || !self.face_fraction.validate()
            || self.face_fraction.reject_below < 0.0
            || self.face_fraction.reject_above > 1.0
            || !self.horizontal_centering.validate()
            || self.horizontal_centering.reject_at_or_above > 0.5
            || !self.vertical_centering.validate()
            || self.vertical_centering.reject_at_or_above > 0.5
            || !self.yaw.validate()
            || self.yaw.reject_at_or_above > 90.0
            || !self.pitch.validate()
            || self.pitch.reject_at_or_above > 90.0
            || !self.roll.validate()
            || self.roll.reject_at_or_above > 180.0
            || !self.visibility.validate()
            || self.visibility.reject_at_or_below < 0.0
            || self.visibility.ideal_at_or_above > 1.0
            || !self.landmark_confidence.validate()
            || self.landmark_confidence.reject_at_or_below < 0.0
            || self.landmark_confidence.ideal_at_or_above > 1.0
            || !self.weights.validate()
            || !self.weakest_component_headroom.is_finite()
            || !(0.0..=0.5).contains(&self.weakest_component_headroom)
        {
            return Err(QualityError::InvalidConfig);
        }
        Ok(())
    }
}

/// Calibrated component scores, all in the inclusive range 0..=1.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualityComponents {
    /// Mean face-region luma score.
    pub exposure: f32,
    /// Robust percentile dynamic-range score.
    pub dynamic_range: f32,
    /// Laplacian-variance sharpness score.
    pub sharpness: f32,
    /// Face box scale score.
    pub face_scale: f32,
    /// Combined horizontal/vertical centering score.
    pub centering: f32,
    /// Combined yaw/pitch/roll score.
    pub pose: f32,
    /// Unoccluded visible-face score.
    pub occlusion: f32,
    /// Landmark-confidence score.
    pub landmark_confidence: f32,
}

impl QualityComponents {
    const fn values(self) -> [f32; 8] {
        [
            self.exposure,
            self.dynamic_range,
            self.sharpness,
            self.face_scale,
            self.centering,
            self.pose,
            self.occlusion,
            self.landmark_confidence,
        ]
    }
}

/// Non-image calibration measurements derived from the face region.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualityMeasurements {
    /// Mean face-region luma in 0..=255.
    pub mean_luma: f32,
    /// Fraction of face pixels in configured dark or highlight clipping bins.
    pub clipped_fraction: f32,
    /// Difference between the configured high and low luma percentiles in 0..=255.
    pub percentile_dynamic_range: f32,
    /// Variance of the four-neighbor Laplacian divided by 255 squared.
    pub normalized_laplacian_variance: f32,
    /// Face box area relative to full-frame area.
    pub face_fraction: f32,
}

/// Complete derived quality result; it contains no pixels, landmarks, or reusable biometric data.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualityReport {
    /// Conservative aggregate in 0..=1.
    pub aggregate: f32,
    /// Calibrated component scores.
    pub components: QualityComponents,
    /// Derived measurements useful for camera-specific calibration.
    pub measurements: QualityMeasurements,
}

/// Assess one face observation with the default non-cancelling callback.
///
/// # Errors
///
/// Returns [`QualityError`] for malformed configuration, geometry, image bounds, insufficient face
/// pixels, cancellation, or invalid arithmetic.
pub fn assess(
    image: ImageView<'_>,
    geometry: FaceGeometry,
    config: QualityConfig,
) -> Result<QualityReport, QualityError> {
    assess_cancellable(image, geometry, config, || false)
}

/// Assess one face observation while polling an owning transaction's cancellation signal.
///
/// Cancellation is checked before validation and at bounded intervals during both image passes.
/// No partial result is returned, and this crate retains no image state.
///
/// # Errors
///
/// Returns [`QualityError::Cancelled`] when requested, or the same errors as [`assess`].
pub fn assess_cancellable(
    image: ImageView<'_>,
    geometry: FaceGeometry,
    config: QualityConfig,
    mut should_cancel: impl FnMut() -> bool,
) -> Result<QualityReport, QualityError> {
    check_cancelled(&mut should_cancel)?;
    config.validate()?;
    geometry.validate()?;

    let (width, height) = image.dimensions();
    let image_pixels =
        (width as usize).checked_mul(height as usize).ok_or(QualityError::DimensionOverflow)?;
    if image_pixels > config.max_image_pixels {
        return Err(QualityError::ConfiguredImageLimitExceeded {
            actual: image_pixels,
            maximum: config.max_image_pixels,
        });
    }
    let region = PixelRegion::from_geometry(width, height, geometry)?;
    if region.width() < config.minimum_face_width_pixels as usize
        || region.height() < config.minimum_face_height_pixels as usize
    {
        return Err(QualityError::FaceRegionTooSmall {
            width: region.width(),
            height: region.height(),
        });
    }

    let statistics = scan_luma(image, region, &mut should_cancel)?;
    let laplacian_variance = scan_laplacian(image, region, &mut should_cancel)?;
    check_cancelled(&mut should_cancel)?;

    let low_luma =
        histogram_percentile(&statistics.histogram, statistics.pixel_count, config.low_percentile);
    let high_luma =
        histogram_percentile(&statistics.histogram, statistics.pixel_count, config.high_percentile);
    let mean_luma = f64::from(statistics.luma_sum) / f64::from(statistics.pixel_count);
    let clipped_pixels =
        histogram_count_inclusive(&statistics.histogram, 0, config.dark_clip_at_or_below)
            + histogram_count_inclusive(
                &statistics.histogram,
                config.bright_clip_at_or_above,
                u8::MAX,
            );
    let clipped_fraction = f64::from(clipped_pixels) / f64::from(statistics.pixel_count);
    let dynamic_range = f32::from(high_luma.saturating_sub(low_luma));
    let normalized_laplacian_variance = laplacian_variance / 65_025.0;
    let face_fraction = geometry.width * geometry.height;

    let horizontal_offset = (geometry.center_x - 0.5).abs();
    let vertical_offset = (geometry.center_y - 0.5).abs();
    let centering = config
        .horizontal_centering
        .score(horizontal_offset)
        .min(config.vertical_centering.score(vertical_offset));
    let pose = config
        .yaw
        .score(geometry.yaw_degrees.abs())
        .min(config.pitch.score(geometry.pitch_degrees.abs()))
        .min(config.roll.score(geometry.roll_degrees.abs()));
    let components = QualityComponents {
        exposure: config
            .exposure
            .score(bounded_f64_to_f32(mean_luma)?)
            .min(config.clipped_fraction.score(bounded_f64_to_f32(clipped_fraction)?)),
        dynamic_range: config.dynamic_range.score(dynamic_range),
        sharpness: config.sharpness.score(bounded_f64_to_f32(normalized_laplacian_variance)?),
        face_scale: config.face_fraction.score(face_fraction),
        centering,
        pose,
        occlusion: config.visibility.score(geometry.visible_fraction),
        landmark_confidence: config.landmark_confidence.score(geometry.landmark_confidence),
    };
    let aggregate =
        conservative_aggregate(components, config.weights, config.weakest_component_headroom)?;
    let measurements = QualityMeasurements {
        mean_luma: bounded_f64_to_f32(mean_luma)?,
        clipped_fraction: bounded_f64_to_f32(clipped_fraction)?,
        percentile_dynamic_range: dynamic_range,
        normalized_laplacian_variance: bounded_f64_to_f32(normalized_laplacian_variance)?,
        face_fraction,
    };
    if components
        .values()
        .iter()
        .chain(
            [
                aggregate,
                measurements.mean_luma,
                measurements.clipped_fraction,
                measurements.percentile_dynamic_range,
                measurements.normalized_laplacian_variance,
                measurements.face_fraction,
            ]
            .iter(),
        )
        .any(|value| !value.is_finite())
    {
        return Err(QualityError::InvalidComputation);
    }
    Ok(QualityReport { aggregate, components, measurements })
}

#[derive(Clone, Copy)]
struct PixelRegion {
    x_start: usize,
    x_end: usize,
    y_start: usize,
    y_end: usize,
}

impl PixelRegion {
    fn from_geometry(
        image_width: u32,
        image_height: u32,
        geometry: FaceGeometry,
    ) -> Result<Self, QualityError> {
        let left = f64::from(geometry.center_x - geometry.width / 2.0) * f64::from(image_width);
        let right = f64::from(geometry.center_x + geometry.width / 2.0) * f64::from(image_width);
        let top = f64::from(geometry.center_y - geometry.height / 2.0) * f64::from(image_height);
        let bottom = f64::from(geometry.center_y + geometry.height / 2.0) * f64::from(image_height);
        let region = Self {
            x_start: bounded_coordinate_to_usize(left.floor(), image_width)?,
            x_end: bounded_coordinate_to_usize(right.ceil(), image_width)?,
            y_start: bounded_coordinate_to_usize(top.floor(), image_height)?,
            y_end: bounded_coordinate_to_usize(bottom.ceil(), image_height)?,
        };
        if region.x_start >= region.x_end || region.y_start >= region.y_end {
            return Err(QualityError::InvalidGeometry);
        }
        Ok(region)
    }

    const fn width(self) -> usize {
        self.x_end - self.x_start
    }

    const fn height(self) -> usize {
        self.y_end - self.y_start
    }
}

struct LumaStatistics {
    histogram: [u32; LUMA_LEVELS],
    luma_sum: u32,
    pixel_count: u32,
}

fn scan_luma(
    image: ImageView<'_>,
    region: PixelRegion,
    should_cancel: &mut impl FnMut() -> bool,
) -> Result<LumaStatistics, QualityError> {
    let mut statistics =
        LumaStatistics { histogram: [0; LUMA_LEVELS], luma_sum: 0, pixel_count: 0 };
    let mut until_cancel_check = 0;
    for y in region.y_start..region.y_end {
        for x in region.x_start..region.x_end {
            if until_cancel_check == 0 {
                check_cancelled(should_cancel)?;
                until_cancel_check = CANCEL_CHECK_PIXELS;
            }
            until_cancel_check -= 1;
            let luma = image.luma(x, y);
            statistics.histogram[usize::from(luma)] += 1;
            statistics.luma_sum += u32::from(luma);
            statistics.pixel_count += 1;
        }
    }
    Ok(statistics)
}

fn scan_laplacian(
    image: ImageView<'_>,
    region: PixelRegion,
    should_cancel: &mut impl FnMut() -> bool,
) -> Result<f64, QualityError> {
    let mut count = 0_u32;
    let mut mean = 0.0_f64;
    let mut sum_squared_deviation = 0.0_f64;
    let mut until_cancel_check = 0;
    for y in (region.y_start + 1)..(region.y_end - 1) {
        for x in (region.x_start + 1)..(region.x_end - 1) {
            if until_cancel_check == 0 {
                check_cancelled(should_cancel)?;
                until_cancel_check = CANCEL_CHECK_PIXELS;
            }
            until_cancel_check -= 1;
            let center = i32::from(image.luma(x, y));
            let laplacian = 4 * center
                - i32::from(image.luma(x - 1, y))
                - i32::from(image.luma(x + 1, y))
                - i32::from(image.luma(x, y - 1))
                - i32::from(image.luma(x, y + 1));
            let laplacian = f64::from(laplacian);
            count += 1;
            let delta = laplacian - mean;
            mean += delta / f64::from(count);
            let delta_after_mean = laplacian - mean;
            sum_squared_deviation += delta * delta_after_mean;
        }
    }
    if count == 0 {
        return Err(QualityError::FaceRegionTooSmall {
            width: region.width(),
            height: region.height(),
        });
    }
    let variance = sum_squared_deviation / f64::from(count);
    if variance.is_finite() && variance >= 0.0 {
        Ok(variance)
    } else {
        Err(QualityError::InvalidComputation)
    }
}

fn histogram_percentile(histogram: &[u32; LUMA_LEVELS], count: u32, percentile: u8) -> u8 {
    let target_rank = (count.saturating_sub(1) * u32::from(percentile)) / 100;
    let mut cumulative = 0_u32;
    for (value, frequency) in histogram.iter().enumerate() {
        cumulative += *frequency;
        if cumulative > target_rank {
            return u8::try_from(value).unwrap_or(u8::MAX);
        }
    }
    u8::MAX
}

fn histogram_count_inclusive(histogram: &[u32; LUMA_LEVELS], first: u8, last: u8) -> u32 {
    histogram[usize::from(first)..=usize::from(last)].iter().sum()
}

fn conservative_aggregate(
    components: QualityComponents,
    weights: QualityWeights,
    weakest_component_headroom: f32,
) -> Result<f32, QualityError> {
    let scores = components.values();
    let weights = weights.values();
    let weakest = scores.iter().copied().fold(1.0_f32, f32::min);
    if weakest == 0.0 {
        return Ok(0.0);
    }
    let mut weight_sum = 0.0_f64;
    let mut reciprocal_sum = 0.0_f64;
    for (score, weight) in scores.into_iter().zip(weights) {
        weight_sum += f64::from(weight);
        reciprocal_sum += f64::from(weight) / f64::from(score);
    }
    let harmonic = weight_sum / reciprocal_sum;
    if !harmonic.is_finite() {
        return Err(QualityError::InvalidComputation);
    }
    Ok(bounded_f64_to_f32(harmonic)?.min((weakest + weakest_component_headroom).min(1.0)))
}

#[allow(clippy::cast_possible_truncation)]
fn bounded_f64_to_f32(value: f64) -> Result<f32, QualityError> {
    if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
        return Err(QualityError::InvalidComputation);
    }
    Ok(value as f32)
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn bounded_coordinate_to_usize(value: f64, dimension: u32) -> Result<usize, QualityError> {
    if !value.is_finite() || value < 0.0 || value > f64::from(dimension) {
        return Err(QualityError::InvalidGeometry);
    }
    Ok(value as usize)
}

fn validate_image_layout(
    width: u32,
    height: u32,
    actual_bytes: usize,
    channels: usize,
) -> Result<(), QualityError> {
    if width == 0
        || height == 0
        || width > ABSOLUTE_MAX_IMAGE_DIMENSION
        || height > ABSOLUTE_MAX_IMAGE_DIMENSION
    {
        return Err(QualityError::InvalidImageDimensions { width, height });
    }
    let pixels =
        (width as usize).checked_mul(height as usize).ok_or(QualityError::DimensionOverflow)?;
    if pixels > ABSOLUTE_MAX_IMAGE_PIXELS {
        return Err(QualityError::AbsoluteImageLimitExceeded {
            actual: pixels,
            maximum: ABSOLUTE_MAX_IMAGE_PIXELS,
        });
    }
    let expected_bytes = pixels.checked_mul(channels).ok_or(QualityError::DimensionOverflow)?;
    if actual_bytes != expected_bytes {
        return Err(QualityError::ImageLengthMismatch {
            expected: expected_bytes,
            actual: actual_bytes,
        });
    }
    Ok(())
}

fn check_cancelled(should_cancel: &mut impl FnMut() -> bool) -> Result<(), QualityError> {
    if should_cancel() { Err(QualityError::Cancelled) } else { Ok(()) }
}

/// Quality assessment failure.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum QualityError {
    /// Image dimensions are empty or exceed the hard per-axis bound.
    #[error("invalid image dimensions {width}x{height}")]
    InvalidImageDimensions {
        /// Supplied width.
        width: u32,
        /// Supplied height.
        height: u32,
    },
    /// Dimension or byte-length arithmetic overflowed.
    #[error("image dimension arithmetic overflowed")]
    DimensionOverflow,
    /// Image pixels exceed the immutable absolute ceiling.
    #[error("image has {actual} pixels, exceeding absolute maximum {maximum}")]
    AbsoluteImageLimitExceeded {
        /// Supplied pixel count.
        actual: usize,
        /// Absolute maximum pixel count.
        maximum: usize,
    },
    /// Image pixels exceed the configured deployment ceiling.
    #[error("image has {actual} pixels, exceeding configured maximum {maximum}")]
    ConfiguredImageLimitExceeded {
        /// Supplied pixel count.
        actual: usize,
        /// Configured maximum pixel count.
        maximum: usize,
    },
    /// The borrowed slice is not the exact tightly packed image length.
    #[error("image requires exactly {expected} bytes but received {actual}")]
    ImageLengthMismatch {
        /// Expected tightly packed byte length.
        expected: usize,
        /// Actual borrowed byte length.
        actual: usize,
    },
    /// Configuration contains a non-finite, contradictory, or excessive setting.
    #[error("invalid quality configuration")]
    InvalidConfig,
    /// Geometry contains a non-finite or out-of-range measurement.
    #[error("invalid face geometry")]
    InvalidGeometry,
    /// The normalized face box crosses the image boundary.
    #[error("face box lies outside the image")]
    FaceBoxOutsideImage,
    /// The face region is too small for bounded spatial measurements.
    #[error("face region {width}x{height} is too small")]
    FaceRegionTooSmall {
        /// Rasterized face width.
        width: usize,
        /// Rasterized face height.
        height: usize,
    },
    /// The owning transaction requested cancellation.
    #[error("quality assessment was cancelled")]
    Cancelled,
    /// A derived computation produced invalid arithmetic.
    #[error("quality computation produced an invalid result")]
    InvalidComputation,
}

#[cfg(test)]
mod tests {
    use super::*;

    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 64;

    fn geometry() -> FaceGeometry {
        FaceGeometry {
            center_x: 0.5,
            center_y: 0.5,
            width: 0.5,
            height: 0.5,
            yaw_degrees: 0.0,
            pitch_degrees: 0.0,
            roll_degrees: 0.0,
            landmark_confidence: 0.95,
            visible_fraction: 0.95,
        }
    }

    fn permissive_config() -> QualityConfig {
        QualityConfig {
            minimum_face_width_pixels: 3,
            minimum_face_height_pixels: 3,
            exposure: BandCalibration {
                reject_below: 0.0,
                ideal_min: 1.0,
                ideal_max: 254.0,
                reject_above: 255.0,
            },
            dark_clip_at_or_below: 0,
            bright_clip_at_or_above: 255,
            clipped_fraction: DeviationCalibration {
                ideal_at_or_below: 0.0,
                reject_at_or_above: 0.5,
            },
            dynamic_range: LowerBoundCalibration {
                reject_at_or_below: 0.0,
                ideal_at_or_above: 1.0,
            },
            sharpness: LowerBoundCalibration { reject_at_or_below: 0.0, ideal_at_or_above: 0.001 },
            face_fraction: BandCalibration {
                reject_below: 0.01,
                ideal_min: 0.02,
                ideal_max: 0.9,
                reject_above: 1.0,
            },
            visibility: LowerBoundCalibration { reject_at_or_below: 0.0, ideal_at_or_above: 0.5 },
            landmark_confidence: LowerBoundCalibration {
                reject_at_or_below: 0.0,
                ideal_at_or_above: 0.5,
            },
            ..QualityConfig::engineering_baseline()
        }
    }

    fn checkerboard() -> Vec<u8> {
        (0..(WIDTH * HEIGHT))
            .map(|index| {
                let x = index % WIDTH;
                let y = index / WIDTH;
                if (x + y).is_multiple_of(2) { 32 } else { 224 }
            })
            .collect()
    }

    #[test]
    fn views_require_exact_tightly_packed_lengths() {
        assert!(matches!(
            Gray8View::new(2, 2, &[0; 3]),
            Err(QualityError::ImageLengthMismatch { expected: 4, actual: 3 })
        ));
        assert!(matches!(
            Rgb8View::new(2, 2, &[0; 11]),
            Err(QualityError::ImageLengthMismatch { expected: 12, actual: 11 })
        ));
        assert!(matches!(
            Gray8View::new(0, 2, &[]),
            Err(QualityError::InvalidImageDimensions { .. })
        ));
    }

    #[test]
    fn image_debug_output_redacts_borrowed_pixels() -> Result<(), QualityError> {
        let secret = [17, 23, 91, 204];
        let gray_debug = format!("{:?}", Gray8View::new(2, 2, &secret)?);
        assert!(gray_debug.contains("<redacted>"));
        assert!(!gray_debug.contains("17, 23, 91, 204"));

        let rgb = [17, 23, 91, 204, 9, 7];
        let rgb_debug = format!("{:?}", Rgb8View::new(2, 1, &rgb)?);
        assert!(rgb_debug.contains("<redacted>"));
        assert!(!rgb_debug.contains("17, 23, 91, 204"));
        Ok(())
    }

    #[test]
    fn views_enforce_absolute_resource_bounds_before_scanning() {
        assert!(matches!(
            Gray8View::new(ABSOLUTE_MAX_IMAGE_DIMENSION + 1, 1, &[]),
            Err(QualityError::InvalidImageDimensions { .. })
        ));
        assert!(matches!(
            Gray8View::new(4_097, 4_097, &[]),
            Err(QualityError::AbsoluteImageLimitExceeded { .. })
        ));
    }

    #[test]
    fn uniform_face_fails_dynamic_range_and_sharpness() -> Result<(), QualityError> {
        let pixels = vec![128; (WIDTH * HEIGHT) as usize];
        let report = assess(
            Gray8View::new(WIDTH, HEIGHT, &pixels)?.into(),
            geometry(),
            QualityConfig::engineering_baseline(),
        )?;
        assert!(report.components.dynamic_range.abs() <= f32::EPSILON);
        assert!(report.components.sharpness.abs() <= f32::EPSILON);
        assert!(report.aggregate.abs() <= f32::EPSILON);
        assert!((report.measurements.mean_luma - 128.0).abs() <= f32::EPSILON);
        Ok(())
    }

    #[test]
    fn textured_well_exposed_face_produces_real_bounded_scores() -> Result<(), QualityError> {
        let pixels = checkerboard();
        let report = assess(
            Gray8View::new(WIDTH, HEIGHT, &pixels)?.into(),
            geometry(),
            QualityConfig::engineering_baseline(),
        )?;
        assert!((report.components.exposure - 1.0).abs() <= f32::EPSILON);
        assert!((report.components.dynamic_range - 1.0).abs() <= f32::EPSILON);
        assert!((report.components.sharpness - 1.0).abs() <= f32::EPSILON);
        assert!(report.aggregate > 0.8 && report.aggregate <= 1.0);
        assert!(report.measurements.normalized_laplacian_variance > 1.0);
        Ok(())
    }

    #[test]
    fn balanced_mean_cannot_hide_severe_shadow_and_highlight_clipping() -> Result<(), QualityError>
    {
        let pixels: Vec<u8> = (0..(WIDTH * HEIGHT))
            .map(|index| if index.is_multiple_of(2) { 0 } else { u8::MAX })
            .collect();
        let report = assess(
            Gray8View::new(WIDTH, HEIGHT, &pixels)?.into(),
            geometry(),
            QualityConfig::engineering_baseline(),
        )?;
        assert!((report.measurements.mean_luma - 127.5).abs() <= f32::EPSILON);
        assert!((report.measurements.clipped_fraction - 1.0).abs() <= f32::EPSILON);
        assert!(report.components.exposure.abs() <= f32::EPSILON);
        assert!(report.aggregate.abs() <= f32::EPSILON);
        Ok(())
    }

    #[test]
    fn rgb_and_gray_views_use_equivalent_integer_luma() -> Result<(), QualityError> {
        let gray = checkerboard();
        let rgb: Vec<u8> = gray.iter().flat_map(|value| [*value, *value, *value]).collect();
        let gray_report =
            assess(Gray8View::new(WIDTH, HEIGHT, &gray)?.into(), geometry(), permissive_config())?;
        let rgb_report =
            assess(Rgb8View::new(WIDTH, HEIGHT, &rgb)?.into(), geometry(), permissive_config())?;
        assert_eq!(gray_report, rgb_report);
        Ok(())
    }

    #[test]
    fn geometry_must_be_finite_bounded_and_inside_image() -> Result<(), QualityError> {
        let pixels = checkerboard();
        let image = Gray8View::new(WIDTH, HEIGHT, &pixels)?.into();
        let mut invalid = geometry();
        invalid.yaw_degrees = f32::NAN;
        assert_eq!(assess(image, invalid, permissive_config()), Err(QualityError::InvalidGeometry));
        let mut outside = geometry();
        outside.center_x = 0.1;
        assert_eq!(
            assess(image, outside, permissive_config()),
            Err(QualityError::FaceBoxOutsideImage)
        );
        Ok(())
    }

    #[test]
    fn configured_image_and_face_region_limits_fail_closed() -> Result<(), QualityError> {
        let pixels = checkerboard();
        let image = Gray8View::new(WIDTH, HEIGHT, &pixels)?.into();
        let mut pixel_limited = permissive_config();
        pixel_limited.max_image_pixels = 1_000;
        assert!(matches!(
            assess(image, geometry(), pixel_limited),
            Err(QualityError::ConfiguredImageLimitExceeded { .. })
        ));
        let mut face_limited = permissive_config();
        face_limited.minimum_face_width_pixels = 48;
        assert!(matches!(
            assess(image, geometry(), face_limited),
            Err(QualityError::FaceRegionTooSmall { .. })
        ));
        Ok(())
    }

    #[test]
    fn cancellation_precedes_validation_and_is_polled_during_scan() -> Result<(), QualityError> {
        let pixels = checkerboard();
        let image = Gray8View::new(WIDTH, HEIGHT, &pixels)?.into();
        let invalid_config =
            QualityConfig { max_image_pixels: 0, ..QualityConfig::engineering_baseline() };
        assert_eq!(
            assess_cancellable(image, geometry(), invalid_config, || true),
            Err(QualityError::Cancelled)
        );

        let large_width = 256;
        let large_height = 256;
        let large = vec![128; (large_width * large_height) as usize];
        let large_image = Gray8View::new(large_width, large_height, &large)?.into();
        let mut polls = 0;
        let result = assess_cancellable(large_image, geometry(), permissive_config(), || {
            polls += 1;
            polls >= 4
        });
        assert_eq!(result, Err(QualityError::Cancelled));
        assert_eq!(polls, 4);
        Ok(())
    }

    #[test]
    fn worst_component_caps_the_harmonic_aggregate() -> Result<(), QualityError> {
        let pixels = checkerboard();
        let mut low_confidence = geometry();
        low_confidence.landmark_confidence = 0.5;
        let config = QualityConfig::engineering_baseline();
        let report =
            assess(Gray8View::new(WIDTH, HEIGHT, &pixels)?.into(), low_confidence, config)?;
        let weakest = report.components.values().into_iter().fold(1.0_f32, f32::min);
        assert!(report.aggregate <= weakest + config.weakest_component_headroom);
        Ok(())
    }

    #[test]
    fn malformed_calibration_is_rejected() {
        let invalid = QualityConfig {
            weakest_component_headroom: f32::NAN,
            ..QualityConfig::engineering_baseline()
        };
        assert_eq!(invalid.validate(), Err(QualityError::InvalidConfig));
        let zero_weight = QualityConfig {
            weights: QualityWeights {
                exposure: 0.0,
                ..QualityConfig::engineering_baseline().weights
            },
            ..QualityConfig::engineering_baseline()
        };
        assert_eq!(zero_weight.validate(), Err(QualityError::InvalidConfig));
        let impossible_face_region = QualityConfig {
            max_image_pixels: 1_000,
            minimum_face_width_pixels: 40,
            minimum_face_height_pixels: 40,
            ..QualityConfig::engineering_baseline()
        };
        assert_eq!(impossible_face_region.validate(), Err(QualityError::InvalidConfig));
    }

    #[test]
    fn all_successful_outputs_are_finite_and_bounded() -> Result<(), QualityError> {
        for offset in 0_u8..=15 {
            let pixels: Vec<u8> = (0..(WIDTH * HEIGHT))
                .map(|index| {
                    u8::try_from((index * 37 + u32::from(offset)) % 256).unwrap_or(u8::MAX)
                })
                .collect();
            let report = assess(
                Gray8View::new(WIDTH, HEIGHT, &pixels)?.into(),
                geometry(),
                permissive_config(),
            )?;
            assert!(report.aggregate.is_finite() && (0.0..=1.0).contains(&report.aggregate));
            assert!(
                report
                    .components
                    .values()
                    .iter()
                    .all(|score| score.is_finite() && (0.0..=1.0).contains(score))
            );
        }
        Ok(())
    }
}
