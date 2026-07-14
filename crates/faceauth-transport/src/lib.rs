//! Peer-credentialed, length-bounded JSON framing for the local authentication hot path.

use std::{
    io::{self, Read, Write},
    os::fd::{AsFd, BorrowedFd, OwnedFd},
    os::unix::net::UnixStream,
    time::Duration,
};

use faceauth_protocol::{Envelope, ProtocolError};
use nix::sys::socket::{
    getsockopt,
    sockopt::{PeerCredentials, PeerPidfd},
};
use serde::{Serialize, de::DeserializeOwned};
use thiserror::Error;

/// Default maximum encoded message size. Protocol messages never contain image or template data.
pub const DEFAULT_MAX_MESSAGE_BYTES: usize = 16 * 1024;

/// Hard ceiling preventing configuration from turning the protocol into a bulk-data channel.
pub const ABSOLUTE_MAX_MESSAGE_BYTES: usize = 64 * 1024;

/// Bounded blocking-I/O configuration for one connected socket.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransportConfig {
    /// Maximum JSON payload following the four-byte length prefix.
    pub max_message_bytes: usize,
    /// Read and write timeout applied to the connected Unix stream.
    pub io_timeout: Duration,
}

impl TransportConfig {
    /// Validate transport resource bounds.
    ///
    /// # Errors
    ///
    /// Returns [`TransportError::InvalidConfig`] for zero, excessive, or unbounded settings.
    pub fn validate(self) -> Result<(), TransportError> {
        if !(1..=ABSOLUTE_MAX_MESSAGE_BYTES).contains(&self.max_message_bytes)
            || !(Duration::from_millis(1)..=Duration::from_secs(30)).contains(&self.io_timeout)
        {
            return Err(TransportError::InvalidConfig);
        }
        Ok(())
    }
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self { max_message_bytes: DEFAULT_MAX_MESSAGE_BYTES, io_timeout: Duration::from_secs(2) }
    }
}

/// Kernel-provided identity fixed to a connected Unix socket.
///
/// These values identify the peer but do not authorize any service, purpose, or target UID.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PeerIdentity {
    /// Peer process identifier at connection time.
    pub pid: u32,
    /// Peer effective user identifier at connection time.
    pub uid: u32,
    /// Peer effective group identifier at connection time.
    pub gid: u32,
}

/// Connected local stream with captured peer credentials and bounded framing.
pub struct PeerStream {
    stream: UnixStream,
    peer: PeerIdentity,
    peer_pidfd: OwnedFd,
    config: TransportConfig,
}

impl PeerStream {
    /// Capture `SO_PEERCRED` and apply blocking I/O bounds before reading any caller bytes.
    ///
    /// # Errors
    ///
    /// Returns [`TransportError`] when configuration, peer credentials, or socket timeouts fail.
    pub fn connect(stream: UnixStream, config: TransportConfig) -> Result<Self, TransportError> {
        config.validate()?;
        let credentials = getsockopt(&stream, PeerCredentials)?;
        let pid = u32::try_from(credentials.pid()).map_err(|_| TransportError::InvalidPeer)?;
        let peer_pidfd =
            getsockopt(&stream, PeerPidfd).map_err(TransportError::PeerPidfdUnavailable)?;
        stream.set_read_timeout(Some(config.io_timeout))?;
        stream.set_write_timeout(Some(config.io_timeout))?;
        Ok(Self {
            stream,
            peer: PeerIdentity { pid, uid: credentials.uid(), gid: credentials.gid() },
            peer_pidfd,
            config,
        })
    }

    /// Return immutable kernel peer identity for daemon authorization policy.
    #[must_use]
    pub const fn peer(&self) -> PeerIdentity {
        self.peer
    }

    /// Borrow the kernel pidfd pinning the peer process against PID reuse.
    #[must_use]
    pub fn peer_pidfd(&self) -> BorrowedFd<'_> {
        self.peer_pidfd.as_fd()
    }

    /// Read, decode, and version-check one envelope.
    ///
    /// The declared size is rejected before allocation when it exceeds policy.
    ///
    /// # Errors
    ///
    /// Returns [`TransportError`] for I/O failure, empty or oversized frames, malformed JSON, or
    /// unsupported protocol versions.
    pub fn read_message<T: DeserializeOwned>(&mut self) -> Result<Envelope<T>, TransportError> {
        let mut header = [0_u8; 4];
        self.stream.read_exact(&mut header)?;
        let declared = usize::try_from(u32::from_be_bytes(header)).map_err(|_| {
            TransportError::MessageTooLarge {
                declared: usize::MAX,
                maximum: self.config.max_message_bytes,
            }
        })?;
        if declared == 0 {
            return Err(TransportError::EmptyMessage);
        }
        if declared > self.config.max_message_bytes {
            return Err(TransportError::MessageTooLarge {
                declared,
                maximum: self.config.max_message_bytes,
            });
        }
        let mut encoded = vec![0_u8; declared];
        self.stream.read_exact(&mut encoded)?;
        let envelope: Envelope<T> = serde_json::from_slice(&encoded)?;
        envelope.validate_version()?;
        Ok(envelope)
    }

    /// Encode and write one current-version envelope.
    ///
    /// # Errors
    ///
    /// Returns [`TransportError`] when serialization fails, the payload exceeds policy, or the
    /// socket write fails.
    pub fn write_message<T: Serialize>(&mut self, message: T) -> Result<(), TransportError> {
        let encoded = serde_json::to_vec(&Envelope::current(message))?;
        if encoded.is_empty() {
            return Err(TransportError::EmptyMessage);
        }
        if encoded.len() > self.config.max_message_bytes {
            return Err(TransportError::MessageTooLarge {
                declared: encoded.len(),
                maximum: self.config.max_message_bytes,
            });
        }
        let length = u32::try_from(encoded.len()).map_err(|_| TransportError::MessageTooLarge {
            declared: encoded.len(),
            maximum: self.config.max_message_bytes,
        })?;
        self.stream.write_all(&length.to_be_bytes())?;
        self.stream.write_all(&encoded)?;
        self.stream.flush()?;
        Ok(())
    }
}

/// Local transport failure.
#[derive(Debug, Error)]
pub enum TransportError {
    /// Configured frame or timeout bounds are invalid.
    #[error("invalid local transport configuration")]
    InvalidConfig,
    /// Kernel peer credentials contained an invalid process identifier.
    #[error("invalid Unix peer credentials")]
    InvalidPeer,
    /// A zero-length message is not a protocol frame.
    #[error("empty local protocol message")]
    EmptyMessage,
    /// The declared or encoded message exceeds the configured bound.
    #[error("local protocol message length {declared} exceeds {maximum}")]
    MessageTooLarge {
        /// Declared or encoded length.
        declared: usize,
        /// Configured maximum length.
        maximum: usize,
    },
    /// Unix socket I/O failed or timed out.
    #[error("local transport I/O failed: {0}")]
    Io(#[from] io::Error),
    /// `SO_PEERCRED` lookup failed.
    #[error("unable to read Unix peer credentials: {0}")]
    PeerCredentials(#[from] nix::errno::Errno),
    /// The kernel could not provide a pidfd for the connected peer.
    #[error("unable to pin Unix peer process with SO_PEERPIDFD: {0}")]
    PeerPidfdUnavailable(nix::errno::Errno),
    /// JSON did not match the closed protocol schema.
    #[error("invalid local protocol encoding: {0}")]
    Json(#[from] serde_json::Error),
    /// The peer used an unsupported protocol version.
    #[error("invalid local protocol version: {0}")]
    Protocol(#[from] ProtocolError),
}

#[cfg(test)]
mod tests {
    use std::os::unix::net::UnixStream;

    use faceauth_protocol::{
        AuthenticationPurpose, Request, RequestContext, ServiceName, TransactionId,
    };
    use nix::sys::socket::UnixCredentials;

    use super::*;

    fn request() -> Result<Request, Box<dyn std::error::Error>> {
        Ok(Request::Authenticate {
            context: RequestContext {
                transaction_id: TransactionId::generate(),
                target_uid: 1000,
                service: ServiceName::parse("faceauth-test")?,
                purpose: AuthenticationPurpose::Test,
            },
        })
    }

    #[test]
    fn peer_identity_comes_from_the_kernel() -> Result<(), Box<dyn std::error::Error>> {
        let (left, _right) = UnixStream::pair()?;
        let stream = PeerStream::connect(left, TransportConfig::default())?;
        let current = UnixCredentials::new();

        assert_eq!(
            stream.peer(),
            PeerIdentity {
                pid: u32::try_from(current.pid())?,
                uid: current.uid(),
                gid: current.gid(),
            }
        );
        Ok(())
    }

    #[test]
    fn current_version_round_trips_with_a_length_prefix() -> Result<(), Box<dyn std::error::Error>>
    {
        let (left, right) = UnixStream::pair()?;
        let mut writer = PeerStream::connect(left, TransportConfig::default())?;
        let mut reader = PeerStream::connect(right, TransportConfig::default())?;
        let expected = request()?;

        writer.write_message(expected.clone())?;
        let actual: Envelope<Request> = reader.read_message()?;

        assert_eq!(actual, Envelope::current(expected));
        Ok(())
    }

    #[test]
    fn oversized_length_is_rejected_before_payload_read() -> Result<(), Box<dyn std::error::Error>>
    {
        let (mut writer, reader) = UnixStream::pair()?;
        let config = TransportConfig { max_message_bytes: 32, ..TransportConfig::default() };
        let mut reader = PeerStream::connect(reader, config)?;
        writer.write_all(&33_u32.to_be_bytes())?;

        assert!(matches!(
            reader.read_message::<Request>(),
            Err(TransportError::MessageTooLarge { declared: 33, maximum: 32 })
        ));
        Ok(())
    }

    #[test]
    fn empty_and_malformed_frames_fail_closed() -> Result<(), Box<dyn std::error::Error>> {
        let (mut empty_writer, empty_reader) = UnixStream::pair()?;
        let mut empty_reader = PeerStream::connect(empty_reader, TransportConfig::default())?;
        empty_writer.write_all(&0_u32.to_be_bytes())?;
        assert!(matches!(
            empty_reader.read_message::<Request>(),
            Err(TransportError::EmptyMessage)
        ));

        let (mut bad_writer, bad_reader) = UnixStream::pair()?;
        let mut bad_reader = PeerStream::connect(bad_reader, TransportConfig::default())?;
        bad_writer.write_all(&3_u32.to_be_bytes())?;
        bad_writer.write_all(b"bad")?;
        assert!(matches!(bad_reader.read_message::<Request>(), Err(TransportError::Json(_))));
        Ok(())
    }

    #[test]
    fn unsupported_version_fails_before_reaching_the_daemon()
    -> Result<(), Box<dyn std::error::Error>> {
        let (mut writer, reader) = UnixStream::pair()?;
        let mut reader = PeerStream::connect(reader, TransportConfig::default())?;
        let encoded = serde_json::to_vec(&Envelope {
            protocol_version: faceauth_protocol::PROTOCOL_VERSION + 1,
            message: request()?,
        })?;
        writer.write_all(&u32::try_from(encoded.len())?.to_be_bytes())?;
        writer.write_all(&encoded)?;

        assert!(matches!(
            reader.read_message::<Request>(),
            Err(TransportError::Protocol(ProtocolError::UnsupportedVersion { .. }))
        ));
        Ok(())
    }

    #[test]
    fn outbound_encoding_cannot_exceed_the_configured_channel()
    -> Result<(), Box<dyn std::error::Error>> {
        let (writer, _reader) = UnixStream::pair()?;
        let config = TransportConfig { max_message_bytes: 8, ..TransportConfig::default() };
        let mut writer = PeerStream::connect(writer, config)?;

        assert!(matches!(
            writer.write_message("larger-than-eight-bytes"),
            Err(TransportError::MessageTooLarge { maximum: 8, .. })
        ));
        Ok(())
    }

    #[test]
    fn configured_read_timeout_bounds_an_idle_peer() -> Result<(), Box<dyn std::error::Error>> {
        let (_idle_peer, reader) = UnixStream::pair()?;
        let config =
            TransportConfig { io_timeout: Duration::from_millis(50), ..TransportConfig::default() };
        let mut reader = PeerStream::connect(reader, config)?;

        let error =
            reader.read_message::<Request>().err().ok_or("idle read unexpectedly passed")?;
        assert!(matches!(
            error,
            TransportError::Io(ref io_error)
                if matches!(io_error.kind(), io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock)
        ));
        Ok(())
    }
}
