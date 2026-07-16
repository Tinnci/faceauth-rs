//! Peer-credentialed, length-bounded JSON framing for the local authentication hot path.

use std::{
    fs,
    io::{self, Read, Write},
    os::fd::{AsFd, BorrowedFd, OwnedFd},
    os::unix::net::UnixStream,
    os::unix::{
        fs::{FileTypeExt, MetadataExt, PermissionsExt},
        net::UnixListener,
    },
    path::{Path, PathBuf},
    thread,
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

/// Root-owned Unix listener whose path is never implicitly replaced or removed.
pub struct SecureListener {
    listener: UnixListener,
    path: PathBuf,
    identity: SocketIdentity,
    cleaned: bool,
}

#[derive(Clone, Copy)]
struct SocketIdentity {
    device: u64,
    inode: u64,
}

impl SecureListener {
    /// Bind a socket inside an existing root-owned directory that is not writable by group or
    /// others.
    ///
    /// Existing socket or filesystem entries are never unlinked automatically. The supported
    /// socket modes are `0600` and `0660`; group ownership is configured by the service manager.
    ///
    /// # Errors
    ///
    /// Returns [`ListenerError`] for invalid mode, missing/unsafe parent directory, an existing
    /// path, bind failure, or permission update failure.
    pub fn bind_root_owned(path: impl AsRef<Path>, mode: u32) -> Result<Self, ListenerError> {
        Self::bind_owned(path.as_ref(), mode, 0)
    }

    fn bind_owned(path: &Path, mode: u32, required_uid: u32) -> Result<Self, ListenerError> {
        if !matches!(mode, 0o600 | 0o660) {
            return Err(ListenerError::InvalidSocketMode { mode });
        }
        let parent = path.parent().ok_or(ListenerError::MissingParent)?;
        let metadata = fs::symlink_metadata(parent)?;
        if metadata.file_type().is_symlink()
            || !metadata.is_dir()
            || metadata.uid() != required_uid
            || metadata.permissions().mode() & 0o022 != 0
        {
            return Err(ListenerError::UnsafeParent { path: parent.to_owned() });
        }
        match fs::symlink_metadata(path) {
            Ok(_) => return Err(ListenerError::PathExists { path: path.to_owned() }),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(ListenerError::Io(error)),
        }
        let listener = UnixListener::bind(path)?;
        if let Err(error) = fs::set_permissions(path, fs::Permissions::from_mode(mode)) {
            let _ = fs::remove_file(path);
            return Err(ListenerError::Io(error));
        }
        let metadata = fs::symlink_metadata(path)?;
        if !metadata.file_type().is_socket() {
            let _ = fs::remove_file(path);
            return Err(ListenerError::CreatedPathInvalid { path: path.to_owned() });
        }
        Ok(Self {
            listener,
            path: path.to_owned(),
            identity: SocketIdentity { device: metadata.dev(), inode: metadata.ino() },
            cleaned: false,
        })
    }

    /// Accept one peer and immediately capture credentials, pidfd, and I/O bounds.
    ///
    /// # Errors
    ///
    /// Returns [`TransportError`] when accept or connected-stream initialization fails.
    pub fn accept(&self, config: TransportConfig) -> Result<PeerStream, TransportError> {
        let (stream, _address) = self.listener.accept()?;
        PeerStream::connect(stream, config)
    }

    /// Switch the listener between blocking and non-blocking accept behavior.
    ///
    /// # Errors
    ///
    /// Returns [`TransportError`] when the socket option cannot be changed.
    pub fn set_nonblocking(&self, nonblocking: bool) -> Result<(), TransportError> {
        self.listener.set_nonblocking(nonblocking)?;
        Ok(())
    }

    /// Attempt one non-blocking accept.
    ///
    /// # Errors
    ///
    /// Returns [`TransportError`] when accept or peer credential/pidfd initialization fails.
    pub fn try_accept(
        &self,
        config: TransportConfig,
    ) -> Result<Option<PeerStream>, TransportError> {
        match self.listener.accept() {
            Ok((stream, _address)) => PeerStream::connect(stream, config).map(Some),
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => Ok(None),
            Err(error) => Err(TransportError::Io(error)),
        }
    }

    /// Return the bound filesystem path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Remove only the exact socket inode created by this listener.
    ///
    /// A missing path is treated as already cleaned. A replaced path is never removed. Drop also
    /// attempts this identity-checked cleanup so unwinding cannot leave an ordinary stale socket;
    /// callers use this method when cleanup failure must be reported.
    ///
    /// # Errors
    ///
    /// Returns [`ListenerError::OwnedPathChanged`] if the path now identifies another inode or
    /// file type, or [`ListenerError::Io`] for a filesystem failure.
    pub fn cleanup(mut self) -> Result<(), ListenerError> {
        self.remove_owned_path()?;
        self.cleaned = true;
        Ok(())
    }

    fn remove_owned_path(&self) -> Result<(), ListenerError> {
        match fs::symlink_metadata(&self.path) {
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(ListenerError::Io(error)),
            Ok(metadata)
                if metadata.file_type().is_socket()
                    && metadata.dev() == self.identity.device
                    && metadata.ino() == self.identity.inode =>
            {
                fs::remove_file(&self.path).map_err(ListenerError::Io)
            }
            Ok(_) => Err(ListenerError::OwnedPathChanged { path: self.path.clone() }),
        }
    }
}

impl Drop for SecureListener {
    fn drop(&mut self) {
        if !self.cleaned {
            let _cleanup_result = self.remove_owned_path();
        }
    }
}

/// Bounds for a sequential non-blocking listener loop.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AcceptLoopConfig {
    /// Delay between empty non-blocking accept attempts.
    pub poll_interval: Duration,
    /// Maximum successfully initialized connections handled before returning.
    pub max_connections: u64,
    /// Maximum consecutive accept/peer-initialization failures tolerated.
    pub max_consecutive_failures: u16,
}

impl AcceptLoopConfig {
    /// Validate listener-loop resource bounds.
    ///
    /// # Errors
    ///
    /// Returns [`TransportError::InvalidAcceptLoopConfig`] for zero or excessive bounds.
    pub fn validate(self) -> Result<(), TransportError> {
        if !(Duration::from_millis(1)..=Duration::from_millis(100)).contains(&self.poll_interval)
            || !(1..=1_000_000).contains(&self.max_connections)
            || !(1..=1_000).contains(&self.max_consecutive_failures)
        {
            Err(TransportError::InvalidAcceptLoopConfig)
        } else {
            Ok(())
        }
    }
}

impl Default for AcceptLoopConfig {
    fn default() -> Self {
        Self {
            poll_interval: Duration::from_millis(25),
            max_connections: 1_000_000,
            max_consecutive_failures: 32,
        }
    }
}

/// Summary returned when a bounded listener loop stops normally.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AcceptLoopReport {
    /// Successfully peer-credentialed connections passed to the handler.
    pub accepted_connections: u64,
    /// Failed accept or peer-initialization attempts tolerated by the loop.
    pub initialization_failures: u64,
}

/// Run a sequential, stoppable accept loop over already-secured listener and stream policies.
///
/// The handler receives only streams whose kernel credentials and pidfd were captured and whose
/// framing I/O is bounded. Handler error and panic policy remains the daemon's responsibility.
/// Sequential delivery intentionally matches the initial single-camera session capacity.
///
/// # Errors
///
/// Returns [`TransportError`] for invalid configuration, listener mode failure, or excessive
/// consecutive accept/peer-initialization failures.
pub fn run_accept_loop(
    listener: &SecureListener,
    transport: TransportConfig,
    config: AcceptLoopConfig,
    mut should_stop: impl FnMut() -> bool,
    mut handle: impl FnMut(PeerStream),
) -> Result<AcceptLoopReport, TransportError> {
    transport.validate()?;
    config.validate()?;
    listener.set_nonblocking(true)?;
    let mut report = AcceptLoopReport { accepted_connections: 0, initialization_failures: 0 };
    let mut consecutive_failures = 0_u16;
    while report.accepted_connections < config.max_connections && !should_stop() {
        match listener.try_accept(transport) {
            Ok(Some(stream)) => {
                report.accepted_connections += 1;
                consecutive_failures = 0;
                handle(stream);
            }
            Ok(None) => thread::sleep(config.poll_interval),
            Err(_) => {
                report.initialization_failures += 1;
                consecutive_failures = consecutive_failures
                    .checked_add(1)
                    .ok_or(TransportError::AcceptFailureBudgetExceeded)?;
                if consecutive_failures >= config.max_consecutive_failures {
                    return Err(TransportError::AcceptFailureBudgetExceeded);
                }
            }
        }
    }
    Ok(report)
}

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
    /// Listener-loop polling or resource bounds are invalid.
    #[error("invalid local listener-loop configuration")]
    InvalidAcceptLoopConfig,
    /// Consecutive accept or peer-initialization failures exceeded policy.
    #[error("local listener accept failure budget exceeded")]
    AcceptFailureBudgetExceeded,
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

/// Secure Unix-listener creation failure.
#[derive(Debug, Error)]
pub enum ListenerError {
    /// Socket permissions must be owner-only or owner/group-only.
    #[error("unsupported Unix socket mode {mode:o}")]
    InvalidSocketMode {
        /// Requested permission bits.
        mode: u32,
    },
    /// Socket path has no parent directory.
    #[error("Unix socket path has no parent directory")]
    MissingParent,
    /// Parent is a symlink, not a directory, has wrong ownership, or permits untrusted writes.
    #[error("Unix socket parent directory is unsafe: {path}")]
    UnsafeParent {
        /// Rejected parent path.
        path: PathBuf,
    },
    /// Listener refuses to replace any existing filesystem entry.
    #[error("Unix socket path already exists: {path}")]
    PathExists {
        /// Existing path.
        path: PathBuf,
    },
    /// The kernel-created path did not resolve to a Unix socket.
    #[error("created Unix listener path is not a socket: {path}")]
    CreatedPathInvalid {
        /// Invalid created path.
        path: PathBuf,
    },
    /// Cleanup refused to remove a path whose type or inode no longer matches this listener.
    #[error("owned Unix socket path changed before cleanup: {path}")]
    OwnedPathChanged {
        /// Replaced path.
        path: PathBuf,
    },
    /// Filesystem or socket operation failed.
    #[error("unable to create secure Unix listener: {0}")]
    Io(#[from] io::Error),
}

#[cfg(test)]
mod tests {
    use std::{
        cell::Cell,
        fs,
        os::unix::{fs::PermissionsExt, net::UnixStream},
    };

    use faceauth_protocol::{
        AuthenticationPurpose, Request, RequestContext, ServiceName, TransactionId,
    };
    use nix::sys::socket::UnixCredentials;

    use super::*;

    fn temporary_directory() -> Result<PathBuf, Box<dyn std::error::Error>> {
        let path = std::env::temp_dir()
            .join(format!("faceauth-transport-{}", faceauth_protocol::TransactionId::generate()));
        fs::create_dir(&path)?;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700))?;
        Ok(path)
    }

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
    fn accept_loop_configuration_is_bounded() {
        assert!(AcceptLoopConfig::default().validate().is_ok());
        assert!(
            AcceptLoopConfig { poll_interval: Duration::ZERO, ..AcceptLoopConfig::default() }
                .validate()
                .is_err()
        );
        assert!(
            AcceptLoopConfig { max_connections: 0, ..AcceptLoopConfig::default() }
                .validate()
                .is_err()
        );
        assert!(
            AcceptLoopConfig { max_consecutive_failures: 0, ..AcceptLoopConfig::default() }
                .validate()
                .is_err()
        );
    }

    #[test]
    fn bounded_accept_loop_delivers_only_initialized_peer_streams()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = temporary_directory()?;
        let socket = directory.join("auth.sock");
        let uid = fs::metadata(&directory)?.uid();
        let listener = SecureListener::bind_owned(&socket, 0o600, uid)?;
        let client = UnixStream::connect(&socket)?;
        let mut client = PeerStream::connect(client, TransportConfig::default())?;
        let request = request()?;
        client.write_message(request.clone())?;
        let received = Cell::new(false);
        let report = run_accept_loop(
            &listener,
            TransportConfig::default(),
            AcceptLoopConfig {
                poll_interval: Duration::from_millis(1),
                max_connections: 1,
                max_consecutive_failures: 2,
            },
            || false,
            |mut stream| {
                let actual = stream.read_message::<Request>().map(|envelope| envelope.message);
                received.set(matches!(actual, Ok(actual) if actual == request));
            },
        )?;
        assert!(received.get());
        assert_eq!(
            report,
            AcceptLoopReport { accepted_connections: 1, initialization_failures: 0 }
        );
        fs::remove_file(&socket)?;
        fs::remove_dir(&directory)?;
        Ok(())
    }

    #[test]
    fn accept_loop_can_stop_without_a_connection() -> Result<(), Box<dyn std::error::Error>> {
        let directory = temporary_directory()?;
        let socket = directory.join("auth.sock");
        let uid = fs::metadata(&directory)?.uid();
        let listener = SecureListener::bind_owned(&socket, 0o600, uid)?;
        let polls = Cell::new(0_u8);
        let report = run_accept_loop(
            &listener,
            TransportConfig::default(),
            AcceptLoopConfig {
                poll_interval: Duration::from_millis(1),
                max_connections: 1,
                max_consecutive_failures: 2,
            },
            || {
                let next = polls.get().saturating_add(1);
                polls.set(next);
                next >= 3
            },
            |_stream| {},
        )?;
        assert_eq!(report.accepted_connections, 0);
        fs::remove_file(&socket)?;
        fs::remove_dir(&directory)?;
        Ok(())
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

    #[test]
    fn listener_requires_safe_parent_and_never_replaces_paths()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = temporary_directory()?;
        let uid = fs::metadata(&directory)?.uid();
        let socket = directory.join("auth.sock");
        let listener = SecureListener::bind_owned(&socket, 0o600, uid)?;
        assert_eq!(fs::metadata(listener.path())?.permissions().mode() & 0o777, 0o600);
        assert!(matches!(
            SecureListener::bind_owned(&socket, 0o600, uid),
            Err(ListenerError::PathExists { .. })
        ));
        drop(listener);
        assert!(!socket.exists());

        fs::set_permissions(&directory, fs::Permissions::from_mode(0o777))?;
        assert!(matches!(
            SecureListener::bind_owned(&socket, 0o600, uid),
            Err(ListenerError::UnsafeParent { .. })
        ));
        fs::remove_dir(&directory)?;
        Ok(())
    }

    #[test]
    fn listener_cleanup_removes_only_its_exact_socket_inode()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = temporary_directory()?;
        let uid = fs::metadata(&directory)?.uid();
        let socket = directory.join("auth.sock");
        let listener = SecureListener::bind_owned(&socket, 0o600, uid)?;
        listener.cleanup()?;
        assert!(!socket.exists());

        let listener = SecureListener::bind_owned(&socket, 0o600, uid)?;
        fs::remove_file(&socket)?;
        fs::write(&socket, b"replacement")?;
        assert!(matches!(listener.cleanup(), Err(ListenerError::OwnedPathChanged { .. })));
        assert_eq!(fs::read(&socket)?, b"replacement");

        fs::remove_file(&socket)?;
        fs::remove_dir(&directory)?;
        Ok(())
    }

    #[test]
    fn secure_listener_accepts_a_peer_credentialed_stream() -> Result<(), Box<dyn std::error::Error>>
    {
        let directory = temporary_directory()?;
        let uid = fs::metadata(&directory)?.uid();
        let socket = directory.join("auth.sock");
        let listener = SecureListener::bind_owned(&socket, 0o600, uid)?;
        let client = UnixStream::connect(&socket)?;
        let server = listener.accept(TransportConfig::default())?;

        assert_eq!(server.peer().uid, UnixCredentials::new().uid());
        drop(client);
        drop(server);
        drop(listener);
        assert!(!socket.exists());
        fs::remove_dir(&directory)?;
        Ok(())
    }
}
