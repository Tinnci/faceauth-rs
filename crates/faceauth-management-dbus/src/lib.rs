//! Thin D-Bus transport adapter for the stable Manager1 management contract.
//!
//! The adapter provides asynchronous system-bus sender credential lookup and the narrow `PolicyKit`
//! enrollment check. A daemon still supplies authenticated-template state and a dedicated root
//! grant issuer before the adapter can return an [`AuthorizedEnrollment`] proof. This keeps the
//! security boundary explicit and keeps the core management crate independent of D-Bus.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use faceauth_authz::AuthorizationGrant;
use faceauth_management::{
    AuthorizedEnrollment, ENROLLMENT_POLKIT_ACTION, MANAGEMENT_SCHEMA_VERSION,
    ManagementCoordinator, ManagementError, ManagementProgress, ManagementResult, ManagementUpdate,
    OperationId,
};
use zbus::{Connection, interface, message::Header, object_server::SignalContext, zvariant::Value};

/// Object-safe asynchronous result returned by management backends.
pub type BackendFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, BackendError>> + Send + 'a>>;

/// Authorization and caller-identity operations required by the D-Bus boundary.
pub trait AuthorizationBackend: Send + Sync + 'static {
    /// Resolve the Unix UID for a unique D-Bus sender name.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when credentials cannot be resolved or are not trusted.
    fn caller_uid<'a>(&'a self, sender: &'a str) -> BackendFuture<'a, u32>;

    /// Return whether the exact target account has an enrolled template.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when template state cannot be read safely.
    fn enrollment_state<'a>(&'a self, sender: &'a str, target_uid: u32) -> BackendFuture<'a, bool>;

    /// Perform the Polkit check and return the exact authorization proof accepted by the core.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when authorization fails closed.
    fn authorize_enrollment<'a>(
        &'a self,
        sender: &'a str,
        target_uid: u32,
    ) -> BackendFuture<'a, AuthorizedEnrollment>;
}

/// Encrypted-template state lookup used by the system-bus backend.
pub trait EnrollmentStateSource: Send + Sync + 'static {
    /// Return whether the target UID has an authenticated encrypted template.
    fn is_enrolled(&self, target_uid: u32) -> BackendFuture<'_, bool>;
}

/// Dedicated root-broker grant issuer supplied by the privileged daemon.
pub trait EnrollmentGrantIssuer: Send + Sync + 'static {
    /// Issue an exact root `faceauth-enroll`/Polkit grant for the authorized target UID.
    fn issue_grant<'a>(
        &'a self,
        sender: &'a str,
        target_uid: u32,
    ) -> BackendFuture<'a, AuthorizationGrant>;
}

/// Credential and `PolicyKit` operations used by [`SystemBusBackend`].
pub trait SystemAuthority: Send + Sync + 'static {
    /// Resolve the exact unique sender's Unix UID.
    fn caller_uid<'a>(&'a self, sender: &'a str) -> BackendFuture<'a, u32>;

    /// Obtain a fresh `PolicyKit` enrollment authorization for the exact unique sender.
    fn authorize_enrollment<'a>(&'a self, sender: &'a str) -> BackendFuture<'a, ()>;
}

/// Monotonic clock used for operation deadlines.
pub trait MonotonicClock: Send + Sync + 'static {
    /// Return monotonic microseconds.
    fn now_micros(&self) -> u64;
}

/// Process-monotonic clock suitable for the daemon.
#[derive(Debug, Default)]
pub struct SystemMonotonicClock;

impl MonotonicClock for SystemMonotonicClock {
    fn now_micros(&self) -> u64 {
        static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
        u64::try_from(START.get_or_init(Instant::now).elapsed().as_micros()).unwrap_or(u64::MAX)
    }
}

/// Backend or caller identity failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendError(String);

impl BackendError {
    /// Construct a fail-closed backend error with a non-sensitive public message.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for BackendError {}

const POLKIT_SERVICE: &str = "org.freedesktop.PolicyKit1";
const POLKIT_PATH: &str = "/org/freedesktop/PolicyKit1/Authority";
const POLKIT_INTERFACE: &str = "org.freedesktop.PolicyKit1.Authority";
const POLKIT_SUBJECT_SYSTEM_BUS_NAME: &str = "system-bus-name";
const POLKIT_ALLOW_USER_INTERACTION: u32 = 1;

/// Minimal `PolicyKit` Authority proxy used for enrollment authorization.
#[zbus::proxy(
    interface = "org.freedesktop.PolicyKit1.Authority",
    default_service = "org.freedesktop.PolicyKit1",
    default_path = "/org/freedesktop/PolicyKit1/Authority",
    gen_blocking = false
)]
trait PolicyKitAuthority {
    /// Check one action for a system-bus subject.
    fn check_authorization(
        &self,
        subject: (&str, HashMap<&str, Value<'_>>),
        action_id: &str,
        details: HashMap<&str, &str>,
        flags: u32,
        cancellation_id: &str,
    ) -> zbus::Result<(bool, bool, HashMap<String, String>)>;
}

/// System-bus credential and `PolicyKit` authority client.
#[derive(Clone)]
pub struct SystemBusAuthority {
    connection: Connection,
}

impl SystemBusAuthority {
    /// Connect to the system bus.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when the system bus is unavailable.
    pub async fn connect() -> Result<Self, BackendError> {
        Connection::system()
            .await
            .map(Self::from_connection)
            .map_err(|error| BackendError::new(format!("unable to connect to system bus: {error}")))
    }

    /// Construct from an existing system-bus connection owned by the daemon.
    #[must_use]
    pub const fn from_connection(connection: Connection) -> Self {
        Self { connection }
    }

    /// Resolve the Unix UID attached by the bus to an exact unique sender name.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] for malformed/non-unique names or failed credential lookup.
    pub async fn caller_uid(&self, sender: &str) -> Result<u32, BackendError> {
        let sender = unique_sender(sender)?;
        let proxy = zbus::fdo::DBusProxy::new(&self.connection)
            .await
            .map_err(|error| BackendError::new(format!("unable to create D-Bus proxy: {error}")))?;
        proxy.get_connection_unix_user(sender.into()).await.map_err(|error| {
            BackendError::new(format!("unable to resolve D-Bus sender UID: {error}"))
        })
    }

    /// Request a fresh interactive `PolicyKit` decision for enrollment.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] unless `PolicyKit` returns an authorized result for the exact unique
    /// sender and `org.faceauth.enroll` action.
    pub async fn authorize_enrollment(&self, sender: &str) -> Result<(), BackendError> {
        let sender = unique_sender(sender)?;
        let proxy = PolicyKitAuthorityProxy::new(&self.connection).await.map_err(|error| {
            BackendError::new(format!("unable to create PolicyKit authority proxy: {error}"))
        })?;
        let mut subject_details = HashMap::new();
        subject_details.insert("name", Value::from(sender.as_str()));
        let subject = (POLKIT_SUBJECT_SYSTEM_BUS_NAME, subject_details);
        let (authorized, _, _) = proxy
            .check_authorization(
                subject,
                ENROLLMENT_POLKIT_ACTION,
                HashMap::new(),
                POLKIT_ALLOW_USER_INTERACTION,
                "",
            )
            .await
            .map_err(|error| {
                BackendError::new(format!("PolicyKit authorization failed: {error}"))
            })?;
        if authorized {
            Ok(())
        } else {
            Err(BackendError::new("PolicyKit denied enrollment authorization"))
        }
    }

    /// Stable `PolicyKit` service, path, and interface used by this client.
    #[must_use]
    pub const fn policykit_endpoint() -> (&'static str, &'static str, &'static str) {
        (POLKIT_SERVICE, POLKIT_PATH, POLKIT_INTERFACE)
    }
}

impl SystemAuthority for SystemBusAuthority {
    fn caller_uid<'a>(&'a self, sender: &'a str) -> BackendFuture<'a, u32> {
        Box::pin(async move { Self::caller_uid(self, sender).await })
    }

    fn authorize_enrollment<'a>(&'a self, sender: &'a str) -> BackendFuture<'a, ()> {
        Box::pin(async move { Self::authorize_enrollment(self, sender).await })
    }
}

/// System-bus composition of identity, `PolicyKit`, template state, and grant issuance.
pub struct SystemBusBackend<A, S, I> {
    authority: A,
    state: S,
    issuer: I,
}

impl<A, S, I> SystemBusBackend<A, S, I>
where
    A: SystemAuthority,
    S: EnrollmentStateSource,
    I: EnrollmentGrantIssuer,
{
    /// Construct a backend without claiming a bus name or installing policy.
    #[must_use]
    pub const fn new(authority: A, state: S, issuer: I) -> Self {
        Self { authority, state, issuer }
    }
}

impl<A, S, I> AuthorizationBackend for SystemBusBackend<A, S, I>
where
    A: SystemAuthority,
    S: EnrollmentStateSource,
    I: EnrollmentGrantIssuer,
{
    fn caller_uid<'a>(&'a self, sender: &'a str) -> BackendFuture<'a, u32> {
        Box::pin(async move { self.authority.caller_uid(sender).await })
    }

    fn enrollment_state<'a>(&'a self, sender: &'a str, target_uid: u32) -> BackendFuture<'a, bool> {
        Box::pin(async move {
            let caller_uid = self.authority.caller_uid(sender).await?;
            if caller_uid != target_uid {
                return Err(BackendError::new("caller is not the target account"));
            }
            self.state.is_enrolled(target_uid).await
        })
    }

    fn authorize_enrollment<'a>(
        &'a self,
        sender: &'a str,
        target_uid: u32,
    ) -> BackendFuture<'a, AuthorizedEnrollment> {
        Box::pin(async move {
            let caller_uid = self.authority.caller_uid(sender).await?;
            if caller_uid != target_uid {
                return Err(BackendError::new("caller is not the target account"));
            }
            self.authority.authorize_enrollment(sender).await?;
            let grant = self.issuer.issue_grant(sender, target_uid).await?;
            AuthorizedEnrollment::from_grant(&grant)
                .map_err(|error| BackendError::new(error.to_string()))
        })
    }
}

/// D-Bus implementation of `org.faceauth.Manager1`.
pub struct Manager1 {
    coordinator: Arc<Mutex<ManagementCoordinator>>,
    backend: Arc<dyn AuthorizationBackend>,
    clock: Arc<dyn MonotonicClock>,
}

/// Daemon-worker view of the coordinator using the adapter's exact monotonic clock.
#[derive(Clone)]
pub struct ManagementWorkerHandle {
    coordinator: Arc<Mutex<ManagementCoordinator>>,
    clock: Arc<dyn MonotonicClock>,
}

impl ManagementWorkerHandle {
    /// Record safe progress for the exact target UID and operation.
    ///
    /// # Errors
    ///
    /// Returns [`WorkerError`] when the binding, deadline, or coordinator lock is invalid.
    pub fn progress(
        &self,
        target_uid: u32,
        operation_id: OperationId,
        progress: ManagementProgress,
    ) -> Result<ManagementUpdate, WorkerError> {
        self.coordinator
            .lock()
            .map_err(|_| WorkerError::CoordinatorPoisoned)?
            .progress(target_uid, operation_id, progress, self.clock.now_micros())
            .map_err(WorkerError::Management)
    }

    /// Complete the exact target UID and operation with a daemon-owned result.
    ///
    /// # Errors
    ///
    /// Returns [`WorkerError`] when the binding, result, deadline, or coordinator lock is invalid.
    pub fn complete(
        &self,
        target_uid: u32,
        operation_id: OperationId,
        result: ManagementResult,
    ) -> Result<ManagementUpdate, WorkerError> {
        self.coordinator
            .lock()
            .map_err(|_| WorkerError::CoordinatorPoisoned)?
            .complete(target_uid, operation_id, result, self.clock.now_micros())
            .map_err(WorkerError::Management)
    }

    /// Reap the active operation if its monotonic deadline elapsed.
    ///
    /// # Errors
    ///
    /// Returns [`WorkerError::CoordinatorPoisoned`] when the coordinator lock is unavailable.
    pub fn expire(&self) -> Result<Option<ManagementUpdate>, WorkerError> {
        Ok(self
            .coordinator
            .lock()
            .map_err(|_| WorkerError::CoordinatorPoisoned)?
            .expire(self.clock.now_micros()))
    }
}

/// Management-worker lifecycle failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkerError {
    /// Core management lifecycle rejected the update.
    Management(ManagementError),
    /// A previous panic poisoned the coordinator lock.
    CoordinatorPoisoned,
}

impl std::fmt::Display for WorkerError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Management(error) => error.fmt(formatter),
            Self::CoordinatorPoisoned => {
                formatter.write_str("management coordinator lock is poisoned")
            }
        }
    }
}

impl std::error::Error for WorkerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Management(error) => Some(error),
            Self::CoordinatorPoisoned => None,
        }
    }
}

impl Manager1 {
    /// Construct a Manager1 object around an existing coordinator and authorization backend.
    #[must_use]
    pub fn new(
        coordinator: ManagementCoordinator,
        backend: impl AuthorizationBackend,
        clock: impl MonotonicClock,
    ) -> Self {
        Self {
            coordinator: Arc::new(Mutex::new(coordinator)),
            backend: Arc::new(backend),
            clock: Arc::new(clock),
        }
    }

    /// Return a worker handle bound to the same coordinator and monotonic clock.
    #[must_use]
    pub fn worker_handle(&self) -> ManagementWorkerHandle {
        ManagementWorkerHandle {
            coordinator: Arc::clone(&self.coordinator),
            clock: Arc::clone(&self.clock),
        }
    }

    /// Emit one safe management update on a caller-provided object-server context.
    ///
    /// The worker may only pass updates produced by [`ManagementCoordinator`]; this method does
    /// not accept raw biometric evidence or arbitrary signal codes.
    ///
    /// # Errors
    ///
    /// Returns [`zbus::Error`] when the connection cannot emit the signal.
    pub async fn emit_update(
        &self,
        ctxt: &SignalContext<'_>,
        update: ManagementUpdate,
    ) -> zbus::Result<()> {
        match update {
            ManagementUpdate::Progress { operation_id, progress } => {
                let operation_id = operation_id.to_string();
                Self::enrollment_progress(ctxt, &operation_id, progress_name(progress)).await
            }
            ManagementUpdate::Completed { operation_id, result } => {
                let operation_id = operation_id.to_string();
                Self::enrollment_completed(ctxt, &operation_id, result_name(result)).await
            }
        }
    }

    fn sender<'a>(header: &'a Header<'a>) -> Result<&'a str, zbus::fdo::Error> {
        header
            .sender()
            .map(zbus::names::UniqueName::as_str)
            .ok_or_else(|| zbus::fdo::Error::AccessDenied("message sender is unavailable".into()))
    }

    async fn require_target(&self, sender: &str, target_uid: u32) -> Result<(), zbus::fdo::Error> {
        let caller_uid =
            self.backend.caller_uid(sender).await.map_err(|error| map_backend_error(&error))?;
        if caller_uid == target_uid {
            Ok(())
        } else {
            Err(zbus::fdo::Error::AccessDenied("caller is not the target account".into()))
        }
    }

    async fn get_enrollment_state_impl(
        &self,
        sender: &str,
        target_uid: u32,
    ) -> Result<bool, zbus::fdo::Error> {
        self.require_target(sender, target_uid).await?;
        self.backend
            .enrollment_state(sender, target_uid)
            .await
            .map_err(|error| map_backend_error(&error))
    }

    async fn begin_enrollment_impl(
        &self,
        sender: &str,
        target_uid: u32,
    ) -> Result<(OperationId, ManagementProgress), zbus::fdo::Error> {
        self.require_target(sender, target_uid).await?;
        let authorization = self
            .backend
            .authorize_enrollment(sender, target_uid)
            .await
            .map_err(|error| map_backend_error(&error))?;
        if authorization.target_uid() != target_uid {
            return Err(zbus::fdo::Error::AccessDenied(
                "authorization proof is bound to another account".into(),
            ));
        }
        let update = {
            let mut coordinator = self.coordinator.lock().map_err(|_| lock_error())?;
            coordinator
                .start(authorization, self.clock.now_micros())
                .map_err(map_management_error)?
        };
        let (operation_id, progress) = match update {
            ManagementUpdate::Progress { operation_id, progress } => (operation_id, progress),
            ManagementUpdate::Completed { .. } => {
                return Err(zbus::fdo::Error::Failed(
                    "unexpected terminal enrollment state".into(),
                ));
            }
        };
        Ok((operation_id, progress))
    }

    async fn cancel_enrollment_impl(
        &self,
        sender: &str,
        target_uid: u32,
        operation_id: &str,
    ) -> Result<(OperationId, ManagementResult), zbus::fdo::Error> {
        self.require_target(sender, target_uid).await?;
        let operation_id = OperationId::parse(operation_id).map_err(map_management_error)?;
        let update = {
            let mut coordinator = self.coordinator.lock().map_err(|_| lock_error())?;
            coordinator
                .cancel(target_uid, operation_id, self.clock.now_micros())
                .map_err(map_management_error)?
        };
        let (operation_id, result) = match update {
            ManagementUpdate::Completed { operation_id, result } => (operation_id, result),
            ManagementUpdate::Progress { .. } => {
                return Err(zbus::fdo::Error::Failed(
                    "unexpected progress cancellation state".into(),
                ));
            }
        };
        Ok((operation_id, result))
    }
}

#[interface(name = "org.faceauth.Manager1")]
impl Manager1 {
    /// Return the management schema version.
    #[zbus(out_args("schema_version"))]
    const fn get_version(&self) -> (u16,) {
        let _ = self;
        (MANAGEMENT_SCHEMA_VERSION,)
    }

    /// Return whether the target account has an enrolled template.
    #[allow(clippy::needless_pass_by_value)]
    #[zbus(out_args("enrolled"))]
    async fn get_enrollment_state(
        &self,
        target_uid: u32,
        #[zbus(header)] header: Header<'_>,
    ) -> Result<(bool,), zbus::fdo::Error> {
        Ok((Self::get_enrollment_state_impl(self, Self::sender(&header)?, target_uid).await?,))
    }

    /// Begin a Polkit-authorized enrollment operation for the target account.
    #[zbus(out_args("operation_id"))]
    async fn begin_enrollment(
        &self,
        target_uid: u32,
        #[zbus(header)] header: Header<'_>,
        #[zbus(signal_context)] ctxt: SignalContext<'_>,
    ) -> Result<(String,), zbus::fdo::Error> {
        let (operation_id, progress) =
            Self::begin_enrollment_impl(self, Self::sender(&header)?, target_uid).await?;
        let operation_id = operation_id.to_string();
        Self::enrollment_progress(&ctxt, &operation_id, progress_name(progress)).await?;
        Ok((operation_id,))
    }

    /// Cancel the exact operation owned by the target account.
    async fn cancel_enrollment(
        &self,
        target_uid: u32,
        operation_id: &str,
        #[zbus(header)] header: Header<'_>,
        #[zbus(signal_context)] ctxt: SignalContext<'_>,
    ) -> Result<(), zbus::fdo::Error> {
        let (operation_id, result) =
            Self::cancel_enrollment_impl(self, Self::sender(&header)?, target_uid, operation_id)
                .await?;
        let operation_id = operation_id.to_string();
        Self::enrollment_completed(&ctxt, &operation_id, result_name(result))
            .await
            .map_err(zbus::fdo::Error::ZBus)
    }

    /// Broadcast safe UI progress for an active operation.
    #[zbus(signal)]
    async fn enrollment_progress(
        ctxt: &SignalContext<'_>,
        operation_id: &str,
        progress: &str,
    ) -> zbus::Result<()>;

    /// Broadcast a safe terminal result for an operation.
    #[zbus(signal)]
    async fn enrollment_completed(
        ctxt: &SignalContext<'_>,
        operation_id: &str,
        result: &str,
    ) -> zbus::Result<()>;
}

const fn progress_name(progress: ManagementProgress) -> &'static str {
    match progress {
        ManagementProgress::Preparing => "preparing",
        ManagementProgress::PositionFace => "position-face",
        ManagementProgress::HoldStill => "hold-still",
        ManagementProgress::ActiveChallenge => "active-challenge",
        ManagementProgress::Processing => "processing",
    }
}

const fn result_name(result: ManagementResult) -> &'static str {
    match result {
        ManagementResult::Completed => "completed",
        ManagementResult::Cancelled => "cancelled",
        ManagementResult::TimedOut => "timed-out",
        ManagementResult::Failed => "failed",
    }
}

fn lock_error() -> zbus::fdo::Error {
    zbus::fdo::Error::Failed("management coordinator lock is poisoned".into())
}

fn map_backend_error(error: &BackendError) -> zbus::fdo::Error {
    zbus::fdo::Error::AccessDenied(error.to_string())
}

fn unique_sender(sender: &str) -> Result<zbus::names::UniqueName<'_>, BackendError> {
    zbus::names::UniqueName::try_from(sender)
        .map_err(|_| BackendError::new("D-Bus sender is not a unique name"))
}

fn map_management_error(error: ManagementError) -> zbus::fdo::Error {
    match error {
        ManagementError::Unauthorized | ManagementError::WrongUid => {
            zbus::fdo::Error::AccessDenied(error.to_string())
        }
        ManagementError::Busy => zbus::fdo::Error::LimitsExceeded(error.to_string()),
        ManagementError::InvalidOperationId
        | ManagementError::NoActiveOperation
        | ManagementError::WrongOperation
        | ManagementError::ReservedResult => zbus::fdo::Error::InvalidArgs(error.to_string()),
        ManagementError::InvalidConfig | ManagementError::DeadlineOverflow => {
            zbus::fdo::Error::Failed(error.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    use faceauth_authz::{AuthorizationGrant, ExecutableFingerprint};
    use faceauth_management::{ENROLLMENT_SERVICE, ManagementConfig};
    use faceauth_protocol::{AuthenticationPurpose, ServiceName, TransactionId};
    use faceauth_transport::PeerIdentity;

    use super::*;

    struct Backend;

    impl AuthorizationBackend for Backend {
        fn caller_uid<'a>(&'a self, sender: &'a str) -> BackendFuture<'a, u32> {
            Box::pin(async move {
                (sender == ":1.7")
                    .then_some(1000)
                    .ok_or_else(|| BackendError::new("unknown sender"))
            })
        }

        fn authorize_enrollment<'a>(
            &'a self,
            sender: &'a str,
            target_uid: u32,
        ) -> BackendFuture<'a, AuthorizedEnrollment> {
            Box::pin(async move {
                let peer = PeerIdentity { pid: 1, uid: 0, gid: 0 };
                let grant = AuthorizationGrant {
                    transaction_id: TransactionId::generate(),
                    peer,
                    target_uid,
                    service: ServiceName::parse(ENROLLMENT_SERVICE)
                        .map_err(|error| BackendError::new(error.to_string()))?,
                    purpose: AuthenticationPurpose::Polkit,
                    executable: ExecutableFingerprint { device: 1, inode: 2 },
                };
                if sender == ":1.7" {
                    AuthorizedEnrollment::from_grant(&grant)
                        .map_err(|error| BackendError::new(error.to_string()))
                } else {
                    Err(BackendError::new("denied"))
                }
            })
        }

        fn enrollment_state<'a>(
            &'a self,
            sender: &'a str,
            target_uid: u32,
        ) -> BackendFuture<'a, bool> {
            Box::pin(async move {
                if sender == ":1.7" && target_uid == 1000 {
                    Ok(true)
                } else {
                    Err(BackendError::new("unknown account"))
                }
            })
        }
    }

    struct TestClock(AtomicU64);

    impl MonotonicClock for TestClock {
        fn now_micros(&self) -> u64 {
            self.0.load(Ordering::Relaxed)
        }
    }

    struct MockAuthority {
        uid: u32,
        authorized: bool,
        authorization_calls: Arc<AtomicUsize>,
    }

    impl SystemAuthority for MockAuthority {
        fn caller_uid<'a>(&'a self, _sender: &'a str) -> BackendFuture<'a, u32> {
            Box::pin(async move { Ok(self.uid) })
        }

        fn authorize_enrollment<'a>(&'a self, _sender: &'a str) -> BackendFuture<'a, ()> {
            Box::pin(async move {
                self.authorization_calls.fetch_add(1, Ordering::Relaxed);
                if self.authorized { Ok(()) } else { Err(BackendError::new("denied")) }
            })
        }
    }

    struct MockState(bool);

    impl EnrollmentStateSource for MockState {
        fn is_enrolled(&self, _target_uid: u32) -> BackendFuture<'_, bool> {
            Box::pin(async move { Ok(self.0) })
        }
    }

    struct MockIssuer {
        peer_uid: u32,
    }

    impl EnrollmentGrantIssuer for MockIssuer {
        fn issue_grant<'a>(
            &'a self,
            _sender: &'a str,
            target_uid: u32,
        ) -> BackendFuture<'a, AuthorizationGrant> {
            Box::pin(async move {
                Ok(AuthorizationGrant {
                    transaction_id: TransactionId::generate(),
                    peer: PeerIdentity { pid: 1, uid: self.peer_uid, gid: 0 },
                    target_uid,
                    service: ServiceName::parse(ENROLLMENT_SERVICE)
                        .map_err(|error| BackendError::new(error.to_string()))?,
                    purpose: AuthenticationPurpose::Polkit,
                    executable: ExecutableFingerprint { device: 1, inode: 2 },
                })
            })
        }
    }

    #[test]
    fn begin_and_cancel_are_sender_and_uid_bound() -> Result<(), Box<dyn std::error::Error>> {
        let coordinator = ManagementCoordinator::new(ManagementConfig::default())?;
        let clock = TestClock(AtomicU64::new(1_000_000));
        let manager = Manager1::new(coordinator, Backend, clock);
        assert_eq!(manager.get_version(), (MANAGEMENT_SCHEMA_VERSION,));
        assert!(futures_lite::future::block_on(manager.get_enrollment_state_impl(":1.7", 1000))?);
        let (operation, progress) =
            futures_lite::future::block_on(manager.begin_enrollment_impl(":1.7", 1000))?;
        assert_eq!(progress, ManagementProgress::Preparing);
        let worker = manager.worker_handle();
        assert_eq!(
            worker.progress(1000, operation, ManagementProgress::HoldStill)?,
            ManagementUpdate::Progress {
                operation_id: operation,
                progress: ManagementProgress::HoldStill,
            }
        );
        assert_eq!(
            worker.progress(1001, operation, ManagementProgress::Processing),
            Err(WorkerError::Management(ManagementError::WrongUid))
        );
        let (_, result) = futures_lite::future::block_on(manager.cancel_enrollment_impl(
            ":1.7",
            1000,
            &operation.to_string(),
        ))?;
        assert_eq!(result, ManagementResult::Cancelled);
        assert!(futures_lite::future::block_on(manager.get_enrollment_state_impl(":1.7", 1000))?);
        assert!(matches!(
            futures_lite::future::block_on(manager.get_enrollment_state_impl(":1.8", 1000)),
            Err(zbus::fdo::Error::AccessDenied(_))
        ));
        assert!(matches!(
            futures_lite::future::block_on(manager.cancel_enrollment_impl(
                ":1.7",
                1000,
                "not-a-uuid",
            )),
            Err(zbus::fdo::Error::InvalidArgs(_))
        ));
        Ok(())
    }

    #[test]
    fn public_signal_codes_are_stable_and_non_biometric() {
        assert_eq!(progress_name(ManagementProgress::Preparing), "preparing");
        assert_eq!(progress_name(ManagementProgress::PositionFace), "position-face");
        assert_eq!(progress_name(ManagementProgress::HoldStill), "hold-still");
        assert_eq!(progress_name(ManagementProgress::ActiveChallenge), "active-challenge");
        assert_eq!(progress_name(ManagementProgress::Processing), "processing");
        assert_eq!(result_name(ManagementResult::Completed), "completed");
        assert_eq!(result_name(ManagementResult::Cancelled), "cancelled");
        assert_eq!(result_name(ManagementResult::TimedOut), "timed-out");
        assert_eq!(result_name(ManagementResult::Failed), "failed");
    }

    #[test]
    fn system_backend_requires_uid_polkit_and_exact_root_grant()
    -> Result<(), Box<dyn std::error::Error>> {
        let calls = Arc::new(AtomicUsize::new(0));
        let valid = SystemBusBackend::new(
            MockAuthority { uid: 1000, authorized: true, authorization_calls: Arc::clone(&calls) },
            MockState(true),
            MockIssuer { peer_uid: 0 },
        );
        assert_eq!(futures_lite::future::block_on(valid.caller_uid(":1.7"))?, 1000);
        assert!(futures_lite::future::block_on(valid.enrollment_state(":1.7", 1000))?);
        let authorization =
            futures_lite::future::block_on(valid.authorize_enrollment(":1.7", 1000))?;
        assert_eq!(authorization.target_uid(), 1000);
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        let mismatch_calls = Arc::new(AtomicUsize::new(0));
        let mismatched_uid = SystemBusBackend::new(
            MockAuthority {
                uid: 1001,
                authorized: true,
                authorization_calls: Arc::clone(&mismatch_calls),
            },
            MockState(true),
            MockIssuer { peer_uid: 0 },
        );
        assert!(
            futures_lite::future::block_on(mismatched_uid.authorize_enrollment(":1.7", 1000))
                .is_err()
        );
        assert_eq!(mismatch_calls.load(Ordering::Relaxed), 0);

        let wrong_grant = SystemBusBackend::new(
            MockAuthority {
                uid: 1000,
                authorized: true,
                authorization_calls: Arc::new(AtomicUsize::new(0)),
            },
            MockState(true),
            MockIssuer { peer_uid: 1000 },
        );
        assert!(
            futures_lite::future::block_on(wrong_grant.authorize_enrollment(":1.7", 1000)).is_err()
        );
        assert_eq!(
            SystemBusAuthority::policykit_endpoint(),
            (POLKIT_SERVICE, POLKIT_PATH, POLKIT_INTERFACE)
        );
        assert!(unique_sender(":1.7").is_ok());
        assert!(unique_sender("org.faceauth.Manager1").is_err());
        Ok(())
    }

    #[test]
    fn generated_introspection_matches_manager1_contract() -> Result<(), Box<dyn std::error::Error>>
    {
        use zbus::object_server::Interface;

        let coordinator = ManagementCoordinator::new(ManagementConfig::default())?;
        let manager = Manager1::new(coordinator, Backend, TestClock(AtomicU64::new(1_000_000)));
        let mut xml = String::new();
        Interface::introspect_to_writer(&manager, &mut xml, 0);

        for expected in [
            "<interface name=\"org.faceauth.Manager1\">",
            "<method name=\"GetVersion\">",
            "<arg name=\"schema_version\" type=\"q\" direction=\"out\"/>",
            "<method name=\"GetEnrollmentState\">",
            "<arg name=\"target_uid\" type=\"u\" direction=\"in\"/>",
            "<arg name=\"enrolled\" type=\"b\" direction=\"out\"/>",
            "<method name=\"BeginEnrollment\">",
            "<arg name=\"operation_id\" type=\"s\" direction=\"out\"/>",
            "<method name=\"CancelEnrollment\">",
            "<signal name=\"EnrollmentProgress\">",
            "<signal name=\"EnrollmentCompleted\">",
        ] {
            assert!(xml.contains(expected), "missing introspection fragment: {expected}\n{xml}");
        }
        Ok(())
    }
}
