//! Thin D-Bus transport adapter for the stable Manager1 management contract.
//!
//! The adapter deliberately does not implement Polkit.  A daemon supplies an authorization
//! backend that resolves the message sender and performs the policy check before returning an
//! [`AuthorizedEnrollment`] proof.  This keeps the security boundary explicit and keeps the core
//! management crate independent of D-Bus.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use faceauth_management::{
    AuthorizedEnrollment, MANAGEMENT_SCHEMA_VERSION, ManagementCoordinator, ManagementError,
    ManagementProgress, ManagementResult, ManagementUpdate, OperationId,
};
use zbus::{interface, message::Header, object_server::SignalContext};

/// Authorization and caller-identity operations required by the D-Bus boundary.
pub trait AuthorizationBackend: Send + Sync + 'static {
    /// Resolve the Unix UID for a unique D-Bus sender name.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when credentials cannot be resolved or are not trusted.
    fn caller_uid(&self, sender: &str) -> Result<u32, BackendError>;

    /// Return whether the exact target account has an enrolled template.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when template state cannot be read safely.
    fn enrollment_state(&self, sender: &str, target_uid: u32) -> Result<bool, BackendError>;

    /// Perform the Polkit check and return the exact authorization proof accepted by the core.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when authorization fails closed.
    fn authorize_enrollment(
        &self,
        sender: &str,
        target_uid: u32,
    ) -> Result<AuthorizedEnrollment, BackendError>;
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

    fn require_target(&self, sender: &str, target_uid: u32) -> Result<(), zbus::fdo::Error> {
        let caller_uid =
            self.backend.caller_uid(sender).map_err(|error| map_backend_error(&error))?;
        if caller_uid == target_uid {
            Ok(())
        } else {
            Err(zbus::fdo::Error::AccessDenied("caller is not the target account".into()))
        }
    }

    fn get_enrollment_state_impl(
        &self,
        sender: &str,
        target_uid: u32,
    ) -> Result<bool, zbus::fdo::Error> {
        self.require_target(sender, target_uid)?;
        self.backend.enrollment_state(sender, target_uid).map_err(|error| map_backend_error(&error))
    }

    fn begin_enrollment_impl(
        &self,
        sender: &str,
        target_uid: u32,
    ) -> Result<(OperationId, ManagementProgress), zbus::fdo::Error> {
        self.require_target(sender, target_uid)?;
        let authorization = self
            .backend
            .authorize_enrollment(sender, target_uid)
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

    fn cancel_enrollment_impl(
        &self,
        sender: &str,
        target_uid: u32,
        operation_id: &str,
    ) -> Result<(OperationId, ManagementResult), zbus::fdo::Error> {
        self.require_target(sender, target_uid)?;
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
    fn get_enrollment_state(
        &self,
        target_uid: u32,
        #[zbus(header)] header: Header<'_>,
    ) -> Result<(bool,), zbus::fdo::Error> {
        Ok((Self::get_enrollment_state_impl(self, Self::sender(&header)?, target_uid)?,))
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
            Self::begin_enrollment_impl(self, Self::sender(&header)?, target_uid)?;
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
            Self::cancel_enrollment_impl(self, Self::sender(&header)?, target_uid, operation_id)?;
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
    use std::sync::atomic::{AtomicU64, Ordering};

    use faceauth_authz::{AuthorizationGrant, ExecutableFingerprint};
    use faceauth_management::{ENROLLMENT_SERVICE, ManagementConfig};
    use faceauth_protocol::{AuthenticationPurpose, ServiceName, TransactionId};
    use faceauth_transport::PeerIdentity;

    use super::*;

    struct Backend;

    impl AuthorizationBackend for Backend {
        fn caller_uid(&self, sender: &str) -> Result<u32, BackendError> {
            (sender == ":1.7").then_some(1000).ok_or_else(|| BackendError::new("unknown sender"))
        }

        fn authorize_enrollment(
            &self,
            sender: &str,
            target_uid: u32,
        ) -> Result<AuthorizedEnrollment, BackendError> {
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
        }

        fn enrollment_state(&self, sender: &str, target_uid: u32) -> Result<bool, BackendError> {
            if sender == ":1.7" && target_uid == 1000 {
                Ok(true)
            } else {
                Err(BackendError::new("unknown account"))
            }
        }
    }

    struct TestClock(AtomicU64);

    impl MonotonicClock for TestClock {
        fn now_micros(&self) -> u64 {
            self.0.load(Ordering::Relaxed)
        }
    }

    #[test]
    fn begin_and_cancel_are_sender_and_uid_bound() -> Result<(), Box<dyn std::error::Error>> {
        let coordinator = ManagementCoordinator::new(ManagementConfig::default())?;
        let clock = TestClock(AtomicU64::new(1_000_000));
        let manager = Manager1::new(coordinator, Backend, clock);
        assert_eq!(manager.get_version(), (MANAGEMENT_SCHEMA_VERSION,));
        assert!(manager.get_enrollment_state_impl(":1.7", 1000)?);
        let (operation, progress) = manager.begin_enrollment_impl(":1.7", 1000)?;
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
        let (_, result) = manager.cancel_enrollment_impl(":1.7", 1000, &operation.to_string())?;
        assert_eq!(result, ManagementResult::Cancelled);
        assert!(manager.get_enrollment_state_impl(":1.7", 1000)?);
        assert!(matches!(
            manager.get_enrollment_state_impl(":1.8", 1000),
            Err(zbus::fdo::Error::AccessDenied(_))
        ));
        assert!(matches!(
            manager.cancel_enrollment_impl(":1.7", 1000, "not-a-uuid"),
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
