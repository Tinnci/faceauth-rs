//! Thin D-Bus transport adapter for the stable Manager1 management contract.
//!
//! The adapter provides asynchronous system-bus sender credential lookup and the narrow `PolicyKit`
//! enrollment check. A daemon still supplies authenticated-template state and a dedicated root
//! grant issuer before the adapter can return an [`AuthorizedEnrollment`] proof. This keeps the
//! security boundary explicit and keeps the core management crate independent of D-Bus.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::Instant;

use faceauth_authz::{AuthorizationGrant, BoundAuthorizationIssuer};
use faceauth_management::{
    AuthorizedEnrollment, ENROLLMENT_POLKIT_ACTION, MANAGEMENT_SCHEMA_VERSION, MANAGER_BUS_NAME,
    MANAGER_OBJECT_PATH, ManagementCoordinator, ManagementError, ManagementProgress,
    ManagementResult, ManagementUpdate, OperationId,
};
use futures_util::{
    StreamExt,
    future::{Either, select},
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

/// Maximum queued encrypted-template state lookups.
pub const MAX_TEMPLATE_STATE_QUEUE_CAPACITY: usize = 64;

/// Synchronous authenticated-template reader executed outside the D-Bus executor.
pub trait TemplateStateReader: Send + 'static {
    /// Load and authenticate enough template state to answer whether a UID is enrolled.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] for corrupt, untrusted, or unavailable storage.
    fn is_enrolled(&self, target_uid: u32) -> Result<bool, BackendError>;
}

enum TemplateStateRequest {
    Query {
        target_uid: u32,
        response: futures_channel::oneshot::Sender<Result<bool, BackendError>>,
    },
    Shutdown,
}

/// Single-worker, bounded bridge from asynchronous Manager1 calls to blocking template storage.
pub struct BoundedEnrollmentStateSource {
    requests: mpsc::SyncSender<TemplateStateRequest>,
    worker: Option<thread::JoinHandle<()>>,
}

impl BoundedEnrollmentStateSource {
    /// Start one named storage worker with a bounded request queue.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] for zero/excessive capacity or thread creation failure.
    pub fn new(
        reader: impl TemplateStateReader,
        queue_capacity: usize,
    ) -> Result<Self, BackendError> {
        if queue_capacity == 0 || queue_capacity > MAX_TEMPLATE_STATE_QUEUE_CAPACITY {
            return Err(BackendError::new("template state queue capacity is invalid"));
        }
        let (requests, receiver) = mpsc::sync_channel(queue_capacity);
        let worker = thread::Builder::new()
            .name("faceauth-template-state".into())
            .spawn(move || template_state_worker(&reader, &receiver))
            .map_err(|error| {
                BackendError::new(format!("unable to start template state worker: {error}"))
            })?;
        Ok(Self { requests, worker: Some(worker) })
    }
}

impl EnrollmentStateSource for BoundedEnrollmentStateSource {
    fn is_enrolled(&self, target_uid: u32) -> BackendFuture<'_, bool> {
        let (response, receiver) = futures_channel::oneshot::channel();
        if self.requests.try_send(TemplateStateRequest::Query { target_uid, response }).is_err() {
            return Box::pin(async {
                Err(BackendError::new("template state worker queue is unavailable or full"))
            });
        }
        Box::pin(async move {
            receiver.await.map_err(|_| BackendError::new("template state worker stopped"))?
        })
    }
}

impl Drop for BoundedEnrollmentStateSource {
    fn drop(&mut self) {
        let _ = self.requests.send(TemplateStateRequest::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn template_state_worker(
    reader: &impl TemplateStateReader,
    receiver: &mpsc::Receiver<TemplateStateRequest>,
) {
    while let Ok(request) = receiver.recv() {
        match request {
            TemplateStateRequest::Query { target_uid, response } => {
                let _ = response.send(reader.is_enrolled(target_uid));
            }
            TemplateStateRequest::Shutdown => break,
        }
    }
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

impl EnrollmentGrantIssuer for BoundAuthorizationIssuer {
    fn issue_grant<'a>(
        &'a self,
        sender: &'a str,
        target_uid: u32,
    ) -> BackendFuture<'a, AuthorizationGrant> {
        Box::pin(async move {
            unique_sender(sender)?;
            let grant =
                self.issue(target_uid).map_err(|error| BackendError::new(error.to_string()))?;
            AuthorizedEnrollment::from_grant(&grant)
                .map_err(|error| BackendError::new(error.to_string()))?;
            Ok(grant)
        })
    }
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

#[derive(Debug, serde::Deserialize, serde::Serialize, zbus::zvariant::Type)]
struct PolkitAuthorizationResult(bool, bool, HashMap<String, String>);

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
    ) -> zbus::Result<PolkitAuthorizationResult>;
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
        let PolkitAuthorizationResult(authorized, _, _) = proxy
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
    state: Arc<Mutex<ManagementState>>,
    backend: Arc<dyn AuthorizationBackend>,
    clock: Arc<dyn MonotonicClock>,
}

struct OperationOwner {
    sender: String,
    target_uid: u32,
    operation_id: OperationId,
}

struct ManagementState {
    coordinator: ManagementCoordinator,
    owner: Option<OperationOwner>,
}

impl ManagementState {
    fn clear_owner_if_terminal(&mut self, update: &ManagementUpdate) {
        if matches!(update, ManagementUpdate::Completed { .. }) {
            self.owner = None;
        }
    }
}

/// Daemon-worker view of the coordinator using the adapter's exact monotonic clock.
#[derive(Clone)]
pub struct ManagementWorkerHandle {
    state: Arc<Mutex<ManagementState>>,
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
        let mut state = self.state.lock().map_err(|_| WorkerError::CoordinatorPoisoned)?;
        let update = state
            .coordinator
            .progress(target_uid, operation_id, progress, self.clock.now_micros())
            .map_err(WorkerError::Management)?;
        state.clear_owner_if_terminal(&update);
        drop(state);
        Ok(update)
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
        let mut state = self.state.lock().map_err(|_| WorkerError::CoordinatorPoisoned)?;
        let update = state
            .coordinator
            .complete(target_uid, operation_id, result, self.clock.now_micros())
            .map_err(WorkerError::Management)?;
        state.clear_owner_if_terminal(&update);
        drop(state);
        Ok(update)
    }

    /// Reap the active operation if its monotonic deadline elapsed.
    ///
    /// # Errors
    ///
    /// Returns [`WorkerError::CoordinatorPoisoned`] when the coordinator lock is unavailable.
    pub fn expire(&self) -> Result<Option<ManagementUpdate>, WorkerError> {
        let mut state = self.state.lock().map_err(|_| WorkerError::CoordinatorPoisoned)?;
        let update = state.coordinator.expire(self.clock.now_micros());
        if let Some(update) = update.as_ref() {
            state.clear_owner_if_terminal(update);
        }
        drop(state);
        Ok(update)
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

/// Cloneable system-bus disconnect monitor for a registered Manager1 object.
#[derive(Clone)]
pub struct ManagementDisconnectHandle {
    state: Arc<Mutex<ManagementState>>,
    clock: Arc<dyn MonotonicClock>,
}

impl ManagementDisconnectHandle {
    /// Cancel and consume the operation owned by a disconnected unique D-Bus sender.
    ///
    /// Other senders and an idle coordinator return `Ok(None)` without changing state.
    ///
    /// # Errors
    ///
    /// Returns [`WorkerError`] if the coordinator lock or lifecycle invariant fails.
    pub fn cancel_sender(&self, sender: &str) -> Result<Option<ManagementUpdate>, WorkerError> {
        let mut state = self.state.lock().map_err(|_| WorkerError::CoordinatorPoisoned)?;
        let Some(owner) = state.owner.as_ref() else {
            return Ok(None);
        };
        if owner.sender != sender {
            return Ok(None);
        }
        let target_uid = owner.target_uid;
        let operation_id = owner.operation_id;
        let update = state
            .coordinator
            .cancel(target_uid, operation_id, self.clock.now_micros())
            .map_err(WorkerError::Management)?;
        state.clear_owner_if_terminal(&update);
        drop(state);
        Ok(Some(update))
    }

    /// Watch system-bus ownership changes and cancel operations whose unique sender disappears.
    ///
    /// The caller should spawn this future alongside the object server and stop the service if it
    /// returns an error, because continuing without disconnect cleanup can retain camera capacity.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] if the bus stream, cancellation, or terminal signal fails.
    pub async fn watch(
        &self,
        connection: &Connection,
        signal_context: &SignalContext<'_>,
    ) -> Result<(), BackendError> {
        let proxy = zbus::fdo::DBusProxy::new(connection).await.map_err(|error| {
            BackendError::new(format!("unable to create D-Bus disconnect proxy: {error}"))
        })?;
        let mut changes = proxy.receive_name_owner_changed().await.map_err(|error| {
            BackendError::new(format!("unable to subscribe to D-Bus owner changes: {error}"))
        })?;
        while let Some(change) = changes.next().await {
            let args = change.args().map_err(|error| {
                BackendError::new(format!("invalid D-Bus owner-change signal: {error}"))
            })?;
            let old_owner_present = args.old_owner().as_ref().is_some();
            let new_owner_present = args.new_owner().as_ref().is_some();
            if let Some(update) = self
                .handle_owner_change(args.name().as_str(), old_owner_present, new_owner_present)
                .map_err(|error| {
                    BackendError::new(format!("unable to cancel disconnected sender: {error}"))
                })?
            {
                emit_update_signal(signal_context, update).await.map_err(|error| {
                    BackendError::new(format!("unable to emit disconnect cancellation: {error}"))
                })?;
            }
        }
        Err(BackendError::new("D-Bus owner-change stream ended"))
    }

    fn handle_owner_change(
        &self,
        name: &str,
        old_owner_present: bool,
        new_owner_present: bool,
    ) -> Result<Option<ManagementUpdate>, WorkerError> {
        if !old_owner_present || new_owner_present || unique_sender(name).is_err() {
            return Ok(None);
        }
        self.cancel_sender(name)
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
            state: Arc::new(Mutex::new(ManagementState { coordinator, owner: None })),
            backend: Arc::new(backend),
            clock: Arc::new(clock),
        }
    }

    /// Return a worker handle bound to the same coordinator and monotonic clock.
    #[must_use]
    pub fn worker_handle(&self) -> ManagementWorkerHandle {
        ManagementWorkerHandle { state: Arc::clone(&self.state), clock: Arc::clone(&self.clock) }
    }

    /// Return a handle that watches unique sender ownership and releases abandoned operations.
    #[must_use]
    pub fn disconnect_handle(&self) -> ManagementDisconnectHandle {
        ManagementDisconnectHandle {
            state: Arc::clone(&self.state),
            clock: Arc::clone(&self.clock),
        }
    }

    /// Cancel and consume the operation owned by a disconnected unique D-Bus sender.
    ///
    /// A `NameOwnerChanged` watcher should call this only when the unique name loses its owner.
    /// Other senders and an idle coordinator return `Ok(None)` without changing state.
    ///
    /// # Errors
    ///
    /// Returns [`WorkerError`] if the coordinator lock or lifecycle invariant fails.
    pub fn cancel_disconnected_sender(
        &self,
        sender: &str,
    ) -> Result<Option<ManagementUpdate>, WorkerError> {
        self.disconnect_handle().cancel_sender(sender)
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
        emit_update_signal(ctxt, update).await
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
        unique_sender(sender).map_err(|error| map_backend_error(&error))?;
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
            let mut state = self.state.lock().map_err(|_| lock_error())?;
            let update = state
                .coordinator
                .start(authorization, self.clock.now_micros())
                .map_err(map_management_error)?;
            if let ManagementUpdate::Progress { operation_id, .. } = update {
                state.owner =
                    Some(OperationOwner { sender: sender.to_owned(), target_uid, operation_id });
            }
            update
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
            let mut state = self.state.lock().map_err(|_| lock_error())?;
            let owner = state
                .owner
                .as_ref()
                .ok_or_else(|| map_management_error(ManagementError::NoActiveOperation))?;
            if owner.sender != sender {
                return Err(zbus::fdo::Error::AccessDenied(
                    "management operation belongs to another D-Bus sender".into(),
                ));
            }
            let update = state
                .coordinator
                .cancel(target_uid, operation_id, self.clock.now_micros())
                .map_err(map_management_error)?;
            state.clear_owner_if_terminal(&update);
            update
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
        let sender = Self::sender(&header)?.to_owned();
        let (operation_id, progress) =
            Self::begin_enrollment_impl(self, &sender, target_uid).await?;
        let operation_id = operation_id.to_string();
        if let Err(error) =
            Self::enrollment_progress(&ctxt, &operation_id, progress_name(progress)).await
        {
            let _ = self.cancel_disconnected_sender(&sender);
            return Err(zbus::fdo::Error::ZBus(error));
        }
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

/// Run one Manager1 object on an already-connected message bus.
///
/// Startup subscribes to sender ownership changes before registering the object and requesting the
/// well-known name. Name acquisition uses `DoNotQueue` without replacement. The future runs until
/// the bus stream or cancellation signaling fails, then releases the name and removes the object.
/// The caller should treat every return as service-fatal.
///
/// # Errors
///
/// Returns [`BackendError`] for subscription, object registration, exclusive name acquisition,
/// disconnect cleanup, signaling, or bus-stream failure.
pub async fn run_manager1_service(
    connection: Connection,
    manager: Manager1,
) -> Result<(), BackendError> {
    run_manager1_service_until_shutdown(connection, manager, std::future::pending()).await
}

/// Run one Manager1 object until an explicit graceful shutdown future resolves.
///
/// Activation ordering and failure behavior match [`run_manager1_service`]. A completed shutdown
/// future releases the well-known name, removes the object, and returns `Ok(())`; bus, ownership,
/// disconnect-cleanup, and signal failures remain fatal.
///
/// # Errors
///
/// Returns [`BackendError`] for subscription, object registration, exclusive name acquisition,
/// disconnect cleanup, signaling, or bus-stream failure.
pub async fn run_manager1_service_until_shutdown<F>(
    connection: Connection,
    manager: Manager1,
    shutdown: F,
) -> Result<(), BackendError>
where
    F: Future<Output = ()>,
{
    let proxy = zbus::fdo::DBusProxy::new(&connection).await.map_err(|error| {
        BackendError::new(format!("unable to create Manager1 lifecycle proxy: {error}"))
    })?;
    let mut changes = proxy.receive_name_owner_changed().await.map_err(|error| {
        BackendError::new(format!("unable to subscribe before Manager1 activation: {error}"))
    })?;
    let disconnect = manager.disconnect_handle();
    let signal_context = SignalContext::new(&connection, MANAGER_OBJECT_PATH)
        .map_err(|error| BackendError::new(format!("invalid Manager1 object path: {error}")))?;
    let added =
        connection.object_server().at(MANAGER_OBJECT_PATH, manager).await.map_err(|error| {
            BackendError::new(format!("unable to register Manager1 object: {error}"))
        })?;
    if !added {
        return Err(BackendError::new("Manager1 object was already registered"));
    }
    let reply = connection
        .request_name_with_flags(MANAGER_BUS_NAME, zbus::fdo::RequestNameFlags::DoNotQueue.into())
        .await
        .map_err(|error| BackendError::new(format!("unable to request Manager1 name: {error}")))?;
    if !matches!(
        reply,
        zbus::fdo::RequestNameReply::PrimaryOwner | zbus::fdo::RequestNameReply::AlreadyOwner
    ) {
        let _ = connection.object_server().remove::<Manager1, _>(MANAGER_OBJECT_PATH).await;
        return Err(BackendError::new("Manager1 bus name is already owned"));
    }

    let mut shutdown = Box::pin(shutdown);
    let result = loop {
        let change = Box::pin(changes.next());
        let (change, pending_shutdown) = match select(change, shutdown).await {
            Either::Left((change, pending_shutdown)) => (change, pending_shutdown),
            Either::Right(((), pending_change)) => {
                drop(pending_change);
                break Ok(());
            }
        };
        shutdown = pending_shutdown;
        let Some(change) = change else {
            break Err(BackendError::new("Manager1 owner-change stream ended"));
        };
        let args = match change.args() {
            Ok(args) => args,
            Err(error) => {
                break Err(BackendError::new(format!(
                    "invalid Manager1 owner-change signal: {error}"
                )));
            }
        };
        let update = disconnect
            .handle_owner_change(
                args.name().as_str(),
                args.old_owner().as_ref().is_some(),
                args.new_owner().as_ref().is_some(),
            )
            .map_err(|error| {
                BackendError::new(format!("unable to cancel disconnected Manager1 sender: {error}"))
            });
        let update = match update {
            Ok(update) => update,
            Err(error) => break Err(error),
        };
        if let Some(update) = update
            && let Err(error) = emit_update_signal(&signal_context, update).await
        {
            break Err(BackendError::new(format!(
                "unable to emit Manager1 disconnect result: {error}"
            )));
        }
    };

    let _ = connection.release_name(MANAGER_BUS_NAME).await;
    let _ = connection.object_server().remove::<Manager1, _>(MANAGER_OBJECT_PATH).await;
    result
}

async fn emit_update_signal(
    ctxt: &SignalContext<'_>,
    update: ManagementUpdate,
) -> zbus::Result<()> {
    match update {
        ManagementUpdate::Progress { operation_id, progress } => {
            let operation_id = operation_id.to_string();
            Manager1::enrollment_progress(ctxt, &operation_id, progress_name(progress)).await
        }
        ManagementUpdate::Completed { operation_id, result } => {
            let operation_id = operation_id.to_string();
            Manager1::enrollment_completed(ctxt, &operation_id, result_name(result)).await
        }
    }
}

const fn progress_name(progress: ManagementProgress) -> &'static str {
    match progress {
        ManagementProgress::Preparing => "preparing",
        ManagementProgress::PositionFace => "position-face",
        ManagementProgress::HoldStill => "hold-still",
        ManagementProgress::ActiveChallenge => "active-challenge",
        ManagementProgress::Blink => "blink",
        ManagementProgress::TurnLeft => "turn-left",
        ManagementProgress::TurnRight => "turn-right",
        ManagementProgress::ReturnToCenter => "return-to-center",
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
    use std::io::{BufRead, BufReader};
    use std::os::unix::fs::MetadataExt;
    use std::process::{Child, ChildStdout, Command, Stdio};
    use std::sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        mpsc as test_mpsc,
    };
    use std::time::{Duration, Instant as TestInstant};

    use faceauth_authz::{AuthorizationGrant, ExecutableFingerprint};
    use faceauth_management::{ENROLLMENT_SERVICE, ManagementConfig};
    use faceauth_protocol::{AuthenticationPurpose, ServiceName, TransactionId};
    use faceauth_transport::PeerIdentity;

    use super::*;

    struct PrivateBus {
        child: Child,
        _stdout: BufReader<ChildStdout>,
        address: String,
    }

    impl PrivateBus {
        fn start() -> Result<Self, Box<dyn std::error::Error>> {
            let mut child = Command::new("dbus-daemon")
                .args(["--session", "--nofork", "--nopidfile", "--print-address=1"])
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()?;
            let stdout = child.stdout.take().ok_or("dbus-daemon stdout is unavailable")?;
            let mut stdout = BufReader::new(stdout);
            let mut address = String::new();
            stdout.read_line(&mut address)?;
            let address = address.trim().to_owned();
            if address.is_empty() {
                return Err("dbus-daemon did not print an address".into());
            }
            Ok(Self { child, _stdout: stdout, address })
        }

        fn stop(&mut self) -> Result<(), Box<dyn std::error::Error>> {
            if self.child.try_wait()?.is_none() {
                self.child.kill()?;
            }
            let _ = self.child.wait()?;
            Ok(())
        }
    }

    impl Drop for PrivateBus {
        fn drop(&mut self) {
            let _ = self.stop();
        }
    }

    #[derive(Debug)]
    struct ObservedPolkitCall {
        subject_kind: String,
        subject_name: String,
        action_id: String,
        flags: u32,
        cancellation_id: String,
    }

    struct FakePolicyKit {
        authorized: Arc<std::sync::atomic::AtomicBool>,
        observations: test_mpsc::Sender<ObservedPolkitCall>,
    }

    #[interface(name = "org.freedesktop.PolicyKit1.Authority")]
    impl FakePolicyKit {
        #[allow(clippy::needless_pass_by_value)]
        fn check_authorization(
            &self,
            subject: (&str, HashMap<&str, Value<'_>>),
            action_id: &str,
            details: HashMap<&str, &str>,
            flags: u32,
            cancellation_id: &str,
        ) -> PolkitAuthorizationResult {
            let _ = details;
            let subject_name = subject
                .1
                .get("name")
                .and_then(|value| value.downcast_ref::<&str>().ok())
                .unwrap_or_default()
                .to_owned();
            let _ = self.observations.send(ObservedPolkitCall {
                subject_kind: subject.0.to_owned(),
                subject_name,
                action_id: action_id.to_owned(),
                flags,
                cancellation_id: cancellation_id.to_owned(),
            });
            PolkitAuthorizationResult(
                self.authorized.load(Ordering::Relaxed),
                false,
                HashMap::new(),
            )
        }
    }

    struct Backend;

    impl AuthorizationBackend for Backend {
        fn caller_uid<'a>(&'a self, sender: &'a str) -> BackendFuture<'a, u32> {
            Box::pin(async move {
                matches!(sender, ":1.7" | ":1.8")
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

    struct AnyCallerBackend;

    impl AuthorizationBackend for AnyCallerBackend {
        fn caller_uid<'a>(&'a self, sender: &'a str) -> BackendFuture<'a, u32> {
            Box::pin(async move {
                unique_sender(sender)?;
                Ok(1000)
            })
        }

        fn enrollment_state<'a>(
            &'a self,
            sender: &'a str,
            target_uid: u32,
        ) -> BackendFuture<'a, bool> {
            Box::pin(async move {
                unique_sender(sender)?;
                Ok(target_uid == 1000)
            })
        }

        fn authorize_enrollment<'a>(
            &'a self,
            sender: &'a str,
            target_uid: u32,
        ) -> BackendFuture<'a, AuthorizedEnrollment> {
            Box::pin(async move {
                unique_sender(sender)?;
                let grant = AuthorizationGrant {
                    transaction_id: TransactionId::generate(),
                    peer: PeerIdentity { pid: 1, uid: 0, gid: 0 },
                    target_uid,
                    service: ServiceName::parse(ENROLLMENT_SERVICE)
                        .map_err(|error| BackendError::new(error.to_string()))?,
                    purpose: AuthenticationPurpose::Polkit,
                    executable: ExecutableFingerprint { device: 1, inode: 2 },
                };
                AuthorizedEnrollment::from_grant(&grant)
                    .map_err(|error| BackendError::new(error.to_string()))
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

    struct BlockingStateReader;

    impl TemplateStateReader for BlockingStateReader {
        fn is_enrolled(&self, target_uid: u32) -> Result<bool, BackendError> {
            match target_uid {
                1000 => Ok(true),
                1001 => Ok(false),
                _ => Err(BackendError::new("untrusted template state")),
            }
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
        assert!(matches!(
            futures_lite::future::block_on(manager.cancel_enrollment_impl(
                ":1.8",
                1000,
                &operation.to_string(),
            )),
            Err(zbus::fdo::Error::AccessDenied(_))
        ));
        assert_eq!(
            worker.progress(1000, operation, ManagementProgress::Processing)?,
            ManagementUpdate::Progress {
                operation_id: operation,
                progress: ManagementProgress::Processing,
            }
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
    fn disconnected_owner_is_cancelled_and_capacity_is_released()
    -> Result<(), Box<dyn std::error::Error>> {
        let coordinator = ManagementCoordinator::new(ManagementConfig::default())?;
        let manager = Manager1::new(coordinator, Backend, TestClock(AtomicU64::new(1_000_000)));
        let worker = manager.worker_handle();
        let disconnect = manager.disconnect_handle();
        let (operation, _) =
            futures_lite::future::block_on(manager.begin_enrollment_impl(":1.7", 1000))?;

        assert_eq!(disconnect.handle_owner_change(":1.7", false, false)?, None);
        assert_eq!(disconnect.handle_owner_change(":1.7", true, true)?, None);
        assert_eq!(disconnect.handle_owner_change("org.faceauth.Client", true, false)?, None);
        assert_eq!(disconnect.handle_owner_change(":1.8", true, false)?, None);
        assert_eq!(
            disconnect.handle_owner_change(":1.7", true, false)?,
            Some(ManagementUpdate::Completed {
                operation_id: operation,
                result: ManagementResult::Cancelled,
            })
        );
        assert_eq!(disconnect.cancel_sender(":1.7")?, None);
        assert_eq!(
            worker.progress(1000, operation, ManagementProgress::Processing),
            Err(WorkerError::Management(ManagementError::NoActiveOperation))
        );

        let (second, _) =
            futures_lite::future::block_on(manager.begin_enrollment_impl(":1.7", 1000))?;
        assert!(matches!(
            worker.complete(1000, second, ManagementResult::Completed)?,
            ManagementUpdate::Completed {
                operation_id,
                result: ManagementResult::Completed,
            } if operation_id == second
        ));
        assert_eq!(disconnect.cancel_sender(":1.7")?, None);
        Ok(())
    }

    #[test]
    fn public_signal_codes_are_stable_and_non_biometric() {
        assert_eq!(progress_name(ManagementProgress::Preparing), "preparing");
        assert_eq!(progress_name(ManagementProgress::PositionFace), "position-face");
        assert_eq!(progress_name(ManagementProgress::HoldStill), "hold-still");
        assert_eq!(progress_name(ManagementProgress::ActiveChallenge), "active-challenge");
        assert_eq!(progress_name(ManagementProgress::Blink), "blink");
        assert_eq!(progress_name(ManagementProgress::TurnLeft), "turn-left");
        assert_eq!(progress_name(ManagementProgress::TurnRight), "turn-right");
        assert_eq!(progress_name(ManagementProgress::ReturnToCenter), "return-to-center");
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
    fn template_state_queries_use_one_bounded_worker() -> Result<(), Box<dyn std::error::Error>> {
        assert!(BoundedEnrollmentStateSource::new(BlockingStateReader, 0).is_err());
        assert!(
            BoundedEnrollmentStateSource::new(
                BlockingStateReader,
                MAX_TEMPLATE_STATE_QUEUE_CAPACITY + 1,
            )
            .is_err()
        );

        let source = BoundedEnrollmentStateSource::new(BlockingStateReader, 2)?;
        assert!(futures_lite::future::block_on(source.is_enrolled(1000))?);
        assert!(!futures_lite::future::block_on(source.is_enrolled(1001))?);
        assert!(futures_lite::future::block_on(source.is_enrolled(2000)).is_err());
        Ok(())
    }

    #[test]
    fn isolated_policykit_proxy_binds_exact_sender_action_and_flags()
    -> Result<(), Box<dyn std::error::Error>> {
        use zbus::zvariant::Type;

        assert_eq!(PolkitAuthorizationResult::signature().as_str(), "(bba{ss})");
        assert_eq!(<(&str, HashMap<&str, Value<'_>>)>::signature().as_str(), "(sa{sv})");
        let mut bus = PrivateBus::start()?;
        let address = bus.address.clone();
        futures_lite::future::block_on(async move {
            let policy_connection =
                zbus::ConnectionBuilder::address(address.as_str())?.build().await?;
            let authority_connection =
                zbus::ConnectionBuilder::address(address.as_str())?.build().await?;
            let subject_connection =
                zbus::ConnectionBuilder::address(address.as_str())?.build().await?;
            let authorized = Arc::new(std::sync::atomic::AtomicBool::new(true));
            let (observations, observed) = test_mpsc::channel();
            policy_connection
                .object_server()
                .at(
                    POLKIT_PATH,
                    FakePolicyKit { authorized: Arc::clone(&authorized), observations },
                )
                .await?;
            let reply = policy_connection
                .request_name_with_flags(
                    POLKIT_SERVICE,
                    zbus::fdo::RequestNameFlags::DoNotQueue.into(),
                )
                .await?;
            assert_eq!(reply, zbus::fdo::RequestNameReply::PrimaryOwner);

            let sender = subject_connection
                .unique_name()
                .ok_or("private D-Bus subject has no unique name")?
                .as_str()
                .to_owned();
            let authority = SystemBusAuthority::from_connection(authority_connection);
            let resolved_uid = authority.caller_uid(&sender).await?;
            let expected_uid = std::fs::metadata("/proc/self")?.uid();
            assert_eq!(resolved_uid, expected_uid);
            authority.authorize_enrollment(&sender).await?;
            let call = observed.recv_timeout(Duration::from_secs(2))?;
            assert_eq!(call.subject_kind, POLKIT_SUBJECT_SYSTEM_BUS_NAME);
            assert_eq!(call.subject_name, sender);
            assert_eq!(call.action_id, ENROLLMENT_POLKIT_ACTION);
            assert_eq!(call.flags, POLKIT_ALLOW_USER_INTERACTION);
            assert!(call.cancellation_id.is_empty());

            authorized.store(false, Ordering::Relaxed);
            assert!(authority.authorize_enrollment(&sender).await.is_err());
            Ok::<(), Box<dyn std::error::Error>>(())
        })?;
        bus.stop()?;
        Ok(())
    }

    #[test]
    fn isolated_bus_runner_serves_exclusively_and_cancels_disconnected_client()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut bus = PrivateBus::start()?;
        let coordinator = ManagementCoordinator::new(ManagementConfig::default())?;
        let manager = Manager1::new(coordinator, AnyCallerBackend, SystemMonotonicClock);
        let worker = manager.worker_handle();
        let service_address = bus.address.clone();
        let (service_result_tx, service_result_rx) = test_mpsc::channel();
        let service_thread = thread::spawn(move || {
            let result = futures_lite::future::block_on(async move {
                let connection = zbus::ConnectionBuilder::address(service_address.as_str())
                    .map_err(|error| error.to_string())?
                    .build()
                    .await
                    .map_err(|error| error.to_string())?;
                run_manager1_service(connection, manager).await.map_err(|error| error.to_string())
            });
            let _ = service_result_tx.send(result);
        });

        let client = zbus::blocking::connection::Builder::address(bus.address.as_str())?.build()?;
        let proxy = zbus::blocking::Proxy::new(
            &client,
            MANAGER_BUS_NAME,
            MANAGER_OBJECT_PATH,
            faceauth_management::MANAGER_INTERFACE,
        )?;
        let ready_deadline = TestInstant::now() + Duration::from_secs(5);
        loop {
            let version: zbus::Result<u16> = proxy.call("GetVersion", &());
            if version == Ok(MANAGEMENT_SCHEMA_VERSION) {
                break;
            }
            if TestInstant::now() >= ready_deadline {
                return Err("Manager1 service did not become ready".into());
            }
            thread::sleep(Duration::from_millis(20));
        }

        let second_address = bus.address.clone();
        let second_result = futures_lite::future::block_on(async move {
            let connection = zbus::ConnectionBuilder::address(second_address.as_str())
                .map_err(|error| BackendError::new(error.to_string()))?
                .build()
                .await
                .map_err(|error| BackendError::new(error.to_string()))?;
            let manager = Manager1::new(
                ManagementCoordinator::new(ManagementConfig::default())
                    .map_err(|error| BackendError::new(error.to_string()))?,
                AnyCallerBackend,
                SystemMonotonicClock,
            );
            run_manager1_service(connection, manager).await
        });
        assert!(second_result.is_err());
        let version: u16 = proxy.call("GetVersion", &())?;
        assert_eq!(version, MANAGEMENT_SCHEMA_VERSION);

        let enrolled: bool = proxy.call("GetEnrollmentState", &(1000_u32,))?;
        assert!(enrolled);
        let operation: String = proxy.call("BeginEnrollment", &(1000_u32,))?;
        let operation = OperationId::parse(&operation)?;
        drop(proxy);
        drop(client);

        let disconnect_deadline = TestInstant::now() + Duration::from_secs(5);
        loop {
            match worker.progress(1000, operation, ManagementProgress::HoldStill) {
                Err(WorkerError::Management(ManagementError::NoActiveOperation)) => break,
                Ok(_) => {}
                Err(error) => return Err(error.into()),
            }
            if TestInstant::now() >= disconnect_deadline {
                return Err("disconnected Manager1 client retained its operation".into());
            }
            thread::sleep(Duration::from_millis(20));
        }

        bus.stop()?;
        let service_result = service_result_rx.recv_timeout(Duration::from_secs(5))?;
        assert!(service_result.is_err());
        service_thread.join().map_err(|_| "Manager1 service thread panicked")?;
        Ok(())
    }

    #[test]
    fn isolated_bus_runner_releases_name_on_explicit_shutdown()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut bus = PrivateBus::start()?;
        let manager = Manager1::new(
            ManagementCoordinator::new(ManagementConfig::default())?,
            AnyCallerBackend,
            SystemMonotonicClock,
        );
        let service_address = bus.address.clone();
        let (shutdown_tx, shutdown_rx) = futures_channel::oneshot::channel();
        let (result_tx, result_rx) = test_mpsc::channel();
        let service_thread = thread::spawn(move || {
            let result = futures_lite::future::block_on(async move {
                let connection = zbus::ConnectionBuilder::address(service_address.as_str())
                    .map_err(|error| error.to_string())?
                    .build()
                    .await
                    .map_err(|error| error.to_string())?;
                run_manager1_service_until_shutdown(connection, manager, async move {
                    let _shutdown_result = shutdown_rx.await;
                })
                .await
                .map_err(|error| error.to_string())
            });
            let _send_result = result_tx.send(result);
        });

        let client = zbus::blocking::connection::Builder::address(bus.address.as_str())?.build()?;
        let proxy = zbus::blocking::Proxy::new(
            &client,
            MANAGER_BUS_NAME,
            MANAGER_OBJECT_PATH,
            faceauth_management::MANAGER_INTERFACE,
        )?;
        let ready_deadline = TestInstant::now() + Duration::from_secs(5);
        loop {
            if proxy.call::<_, _, u16>("GetVersion", &()) == Ok(MANAGEMENT_SCHEMA_VERSION) {
                break;
            }
            if TestInstant::now() >= ready_deadline {
                return Err("Manager1 service did not become ready".into());
            }
            thread::sleep(Duration::from_millis(20));
        }

        shutdown_tx.send(()).map_err(|()| "Manager1 shutdown receiver disappeared")?;
        assert!(result_rx.recv_timeout(Duration::from_secs(5))?.is_ok());
        service_thread.join().map_err(|_| "Manager1 service thread panicked")?;
        drop(proxy);
        client.request_name(MANAGER_BUS_NAME)?;
        client.release_name(MANAGER_BUS_NAME)?;
        bus.stop()?;
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
