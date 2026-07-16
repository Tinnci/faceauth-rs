//! Fail-fast supervision for the daemon's independently blocking service boundaries.

use std::{
    future::Future,
    panic::{AssertUnwindSafe, catch_unwind},
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    task::{Context, Poll, Waker},
    thread,
    time::{Duration, Instant},
};

use faceauth_transport::{
    AcceptLoopConfig, AcceptLoopReport, PeerStream, SecureListener, TransportConfig,
    TransportError, run_accept_loop,
};
use thiserror::Error;

const MAX_SERVICE_NAME_BYTES: usize = 64;
const ABSOLUTE_MAX_SERVICES: usize = 16;

#[derive(Debug, Default)]
struct ShutdownState {
    requested: AtomicBool,
    waiters: Mutex<Vec<Waker>>,
}

/// Cloneable process-local cooperative shutdown signal.
///
/// This token is not serialized. Service tasks must check it at bounded wait or operation
/// boundaries and return after it is requested.
#[derive(Clone, Debug, Default)]
pub struct ShutdownToken {
    state: Arc<ShutdownState>,
}

impl ShutdownToken {
    /// Request shutdown for every clone of this token.
    ///
    /// Returns true only for the first caller that changed the state.
    #[must_use]
    pub fn request(&self) -> bool {
        let first = !self.state.requested.swap(true, Ordering::AcqRel);
        if first && let Ok(mut waiters) = self.state.waiters.lock() {
            for waiter in waiters.drain(..) {
                waiter.wake();
            }
        }
        first
    }

    /// Return whether cooperative shutdown has been requested.
    #[must_use]
    pub fn is_requested(&self) -> bool {
        self.state.requested.load(Ordering::Acquire)
    }

    /// Return a future that resolves when this token is requested.
    ///
    /// This bridges blocking supervisor ownership into asynchronous boundaries such as Manager1
    /// without exposing a serializable cancellation primitive.
    #[must_use]
    pub fn cancelled(&self) -> ShutdownFuture {
        ShutdownFuture { token: self.clone(), waker: None }
    }
}

/// Future resolved by a [`ShutdownToken`] request.
pub struct ShutdownFuture {
    token: ShutdownToken,
    waker: Option<Waker>,
}

impl Future for ShutdownFuture {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        if self.token.is_requested() {
            return Poll::Ready(());
        }
        let needs_registration =
            self.waker.as_ref().is_none_or(|waker| !waker.will_wake(context.waker()));
        if needs_registration {
            let state = Arc::clone(&self.token.state);
            let Ok(mut waiters) = state.waiters.lock() else {
                return Poll::Ready(());
            };
            if state.requested.load(Ordering::Acquire) {
                return Poll::Ready(());
            }
            let waker = context.waker().clone();
            waiters.push(waker.clone());
            drop(waiters);
            self.waker = Some(waker);
        }
        if self.token.is_requested() { Poll::Ready(()) } else { Poll::Pending }
    }
}

/// Resource bounds for one daemon supervisor run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SupervisorConfig {
    /// Maximum wait between external shutdown checks.
    pub poll_interval: Duration,
    /// Time allowed for all service tasks to cooperate after shutdown is requested.
    pub shutdown_grace: Duration,
    /// Maximum number of independently supervised service tasks.
    pub max_services: usize,
}

impl SupervisorConfig {
    /// Validate polling, shutdown, and task-count bounds.
    ///
    /// # Errors
    ///
    /// Returns [`SupervisorError::InvalidConfig`] for zero or excessive bounds.
    pub fn validate(self) -> Result<(), SupervisorError> {
        if !(Duration::from_millis(1)..=Duration::from_millis(100)).contains(&self.poll_interval)
            || !(Duration::from_millis(10)..=Duration::from_secs(30)).contains(&self.shutdown_grace)
            || !(1..=ABSOLUTE_MAX_SERVICES).contains(&self.max_services)
        {
            Err(SupervisorError::InvalidConfig)
        } else {
            Ok(())
        }
    }
}

impl Default for SupervisorConfig {
    fn default() -> Self {
        Self {
            poll_interval: Duration::from_millis(25),
            shutdown_grace: Duration::from_secs(5),
            max_services: 8,
        }
    }
}

/// Successful cooperative shutdown summary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SupervisorReport {
    /// Number of tasks started under supervision.
    pub services_started: usize,
    /// Number of tasks joined before the grace deadline.
    pub services_stopped: usize,
    /// Names in their observed completion order.
    pub completion_order: Vec<String>,
}

enum ServiceOutcome {
    Completed,
    Failed(String),
    Panicked,
}

struct ServiceEvent {
    index: usize,
    outcome: ServiceOutcome,
}

struct ServiceState {
    name: String,
    handle: Option<thread::JoinHandle<()>>,
    stopped: bool,
}

enum PendingFailure {
    Exited(String),
    Failed { service: String, detail: String },
    Panicked(String),
    EventChannelClosed,
}

impl PendingFailure {
    fn trigger(&self) -> String {
        match self {
            Self::Exited(service) => format!("{service} exited unexpectedly"),
            Self::Failed { service, .. } => format!("{service} failed"),
            Self::Panicked(service) => format!("{service} panicked"),
            Self::EventChannelClosed => "service event channel closed".to_owned(),
        }
    }

    fn into_error(self) -> SupervisorError {
        match self {
            Self::Exited(service) => SupervisorError::ServiceExited { service },
            Self::Failed { service, detail } => SupervisorError::ServiceFailed { service, detail },
            Self::Panicked(service) => SupervisorError::ServicePanicked { service },
            Self::EventChannelClosed => SupervisorError::EventChannelClosed,
        }
    }
}

/// Owns named daemon service threads and propagates any exit to all peers.
pub struct ServiceSupervisor {
    config: SupervisorConfig,
    shutdown: ShutdownToken,
    sender: Option<mpsc::SyncSender<ServiceEvent>>,
    receiver: mpsc::Receiver<ServiceEvent>,
    services: Vec<ServiceState>,
}

impl ServiceSupervisor {
    /// Create an empty bounded supervisor.
    ///
    /// # Errors
    ///
    /// Returns [`SupervisorError::InvalidConfig`] for unsafe resource bounds.
    pub fn new(config: SupervisorConfig) -> Result<Self, SupervisorError> {
        config.validate()?;
        let (sender, receiver) = mpsc::sync_channel(config.max_services);
        Ok(Self {
            config,
            shutdown: ShutdownToken::default(),
            sender: Some(sender),
            receiver,
            services: Vec::with_capacity(config.max_services),
        })
    }

    /// Return the shared cooperative shutdown token.
    #[must_use]
    pub fn shutdown_token(&self) -> ShutdownToken {
        self.shutdown.clone()
    }

    /// Spawn one named service task.
    ///
    /// Returning `Ok(())` before shutdown is itself fatal because a production boundary must not
    /// disappear silently. Returning an error or unwinding requests shutdown for every peer.
    ///
    /// # Errors
    ///
    /// Returns [`SupervisorError`] for invalid/duplicate names, excess capacity, prior shutdown,
    /// or operating-system thread creation failure.
    pub fn spawn(
        &mut self,
        name: impl Into<String>,
        task: impl FnOnce(ShutdownToken) -> Result<(), String> + Send + 'static,
    ) -> Result<(), SupervisorError> {
        let name = name.into();
        validate_service_name(&name)?;
        if self.services.iter().any(|service| service.name == name) {
            return Err(SupervisorError::DuplicateService { service: name });
        }
        if self.services.len() >= self.config.max_services {
            return Err(SupervisorError::TooManyServices { maximum: self.config.max_services });
        }
        if self.shutdown.is_requested() {
            return Err(SupervisorError::ShutdownAlreadyRequested);
        }

        let index = self.services.len();
        let sender = self.sender.as_ref().ok_or(SupervisorError::ShutdownAlreadyRequested)?.clone();
        let task_shutdown = self.shutdown.clone();
        let thread_name = format!("faceauth-{name}");
        let handle = thread::Builder::new()
            .name(thread_name)
            .spawn(move || {
                let outcome = match catch_unwind(AssertUnwindSafe(|| task(task_shutdown))) {
                    Ok(Ok(())) => ServiceOutcome::Completed,
                    Ok(Err(detail)) => ServiceOutcome::Failed(detail),
                    Err(_) => ServiceOutcome::Panicked,
                };
                let _send_result = sender.send(ServiceEvent { index, outcome });
            })
            .map_err(|source| SupervisorError::Spawn { service: name.clone(), source })?;
        self.services.push(ServiceState { name, handle: Some(handle), stopped: false });
        Ok(())
    }

    /// Run until an external shutdown request or any service exit, then join all cooperating tasks.
    ///
    /// The external callback is checked at the configured bounded poll interval. Any service error,
    /// panic, event-channel loss, or clean pre-shutdown exit requests global shutdown. Tasks that do
    /// not stop before the grace deadline are reported and detached so the process can fail closed.
    ///
    /// # Errors
    ///
    /// Returns [`SupervisorError`] when no service exists, a service exits unexpectedly, a task
    /// fails or panics, the event channel closes, or shutdown exceeds its grace deadline.
    pub fn run_until(
        mut self,
        mut external_shutdown: impl FnMut() -> bool,
    ) -> Result<SupervisorReport, SupervisorError> {
        if self.services.is_empty() {
            return Err(SupervisorError::NoServices);
        }
        drop(self.sender.take());
        let services_started = self.services.len();
        let mut services_stopped = 0_usize;
        let mut completion_order = Vec::with_capacity(services_started);
        let mut shutdown_started = None;
        let mut pending_failure = None;

        loop {
            if !self.shutdown.is_requested() && external_shutdown() {
                let _first_request = self.shutdown.request();
                shutdown_started = Some(Instant::now());
            }

            match self.receiver.recv_timeout(self.config.poll_interval) {
                Ok(event) => {
                    let service = self
                        .services
                        .get_mut(event.index)
                        .ok_or(SupervisorError::InvalidServiceEvent)?;
                    if service.stopped {
                        return Err(SupervisorError::InvalidServiceEvent);
                    }
                    service.stopped = true;
                    if let Some(handle) = service.handle.take()
                        && handle.join().is_err()
                        && !matches!(&event.outcome, ServiceOutcome::Panicked)
                    {
                        return Err(SupervisorError::InvalidServiceEvent);
                    }
                    services_stopped += 1;
                    completion_order.push(service.name.clone());
                    let failure = match event.outcome {
                        ServiceOutcome::Completed if !self.shutdown.is_requested() => {
                            Some(PendingFailure::Exited(service.name.clone()))
                        }
                        ServiceOutcome::Completed => None,
                        ServiceOutcome::Failed(detail) => {
                            Some(PendingFailure::Failed { service: service.name.clone(), detail })
                        }
                        ServiceOutcome::Panicked => {
                            Some(PendingFailure::Panicked(service.name.clone()))
                        }
                    };
                    if pending_failure.is_none() {
                        pending_failure = failure;
                    }
                    if pending_failure.is_some() && !self.shutdown.is_requested() {
                        let _first_request = self.shutdown.request();
                        shutdown_started = Some(Instant::now());
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    if pending_failure.is_none() {
                        pending_failure = Some(PendingFailure::EventChannelClosed);
                    }
                    if !self.shutdown.is_requested() {
                        let _first_request = self.shutdown.request();
                        shutdown_started = Some(Instant::now());
                    }
                }
            }

            if services_stopped == services_started {
                return pending_failure.map_or_else(
                    || {
                        Ok(SupervisorReport {
                            services_started,
                            services_stopped,
                            completion_order,
                        })
                    },
                    |failure| Err(failure.into_error()),
                );
            }

            if self.shutdown.is_requested() && shutdown_started.is_none() {
                shutdown_started = Some(Instant::now());
            }
            if shutdown_started
                .is_some_and(|started| started.elapsed() >= self.config.shutdown_grace)
            {
                let services = self
                    .services
                    .iter()
                    .filter(|service| !service.stopped)
                    .map(|service| service.name.clone())
                    .collect();
                return Err(SupervisorError::ShutdownTimeout {
                    trigger: pending_failure.as_ref().map(PendingFailure::trigger),
                    services,
                });
            }
        }
    }
}

impl Drop for ServiceSupervisor {
    fn drop(&mut self) {
        let _first_request = self.shutdown.request();
    }
}

/// Run the peer-credentialed authentication listener under a supervisor shutdown token.
///
/// The owned listener performs exact inode cleanup on every return or unwind. Handler logic still
/// owns the complete authorized capture/inference transaction and must not report biometric success
/// without running that pipeline.
///
/// # Errors
///
/// Returns [`TransportError`] for invalid listener-loop policy or accept/peer initialization
/// failures beyond the configured budget.
pub fn run_supervised_authentication_listener(
    listener: SecureListener,
    transport: TransportConfig,
    config: AcceptLoopConfig,
    shutdown: &ShutdownToken,
    handler: impl FnMut(PeerStream),
) -> Result<AcceptLoopReport, TransportError> {
    let report = run_accept_loop(&listener, transport, config, || shutdown.is_requested(), handler);
    drop(listener);
    report
}

fn validate_service_name(name: &str) -> Result<(), SupervisorError> {
    if name.is_empty()
        || name.len() > MAX_SERVICE_NAME_BYTES
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    {
        Err(SupervisorError::InvalidServiceName)
    } else {
        Ok(())
    }
}

/// Supervisor configuration, task, or lifecycle failure.
#[derive(Debug, Error)]
pub enum SupervisorError {
    /// Polling, grace-period, or capacity bounds are invalid.
    #[error("invalid daemon supervisor configuration")]
    InvalidConfig,
    /// Service names must be bounded lowercase ASCII labels.
    #[error("invalid supervised service name")]
    InvalidServiceName,
    /// A service name can identify only one task.
    #[error("duplicate supervised service name {service}")]
    DuplicateService {
        /// Repeated name.
        service: String,
    },
    /// Configured task capacity was exceeded.
    #[error("supervisor service capacity {maximum} was exceeded")]
    TooManyServices {
        /// Configured maximum.
        maximum: usize,
    },
    /// No service may be added after global shutdown begins.
    #[error("cannot spawn a service after shutdown was requested")]
    ShutdownAlreadyRequested,
    /// The operating system refused to create a named service thread.
    #[error("unable to spawn supervised service {service}: {source}")]
    Spawn {
        /// Service name.
        service: String,
        /// Thread creation error.
        source: std::io::Error,
    },
    /// Running an empty supervisor is a configuration error.
    #[error("daemon supervisor has no services")]
    NoServices,
    /// A service returned successfully before shutdown.
    #[error("supervised service {service} exited before shutdown")]
    ServiceExited {
        /// Service name.
        service: String,
    },
    /// A service returned a fatal error.
    #[error("supervised service {service} failed: {detail}")]
    ServiceFailed {
        /// Service name.
        service: String,
        /// Sanitized task error.
        detail: String,
    },
    /// A service unwound; biometric or management work cannot continue safely.
    #[error("supervised service {service} panicked")]
    ServicePanicked {
        /// Service name.
        service: String,
    },
    /// Internal event delivery ended while services remained live.
    #[error("supervisor event channel closed unexpectedly")]
    EventChannelClosed,
    /// A thread reported an impossible or duplicate event.
    #[error("supervisor received an invalid service event")]
    InvalidServiceEvent,
    /// One or more tasks ignored shutdown past the configured deadline.
    #[error("daemon shutdown timed out; trigger {trigger:?}; remaining services {services:?}")]
    ShutdownTimeout {
        /// Fatal trigger, when shutdown was not externally requested.
        trigger: Option<String>,
        /// Tasks not joined by the deadline.
        services: Vec<String>,
    },
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;

    use super::*;

    fn test_config() -> SupervisorConfig {
        SupervisorConfig {
            poll_interval: Duration::from_millis(1),
            shutdown_grace: Duration::from_millis(100),
            max_services: 4,
        }
    }

    #[test]
    fn external_shutdown_is_broadcast_and_all_tasks_are_joined()
    -> Result<(), Box<dyn std::error::Error>> {
        let stopped = Arc::new(AtomicUsize::new(0));
        let mut supervisor = ServiceSupervisor::new(test_config())?;
        for name in ["authentication", "manager1"] {
            let stopped = Arc::clone(&stopped);
            supervisor.spawn(name, move |shutdown| {
                while !shutdown.is_requested() {
                    thread::sleep(Duration::from_millis(1));
                }
                stopped.fetch_add(1, Ordering::AcqRel);
                Ok(())
            })?;
        }
        let mut polls = 0_u8;
        let report = supervisor.run_until(|| {
            polls = polls.saturating_add(1);
            polls >= 3
        })?;
        assert_eq!(report.services_started, 2);
        assert_eq!(report.services_stopped, 2);
        assert_eq!(stopped.load(Ordering::Acquire), 2);
        Ok(())
    }

    #[test]
    fn asynchronous_shutdown_waiter_is_woken() -> Result<(), Box<dyn std::error::Error>> {
        let token = ShutdownToken::default();
        let waiter = token.cancelled();
        let (completed_tx, completed_rx) = mpsc::channel();
        let thread = thread::spawn(move || {
            futures_lite::future::block_on(waiter);
            let _send_result = completed_tx.send(());
        });
        assert!(token.request());
        completed_rx.recv_timeout(Duration::from_secs(1))?;
        thread.join().map_err(|_| "shutdown waiter thread panicked")?;
        Ok(())
    }

    #[test]
    fn one_failure_stops_and_joins_its_peer() -> Result<(), Box<dyn std::error::Error>> {
        let peer_stopped = Arc::new(AtomicBool::new(false));
        let mut supervisor = ServiceSupervisor::new(test_config())?;
        let peer_state = Arc::clone(&peer_stopped);
        supervisor.spawn("authentication", move |shutdown| {
            while !shutdown.is_requested() {
                thread::sleep(Duration::from_millis(1));
            }
            peer_state.store(true, Ordering::Release);
            Ok(())
        })?;
        supervisor.spawn("manager1", |_shutdown| Err("bus lifecycle ended".to_owned()))?;

        assert!(matches!(
            supervisor.run_until(|| false),
            Err(SupervisorError::ServiceFailed { ref service, .. }) if service == "manager1"
        ));
        assert!(peer_stopped.load(Ordering::Acquire));
        Ok(())
    }

    #[test]
    fn clean_pre_shutdown_exit_is_fatal() -> Result<(), Box<dyn std::error::Error>> {
        let supervisor = {
            let mut supervisor = ServiceSupervisor::new(test_config())?;
            supervisor.spawn("authentication", |_shutdown| Ok(()))?;
            supervisor
        };
        assert!(matches!(
            supervisor.run_until(|| false),
            Err(SupervisorError::ServiceExited { ref service }) if service == "authentication"
        ));
        Ok(())
    }

    #[test]
    fn service_unwind_is_caught_and_stops_its_peer() -> Result<(), Box<dyn std::error::Error>> {
        let peer_stopped = Arc::new(AtomicBool::new(false));
        let mut supervisor = ServiceSupervisor::new(test_config())?;
        let peer_state = Arc::clone(&peer_stopped);
        supervisor.spawn("authentication", move |shutdown| {
            while !shutdown.is_requested() {
                thread::sleep(Duration::from_millis(1));
            }
            peer_state.store(true, Ordering::Release);
            Ok(())
        })?;
        supervisor.spawn("manager1", |_shutdown| {
            std::panic::resume_unwind(Box::new("supervised test unwind"));
        })?;

        assert!(matches!(
            supervisor.run_until(|| false),
            Err(SupervisorError::ServicePanicked { ref service }) if service == "manager1"
        ));
        assert!(peer_stopped.load(Ordering::Acquire));
        Ok(())
    }

    #[test]
    fn duplicate_names_and_uncooperative_shutdown_fail_closed()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut supervisor = ServiceSupervisor::new(SupervisorConfig {
            shutdown_grace: Duration::from_millis(10),
            ..test_config()
        })?;
        supervisor.spawn("authentication", |_shutdown| {
            thread::sleep(Duration::from_millis(100));
            Ok(())
        })?;
        assert!(matches!(
            supervisor.spawn("authentication", |_shutdown| Ok(())),
            Err(SupervisorError::DuplicateService { .. })
        ));
        assert!(matches!(
            supervisor.run_until(|| true),
            Err(SupervisorError::ShutdownTimeout { ref services, .. })
                if services == &["authentication".to_owned()]
        ));
        Ok(())
    }
}
