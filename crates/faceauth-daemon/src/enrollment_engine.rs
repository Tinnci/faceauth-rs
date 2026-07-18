//! Single-capacity enrollment worker with shared resource ownership and atomic template commit.

use std::{
    ops::ControlFlow,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::Duration,
};

use faceauth_enrollment::{
    EnrollmentConfig, EnrollmentError, EnrollmentObservation, EnrollmentSession,
};
use faceauth_liveness::ChallengeConfig;
use faceauth_management::{
    AuthorizedEnrollment, ManagementError, ManagementProgress, ManagementResult, ManagementUpdate,
    OperationId,
};
use faceauth_management_dbus::{
    BackendError, EnrollmentOperationController, ManagementWorkerHandle, WorkerError,
};
use faceauth_storage::{EncryptedTemplateStore, KeyProvider, StorageError, TemplateRecord};
use thiserror::Error;

use crate::{
    BiometricResourceArbiter, BiometricResourceError, BiometricResourceLease,
    BiometricResourceOwner, PassivePadPolicy, ShutdownToken,
    observation::{
        DerivedBiometricObservation, ObservationFailure, ObservationProgress,
        SharedProductionObservationPipeline,
    },
};

const REQUEST_QUEUE_CAPACITY: usize = 1;
const UPDATE_QUEUE_CAPACITY: usize = 34;
const MAX_PROGRESS_UPDATES: usize = 32;
const WORKER_POLL_INTERVAL: Duration = Duration::from_millis(25);

/// Private cancellation signal shared by Manager1 ownership and the enrollment worker.
#[derive(Clone, Debug, Default)]
pub struct EnrollmentCancellation {
    cancelled: Arc<AtomicBool>,
}

impl EnrollmentCancellation {
    /// Signal cancellation before emitting a public terminal update.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Whether the exact operation was cancelled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

/// Exact authorized enrollment operation consumed by the engine worker.
pub struct EnrollmentJob {
    operation_id: OperationId,
    authorization: AuthorizedEnrollment,
    config: EnrollmentConfig,
    issued_at_micros: u64,
    cancellation: EnrollmentCancellation,
    shutdown: ShutdownToken,
}

impl EnrollmentJob {
    /// Construct a job from the exact Manager1 authorization and operation identity.
    #[must_use]
    pub fn new(
        operation_id: OperationId,
        authorization: AuthorizedEnrollment,
        config: EnrollmentConfig,
        issued_at_micros: u64,
        cancellation: EnrollmentCancellation,
    ) -> Self {
        Self {
            operation_id,
            authorization,
            config,
            issued_at_micros,
            cancellation,
            shutdown: ShutdownToken::default(),
        }
    }

    /// Exact management operation.
    #[must_use]
    pub const fn operation_id(&self) -> OperationId {
        self.operation_id
    }

    /// Numeric account authorized by the root Polkit broker.
    #[must_use]
    pub const fn target_uid(&self) -> u32 {
        self.authorization.target_uid()
    }

    /// Whether Manager1 cancellation or daemon shutdown was observed.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled() || self.shutdown.is_requested()
    }

    /// Start the bounded derived-sample state machine for this exact UID and issue time.
    ///
    /// # Errors
    ///
    /// Returns [`EnrollmentError`] if the calibrated enrollment configuration is invalid.
    pub fn start_session(&self) -> Result<EnrollmentSession, EnrollmentError> {
        EnrollmentSession::start(self.config, self.target_uid(), self.issued_at_micros)
    }
}

/// Engine implementation that owns capture streams and mutable inference sessions.
pub trait EnrollmentEngine: Send + 'static {
    /// Capture, validate, and aggregate one exact enrollment into a derived template record.
    ///
    /// Implementations must poll [`EnrollmentJob::is_cancelled`] at every bounded capture,
    /// preprocessing, inference, challenge, and aggregation boundary. No raw image may cross the
    /// return boundary.
    ///
    /// # Errors
    ///
    /// Returns a sanitized failure without exposing frames, scores, landmarks, or embeddings.
    fn enroll(
        &mut self,
        job: &EnrollmentJob,
        progress: &mut dyn FnMut(ManagementProgress) -> Result<(), EnrollmentEngineFailure>,
    ) -> Result<TemplateRecord, EnrollmentEngineFailure>;
}

/// Real enrollment adapter over the shared dual-camera, six-model observation pipeline.
///
/// Authentication and enrollment hold clones of the same pipeline handle. The daemon-wide
/// biometric arbiter admits only one operation, and `try_lock` turns any ownership invariant
/// violation into a fail-closed internal error instead of queueing.
pub struct ProductionEnrollmentEngine {
    pipeline: SharedProductionObservationPipeline,
    challenge: ChallengeConfig,
    passive_pad: PassivePadPolicy,
}

impl ProductionEnrollmentEngine {
    pub(crate) fn from_shared(
        pipeline: SharedProductionObservationPipeline,
        challenge: ChallengeConfig,
        passive_pad: PassivePadPolicy,
    ) -> Result<Self, ProductionEnrollmentEngineError> {
        challenge.validate().map_err(|_| ProductionEnrollmentEngineError::InvalidPolicy)?;
        passive_pad.validate().map_err(|_| ProductionEnrollmentEngineError::InvalidPolicy)?;
        Ok(Self { pipeline, challenge, passive_pad })
    }
}

struct EnrollmentAccumulator<'a> {
    session: Option<EnrollmentSession>,
    passive_pad: &'a PassivePadPolicy,
    minimum_sample_interval_micros: u64,
    last_accepted_timestamp: Option<u64>,
}

impl<'a> EnrollmentAccumulator<'a> {
    const fn new(
        session: EnrollmentSession,
        passive_pad: &'a PassivePadPolicy,
        minimum_sample_interval_micros: u64,
    ) -> Self {
        Self {
            session: Some(session),
            passive_pad,
            minimum_sample_interval_micros,
            last_accepted_timestamp: None,
        }
    }

    fn observe(
        &mut self,
        observation: &DerivedBiometricObservation,
    ) -> Result<ControlFlow<TemplateRecord>, ObservationFailure> {
        if self.last_accepted_timestamp.is_some_and(|previous| {
            observation.completed_at_micros.saturating_sub(previous)
                < self.minimum_sample_interval_micros
        }) {
            return Ok(ControlFlow::Continue(()));
        }
        let passive_liveness_passed = self
            .passive_pad
            .accepts(&observation.passive_liveness)
            .map_err(|_| ObservationFailure::Inference)?;
        let active = self.session.as_mut().ok_or(ObservationFailure::Internal)?;
        match active.observe(EnrollmentObservation {
            embedding: &observation.embedding,
            quality: observation.quality,
            passive_liveness_passed,
            active_liveness_passed: true,
            timestamp_micros: observation.completed_at_micros,
            yaw_degrees: observation.yaw_degrees,
        }) {
            Ok(()) => {
                self.last_accepted_timestamp = Some(observation.completed_at_micros);
            }
            Err(EnrollmentError::InsufficientQuality) => {
                return Ok(ControlFlow::Continue(()));
            }
            Err(_) => return Err(ObservationFailure::InvalidEvidence),
        }
        if active.ready_to_finish() {
            let completed = self.session.take().ok_or(ObservationFailure::Internal)?;
            let record = completed
                .finish(observation.completed_at_micros)
                .map_err(|_| ObservationFailure::InvalidEvidence)?;
            return Ok(ControlFlow::Break(record));
        }
        if active.is_full() {
            return Err(ObservationFailure::AttemptLimit);
        }
        Ok(ControlFlow::Continue(()))
    }
}

impl EnrollmentEngine for ProductionEnrollmentEngine {
    fn enroll(
        &mut self,
        job: &EnrollmentJob,
        progress: &mut dyn FnMut(ManagementProgress) -> Result<(), EnrollmentEngineFailure>,
    ) -> Result<TemplateRecord, EnrollmentEngineFailure> {
        if job.is_cancelled() {
            return Err(EnrollmentEngineFailure::Cancelled);
        }
        progress(ManagementProgress::Preparing)?;
        let session =
            job.start_session().map_err(|_| EnrollmentEngineFailure::InsufficientEvidence)?;
        let mut accumulator = EnrollmentAccumulator::new(
            session,
            &self.passive_pad,
            job.config.minimum_sample_interval_micros,
        );
        let mut pipeline =
            self.pipeline.try_lock().map_err(|_| EnrollmentEngineFailure::Internal)?;
        let mut cancelled = || job.is_cancelled();
        let mut relay_progress = |update| {
            progress(enrollment_progress(update)).map_err(|_| {
                if job.is_cancelled() {
                    ObservationFailure::Cancelled
                } else {
                    ObservationFailure::Internal
                }
            })
        };
        let mut consume = |observation: DerivedBiometricObservation| {
            if job.is_cancelled() {
                return Err(ObservationFailure::Cancelled);
            }
            let result = accumulator.observe(&observation)?;
            if result.is_break() && job.is_cancelled() {
                return Err(ObservationFailure::Cancelled);
            }
            Ok(result)
        };
        pipeline
            .run_session(self.challenge, &mut cancelled, &mut relay_progress, &mut consume)
            .map_err(enrollment_observation_failure)
    }
}

const fn enrollment_progress(progress: ObservationProgress) -> ManagementProgress {
    match progress {
        ObservationProgress::PositionFace => ManagementProgress::PositionFace,
        ObservationProgress::HoldStill => ManagementProgress::HoldStill,
        ObservationProgress::Blink => ManagementProgress::Blink,
        ObservationProgress::TurnLeft => ManagementProgress::TurnLeft,
        ObservationProgress::TurnRight => ManagementProgress::TurnRight,
        ObservationProgress::ReturnToCenter => ManagementProgress::ReturnToCenter,
        ObservationProgress::Processing => ManagementProgress::Processing,
    }
}

const fn enrollment_observation_failure(error: ObservationFailure) -> EnrollmentEngineFailure {
    match error {
        ObservationFailure::Cancelled => EnrollmentEngineFailure::Cancelled,
        ObservationFailure::Capture => EnrollmentEngineFailure::Capture,
        ObservationFailure::Inference => EnrollmentEngineFailure::Inference,
        ObservationFailure::Liveness
        | ObservationFailure::InvalidEvidence
        | ObservationFailure::AttemptLimit => EnrollmentEngineFailure::InsufficientEvidence,
        ObservationFailure::Internal => EnrollmentEngineFailure::Internal,
    }
}

/// Production enrollment adapter construction failure.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum ProductionEnrollmentEngineError {
    /// Active-liveness or passive-PAD calibration is structurally invalid.
    #[error("production enrollment engine policy is invalid")]
    InvalidPolicy,
}

/// Atomic destination for a completed derived enrollment template.
pub trait EnrollmentTemplateSink: Send + Sync + 'static {
    /// Validate and atomically commit one record.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] if authenticated encryption or atomic replacement fails.
    fn save_template(&self, record: &TemplateRecord) -> Result<(), StorageError>;
}

impl<K> EnrollmentTemplateSink for EncryptedTemplateStore<K>
where
    K: KeyProvider + Send + Sync + 'static,
{
    fn save_template(&self, record: &TemplateRecord) -> Result<(), StorageError> {
        self.save(record)
    }
}

/// Sanitized enrollment worker failure.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum EnrollmentEngineFailure {
    /// Manager1 cancellation or daemon shutdown was observed.
    #[error("enrollment was cancelled")]
    Cancelled,
    /// Required cameras or paired capture were unavailable.
    #[error("enrollment capture failed")]
    Capture,
    /// Model preprocessing or bounded inference failed.
    #[error("enrollment inference failed")]
    Inference,
    /// Quality, liveness, consistency, or pose coverage did not produce a usable template.
    #[error("enrollment evidence was insufficient")]
    InsufficientEvidence,
    /// Authenticated template storage failed.
    #[error("enrollment template storage failed")]
    Storage,
    /// Internal coordination failed.
    #[error("enrollment internal coordination failed")]
    Internal,
}

/// One operation-bound update from the worker.
pub enum EnrollmentEngineUpdate {
    /// Closed non-biometric progress.
    Progress {
        /// Exact operation.
        operation_id: OperationId,
        /// Safe management progress code.
        progress: ManagementProgress,
    },
    /// Terminal result emitted only after template storage succeeds or fails.
    Completed {
        /// Exact operation.
        operation_id: OperationId,
        /// Empty success or sanitized failure; no template crosses the channel.
        result: Result<(), EnrollmentEngineFailure>,
    },
}

struct EnrollmentRequest {
    job: EnrollmentJob,
    updates: mpsc::SyncSender<EnrollmentEngineUpdate>,
    _resource_lease: BiometricResourceLease,
}

/// Cloneable non-blocking submission half.
#[derive(Clone)]
pub struct EnrollmentEngineClient {
    requests: mpsc::SyncSender<EnrollmentRequest>,
    active: Arc<AtomicBool>,
    resources: BiometricResourceArbiter,
}

impl EnrollmentEngineClient {
    /// Submit one exact authorized operation without queueing behind other biometric work.
    ///
    /// # Errors
    ///
    /// Returns [`EnrollmentEngineSubmitError`] for busy resources or a stopped worker.
    pub fn submit(
        &self,
        job: EnrollmentJob,
    ) -> Result<EnrollmentEngineHandle, EnrollmentEngineSubmitError> {
        let lease = self
            .resources
            .try_acquire(BiometricResourceOwner::Enrollment(job.operation_id))
            .map_err(|error| match error {
                BiometricResourceError::Busy { .. } => EnrollmentEngineSubmitError::Busy,
                BiometricResourceError::Unavailable => {
                    EnrollmentEngineSubmitError::ResourceUnavailable
                }
            })?;
        if self.active.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
            return Err(EnrollmentEngineSubmitError::Busy);
        }
        let operation_id = job.operation_id;
        let (updates, receiver) = mpsc::sync_channel(UPDATE_QUEUE_CAPACITY);
        if self
            .requests
            .try_send(EnrollmentRequest { job, updates, _resource_lease: lease })
            .is_err()
        {
            self.active.store(false, Ordering::Release);
            return Err(EnrollmentEngineSubmitError::WorkerStopped);
        }
        Ok(EnrollmentEngineHandle { operation_id, receiver })
    }
}

/// Receiving half for an exact enrollment operation.
pub struct EnrollmentEngineHandle {
    operation_id: OperationId,
    receiver: mpsc::Receiver<EnrollmentEngineUpdate>,
}

struct ControlledOperation {
    operation_id: OperationId,
    cancellation: EnrollmentCancellation,
    handle: Option<EnrollmentEngineHandle>,
}

/// Manager1 controller that submits exact operations and preserves cancellation ownership.
///
/// A daemon coordinator takes the worker handle once, relays its closed updates through
/// `ManagementWorkerHandle`, and calls [`Self::finish`] after the terminal update. The private
/// cancellation signal remains registered throughout that relay.
#[derive(Clone)]
pub struct EnrollmentControllerBridge {
    client: EnrollmentEngineClient,
    config: EnrollmentConfig,
    active: Arc<std::sync::Mutex<Option<ControlledOperation>>>,
}

impl EnrollmentControllerBridge {
    /// Bind Manager1 operation starts to one enrollment engine client and calibrated policy.
    #[must_use]
    pub fn new(client: EnrollmentEngineClient, config: EnrollmentConfig) -> Self {
        Self { client, config, active: Arc::new(std::sync::Mutex::new(None)) }
    }

    /// Take the worker update handle exactly once for the expected operation.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] for an absent, mismatched, or already-taken handle.
    pub fn take_handle(
        &self,
        operation_id: OperationId,
    ) -> Result<EnrollmentEngineHandle, BackendError> {
        let mut active = self
            .active
            .lock()
            .map_err(|_| BackendError::new("enrollment controller lock is poisoned"))?;
        let operation = active
            .as_mut()
            .filter(|operation| operation.operation_id == operation_id)
            .ok_or_else(|| BackendError::new("enrollment operation is not active"))?;
        let handle = operation
            .handle
            .take()
            .ok_or_else(|| BackendError::new("enrollment operation handle was already taken"))?;
        drop(active);
        Ok(handle)
    }

    /// Clear private cancellation state after the exact terminal result was relayed.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] for an absent or mismatched operation.
    pub fn finish(&self, operation_id: OperationId) -> Result<(), BackendError> {
        let mut active = self
            .active
            .lock()
            .map_err(|_| BackendError::new("enrollment controller lock is poisoned"))?;
        if active.as_ref().is_none_or(|operation| operation.operation_id != operation_id) {
            return Err(BackendError::new("enrollment operation is not active"));
        }
        *active = None;
        drop(active);
        Ok(())
    }

    /// Relay the exact worker stream through the management coordinator and a safe signal sink.
    ///
    /// The coordinator rechecks UID, operation ID, and deadline before every emitted update. If a
    /// client cancellation already consumed the public operation, the worker's late update is
    /// discarded and private cancellation state is cleared without a duplicate terminal signal.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] for worker-channel loss, invalid lifecycle transitions, or signal
    /// sink failure.
    pub fn relay(
        &self,
        target_uid: u32,
        operation_id: OperationId,
        management: &ManagementWorkerHandle,
        mut emit: impl FnMut(ManagementUpdate) -> Result<(), BackendError>,
    ) -> Result<(), BackendError> {
        let handle = self.take_handle(operation_id)?;
        loop {
            let update = handle
                .recv()
                .map_err(|_| BackendError::new("enrollment worker update channel closed"))?;
            let public = match update {
                EnrollmentEngineUpdate::Progress { operation_id: actual, progress }
                    if actual == operation_id =>
                {
                    match management.progress(target_uid, operation_id, progress) {
                        Ok(update) => update,
                        Err(WorkerError::Management(ManagementError::NoActiveOperation)) => {
                            self.finish(operation_id)?;
                            return Ok(());
                        }
                        Err(error) => return Err(BackendError::new(error.to_string())),
                    }
                }
                EnrollmentEngineUpdate::Completed { operation_id: actual, result }
                    if actual == operation_id =>
                {
                    let result = match result {
                        Ok(()) => ManagementResult::Completed,
                        Err(EnrollmentEngineFailure::Cancelled) => ManagementResult::Cancelled,
                        Err(_) => ManagementResult::Failed,
                    };
                    match management.complete(target_uid, operation_id, result) {
                        Ok(update) => update,
                        Err(WorkerError::Management(ManagementError::NoActiveOperation)) => {
                            self.finish(operation_id)?;
                            return Ok(());
                        }
                        Err(error) => return Err(BackendError::new(error.to_string())),
                    }
                }
                _ => return Err(BackendError::new("enrollment worker update is misbound")),
            };
            let terminal = matches!(public, ManagementUpdate::Completed { .. });
            emit(public)?;
            if terminal {
                self.finish(operation_id)?;
                return Ok(());
            }
        }
    }
}

impl EnrollmentOperationController for EnrollmentControllerBridge {
    fn start(
        &self,
        authorization: AuthorizedEnrollment,
        operation_id: OperationId,
        issued_at_micros: u64,
    ) -> Result<(), BackendError> {
        let mut active = self
            .active
            .lock()
            .map_err(|_| BackendError::new("enrollment controller lock is poisoned"))?;
        if active.is_some() {
            return Err(BackendError::new("another enrollment operation is active"));
        }
        let cancellation = EnrollmentCancellation::default();
        let job = EnrollmentJob::new(
            operation_id,
            authorization,
            self.config,
            issued_at_micros,
            cancellation.clone(),
        );
        let handle =
            self.client.submit(job).map_err(|error| BackendError::new(error.to_string()))?;
        *active = Some(ControlledOperation { operation_id, cancellation, handle: Some(handle) });
        drop(active);
        Ok(())
    }

    fn cancel(&self, operation_id: OperationId) -> Result<(), BackendError> {
        let active = self
            .active
            .lock()
            .map_err(|_| BackendError::new("enrollment controller lock is poisoned"))?;
        let cancellation = active
            .as_ref()
            .filter(|operation| operation.operation_id == operation_id)
            .map(|operation| operation.cancellation.clone())
            .ok_or_else(|| BackendError::new("enrollment operation is not active"))?;
        drop(active);
        cancellation.cancel();
        Ok(())
    }
}

impl EnrollmentEngineHandle {
    /// Exact submitted operation.
    #[must_use]
    pub const fn operation_id(&self) -> OperationId {
        self.operation_id
    }

    /// Wait for the next worker update.
    ///
    /// # Errors
    ///
    /// Returns [`mpsc::RecvError`] if supervision ended without a terminal update.
    pub fn recv(&self) -> Result<EnrollmentEngineUpdate, mpsc::RecvError> {
        self.receiver.recv()
    }
}

/// Supervised worker that owns the mutable engine and authenticated template sink.
pub struct EnrollmentEngineService<E, S> {
    engine: E,
    sink: S,
    requests: mpsc::Receiver<EnrollmentRequest>,
    active: Arc<AtomicBool>,
}

impl<E, S> EnrollmentEngineService<E, S>
where
    E: EnrollmentEngine,
    S: EnrollmentTemplateSink,
{
    /// Create a service using the daemon-wide arbiter shared with authentication.
    #[must_use]
    pub fn new(
        engine: E,
        sink: S,
        resources: BiometricResourceArbiter,
    ) -> (EnrollmentEngineClient, Self) {
        let (requests, receiver) = mpsc::sync_channel(REQUEST_QUEUE_CAPACITY);
        let active = Arc::new(AtomicBool::new(false));
        (
            EnrollmentEngineClient { requests, active: Arc::clone(&active), resources },
            Self { engine, sink, requests: receiver, active },
        )
    }

    /// Run until daemon shutdown, failing if coordination channels disappear unexpectedly.
    ///
    /// # Errors
    ///
    /// Returns [`EnrollmentEngineServiceError`] for request or result channel loss.
    pub fn run(mut self, shutdown: &ShutdownToken) -> Result<(), EnrollmentEngineServiceError> {
        loop {
            if shutdown.is_requested() {
                return Ok(());
            }
            match self.requests.recv_timeout(WORKER_POLL_INTERVAL) {
                Ok(request) => self.run_job(request, shutdown)?,
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(EnrollmentEngineServiceError::RequestChannelClosed);
                }
            }
        }
    }

    fn run_job(
        &mut self,
        mut request: EnrollmentRequest,
        shutdown: &ShutdownToken,
    ) -> Result<(), EnrollmentEngineServiceError> {
        request.job.shutdown = shutdown.clone();
        let operation_id = request.job.operation_id;
        let mut progress_count = 0_usize;
        let mut progress = |progress| {
            if request.job.is_cancelled() || progress_count >= MAX_PROGRESS_UPDATES {
                return Err(if request.job.is_cancelled() {
                    EnrollmentEngineFailure::Cancelled
                } else {
                    EnrollmentEngineFailure::Internal
                });
            }
            request
                .updates
                .try_send(EnrollmentEngineUpdate::Progress { operation_id, progress })
                .map_err(|_| EnrollmentEngineFailure::Internal)?;
            progress_count += 1;
            Ok(())
        };
        let result = if request.job.is_cancelled() {
            Err(EnrollmentEngineFailure::Cancelled)
        } else {
            self.engine.enroll(&request.job, &mut progress)
        };
        let result = match result {
            Ok(record) if request.job.is_cancelled() => {
                let _ = record;
                Err(EnrollmentEngineFailure::Cancelled)
            }
            Ok(record) if record.uid != request.job.target_uid() || record.validate().is_err() => {
                Err(EnrollmentEngineFailure::Internal)
            }
            Ok(record) => {
                self.sink.save_template(&record).map_err(|_| EnrollmentEngineFailure::Storage)
            }
            Err(error) => Err(error),
        };
        let send_result =
            request.updates.send(EnrollmentEngineUpdate::Completed { operation_id, result });
        self.active.store(false, Ordering::Release);
        send_result.map_err(|_| EnrollmentEngineServiceError::ResultReceiverClosed)
    }
}

/// Non-blocking submission failure.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum EnrollmentEngineSubmitError {
    /// Authentication or another enrollment owns the shared biometric pipeline.
    #[error("enrollment engine is busy")]
    Busy,
    /// Shared resource ownership state is unavailable.
    #[error("biometric resources are unavailable")]
    ResourceUnavailable,
    /// Supervision stopped the worker.
    #[error("enrollment engine worker has stopped")]
    WorkerStopped,
}

/// Fatal worker lifecycle failure.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum EnrollmentEngineServiceError {
    /// Every submitter disappeared before orderly shutdown.
    #[error("enrollment engine request channel closed")]
    RequestChannelClosed,
    /// The exact operation coordinator disappeared before terminal delivery.
    #[error("enrollment engine result receiver closed")]
    ResultReceiverClosed,
}

#[cfg(test)]
mod tests {
    use std::{sync::Mutex, thread};

    use faceauth_authz::{AuthorizationGrant, ExecutableFingerprint};
    use faceauth_core::CapturePair;
    use faceauth_inference::{FaceEmbedding, PassiveLivenessScore};
    use faceauth_management::{ManagementConfig, ManagementCoordinator};
    use faceauth_management_dbus::{AuthorizationBackend, BackendFuture, Manager1, MonotonicClock};
    use faceauth_model::ModelRole;
    use faceauth_protocol::{AuthenticationPurpose, ServiceName, TransactionId};
    use faceauth_transport::PeerIdentity;

    use super::*;

    fn authorization() -> Result<AuthorizedEnrollment, Box<dyn std::error::Error>> {
        let grant = AuthorizationGrant {
            transaction_id: TransactionId::generate(),
            peer: PeerIdentity { pid: 1, uid: 0, gid: 0 },
            target_uid: 1000,
            service: ServiceName::parse("faceauth-enroll")?,
            purpose: AuthenticationPurpose::Polkit,
            executable: ExecutableFingerprint { device: 1, inode: 1 },
        };
        Ok(AuthorizedEnrollment::from_grant(&grant)?)
    }

    fn operation() -> Result<OperationId, faceauth_management::ManagementError> {
        OperationId::parse("123e4567-e89b-12d3-a456-426614174000")
    }

    fn config() -> EnrollmentConfig {
        EnrollmentConfig {
            duration_micros: 30_000_000,
            minimum_samples: 3,
            maximum_samples: 5,
            minimum_quality: 0.7,
            minimum_sample_similarity: 0.9,
            minimum_sample_interval_micros: 250_000,
            minimum_yaw_span_degrees: 15.0,
        }
    }

    fn record(uid: u32) -> TemplateRecord {
        let mut embedding = vec![0.0; 32];
        embedding[0] = 1.0;
        TemplateRecord {
            schema_version: faceauth_storage::TEMPLATE_RECORD_SCHEMA_VERSION,
            uid,
            model_compatibility_sha256: "aa".repeat(32),
            embedding,
        }
    }

    fn passive_pad_policy() -> PassivePadPolicy {
        PassivePadPolicy {
            infrared_compatibility_sha256: "11".repeat(32),
            minimum_infrared_probability: 0.8,
            visible_compatibility_sha256: "22".repeat(32),
            minimum_visible_probability: 0.8,
            fusion: Some(("33".repeat(32), 0.8)),
        }
    }

    fn derived_observation(
        timestamp_micros: u64,
        yaw_degrees: f32,
        quality: f32,
        fusion_probability: f32,
    ) -> Result<DerivedBiometricObservation, Box<dyn std::error::Error>> {
        let mut values = vec![0.0_f32; 32];
        values[0] = 1.0;
        Ok(DerivedBiometricObservation {
            timing: CapturePair {
                infrared_timestamp_micros: timestamp_micros - 1,
                visible_timestamp_micros: Some(timestamp_micros),
            },
            quality,
            yaw_degrees,
            embedding: FaceEmbedding::from_normalized_template(&"aa".repeat(32), &values)?,
            passive_liveness: vec![
                PassiveLivenessScore::from_validated_output(
                    ModelRole::PassiveLivenessInfrared,
                    &"11".repeat(32),
                    0.95,
                )?,
                PassiveLivenessScore::from_validated_output(
                    ModelRole::PassiveLivenessVisible,
                    &"22".repeat(32),
                    0.95,
                )?,
                PassiveLivenessScore::from_validated_output(
                    ModelRole::PassiveLivenessFusion,
                    &"33".repeat(32),
                    fusion_probability,
                )?,
            ],
            completed_at_micros: timestamp_micros,
        })
    }

    #[test]
    fn production_accumulator_skips_low_quality_and_too_close_frames_then_finishes()
    -> Result<(), Box<dyn std::error::Error>> {
        let policy = passive_pad_policy();
        let session = EnrollmentSession::start(config(), 1000, 1_000_000)?;
        let mut accumulator =
            EnrollmentAccumulator::new(session, &policy, config().minimum_sample_interval_micros);
        assert!(
            accumulator.observe(&derived_observation(1_500_000, -10.0, 0.2, 0.95)?)?.is_continue()
        );
        assert!(
            accumulator.observe(&derived_observation(2_000_000, -10.0, 0.95, 0.95)?)?.is_continue()
        );
        assert!(
            accumulator.observe(&derived_observation(2_100_000, 10.0, 0.95, 0.95)?)?.is_continue()
        );
        assert!(
            accumulator.observe(&derived_observation(3_000_000, 0.0, 0.95, 0.95)?)?.is_continue()
        );
        let ControlFlow::Break(record) =
            accumulator.observe(&derived_observation(4_000_000, 10.0, 0.95, 0.95)?)?
        else {
            return Err("ready enrollment did not finish".into());
        };
        assert_eq!(record.uid, 1000);
        record.validate()?;
        Ok(())
    }

    #[test]
    fn production_accumulator_fails_closed_when_fusion_pad_rejects()
    -> Result<(), Box<dyn std::error::Error>> {
        let policy = passive_pad_policy();
        let session = EnrollmentSession::start(config(), 1000, 1_000_000)?;
        let mut accumulator =
            EnrollmentAccumulator::new(session, &policy, config().minimum_sample_interval_micros);
        assert!(matches!(
            accumulator.observe(&derived_observation(2_000_000, 0.0, 0.95, 0.1)?),
            Err(ObservationFailure::InvalidEvidence)
        ));
        Ok(())
    }

    #[derive(Clone, Default)]
    struct RecordingSink {
        saved: Arc<Mutex<Vec<u32>>>,
        fail: bool,
    }

    impl EnrollmentTemplateSink for RecordingSink {
        fn save_template(&self, record: &TemplateRecord) -> Result<(), StorageError> {
            if self.fail {
                return Err(StorageError::InvalidRecord("injected storage failure"));
            }
            self.saved
                .lock()
                .map_err(|_| StorageError::InvalidRecord("test sink poisoned"))?
                .push(record.uid);
            Ok(())
        }
    }

    struct SuccessfulEngine;

    impl EnrollmentEngine for SuccessfulEngine {
        fn enroll(
            &mut self,
            job: &EnrollmentJob,
            progress: &mut dyn FnMut(ManagementProgress) -> Result<(), EnrollmentEngineFailure>,
        ) -> Result<TemplateRecord, EnrollmentEngineFailure> {
            progress(ManagementProgress::PositionFace)?;
            progress(ManagementProgress::Processing)?;
            Ok(record(job.target_uid()))
        }
    }

    struct UnusedBackend;

    impl AuthorizationBackend for UnusedBackend {
        fn caller_uid<'a>(&'a self, _sender: &'a str) -> BackendFuture<'a, u32> {
            Box::pin(async { Err(BackendError::new("unused")) })
        }

        fn enrollment_state<'a>(
            &'a self,
            _sender: &'a str,
            _target_uid: u32,
        ) -> BackendFuture<'a, bool> {
            Box::pin(async { Err(BackendError::new("unused")) })
        }

        fn authorize_enrollment<'a>(
            &'a self,
            _sender: &'a str,
            _target_uid: u32,
        ) -> BackendFuture<'a, AuthorizedEnrollment> {
            Box::pin(async { Err(BackendError::new("unused")) })
        }
    }

    struct FixedClock;

    impl MonotonicClock for FixedClock {
        fn now_micros(&self) -> u64 {
            2_000_000
        }
    }

    fn job(
        cancellation: EnrollmentCancellation,
    ) -> Result<EnrollmentJob, Box<dyn std::error::Error>> {
        Ok(EnrollmentJob::new(operation()?, authorization()?, config(), 1_000_000, cancellation))
    }

    #[test]
    fn template_is_committed_before_success_crosses_the_worker_channel()
    -> Result<(), Box<dyn std::error::Error>> {
        let sink = RecordingSink::default();
        let saved = Arc::clone(&sink.saved);
        let (client, service) = EnrollmentEngineService::new(
            SuccessfulEngine,
            sink,
            BiometricResourceArbiter::default(),
        );
        let shutdown = ShutdownToken::default();
        let worker_shutdown = shutdown.clone();
        let worker = thread::spawn(move || service.run(&worker_shutdown));
        let handle = client.submit(job(EnrollmentCancellation::default())?)?;
        assert!(matches!(handle.recv()?, EnrollmentEngineUpdate::Progress { .. }));
        assert!(matches!(handle.recv()?, EnrollmentEngineUpdate::Progress { .. }));
        assert!(matches!(handle.recv()?, EnrollmentEngineUpdate::Completed { result: Ok(()), .. }));
        assert_eq!(*saved.lock().map_err(|_| "test sink poisoned")?, vec![1000]);
        assert!(shutdown.request());
        worker.join().map_err(|_| "worker panicked")??;
        Ok(())
    }

    #[test]
    fn cancellation_and_storage_failure_never_report_success()
    -> Result<(), Box<dyn std::error::Error>> {
        let cancellation = EnrollmentCancellation::default();
        cancellation.cancel();
        let sink = RecordingSink::default();
        let saved = Arc::clone(&sink.saved);
        let (client, service) = EnrollmentEngineService::new(
            SuccessfulEngine,
            sink,
            BiometricResourceArbiter::default(),
        );
        let shutdown = ShutdownToken::default();
        let worker_shutdown = shutdown.clone();
        let worker = thread::spawn(move || service.run(&worker_shutdown));
        let handle = client.submit(job(cancellation)?)?;
        assert!(matches!(
            handle.recv()?,
            EnrollmentEngineUpdate::Completed {
                result: Err(EnrollmentEngineFailure::Cancelled),
                ..
            }
        ));
        assert!(saved.lock().map_err(|_| "test sink poisoned")?.is_empty());
        assert!(shutdown.request());
        worker.join().map_err(|_| "worker panicked")??;

        let sink = RecordingSink { fail: true, ..RecordingSink::default() };
        let (client, service) = EnrollmentEngineService::new(
            SuccessfulEngine,
            sink,
            BiometricResourceArbiter::default(),
        );
        let shutdown = ShutdownToken::default();
        let worker_shutdown = shutdown.clone();
        let worker = thread::spawn(move || service.run(&worker_shutdown));
        let handle = client.submit(job(EnrollmentCancellation::default())?)?;
        let _ = handle.recv()?;
        let _ = handle.recv()?;
        assert!(matches!(
            handle.recv()?,
            EnrollmentEngineUpdate::Completed { result: Err(EnrollmentEngineFailure::Storage), .. }
        ));
        assert!(shutdown.request());
        worker.join().map_err(|_| "worker panicked")??;
        Ok(())
    }

    #[test]
    fn authentication_lease_blocks_enrollment_submission() -> Result<(), Box<dyn std::error::Error>>
    {
        let resources = BiometricResourceArbiter::default();
        let authentication = resources
            .try_acquire(BiometricResourceOwner::Authentication(TransactionId::generate()))?;
        let (client, _service) =
            EnrollmentEngineService::new(SuccessfulEngine, RecordingSink::default(), resources);
        assert!(matches!(
            client.submit(job(EnrollmentCancellation::default())?),
            Err(EnrollmentEngineSubmitError::Busy)
        ));
        drop(authentication);
        Ok(())
    }

    #[test]
    fn manager_bridge_preserves_private_cancellation_until_terminal_relay()
    -> Result<(), Box<dyn std::error::Error>> {
        let (client, service) = EnrollmentEngineService::new(
            SuccessfulEngine,
            RecordingSink::default(),
            BiometricResourceArbiter::default(),
        );
        let bridge = EnrollmentControllerBridge::new(client, config());
        let operation = operation()?;
        EnrollmentOperationController::start(&bridge, authorization()?, operation, 1_000_000)?;
        let handle = bridge.take_handle(operation)?;
        EnrollmentOperationController::cancel(&bridge, operation)?;

        let shutdown = ShutdownToken::default();
        let worker_shutdown = shutdown.clone();
        let worker = thread::spawn(move || service.run(&worker_shutdown));
        assert!(matches!(
            handle.recv()?,
            EnrollmentEngineUpdate::Completed {
                result: Err(EnrollmentEngineFailure::Cancelled),
                ..
            }
        ));
        bridge.finish(operation)?;
        assert!(shutdown.request());
        worker.join().map_err(|_| "worker panicked")??;
        Ok(())
    }

    #[test]
    fn relay_revalidates_and_emits_closed_management_updates()
    -> Result<(), Box<dyn std::error::Error>> {
        let authorization = authorization()?;
        let mut coordinator = ManagementCoordinator::new(ManagementConfig::default())?;
        let started = coordinator.start(authorization, 1_000_000)?;
        let ManagementUpdate::Progress { operation_id, .. } = started else {
            return Err("unexpected terminal start".into());
        };
        let manager = Manager1::new(coordinator, UnusedBackend, FixedClock);
        let management = manager.worker_handle();

        let (client, service) = EnrollmentEngineService::new(
            SuccessfulEngine,
            RecordingSink::default(),
            BiometricResourceArbiter::default(),
        );
        let bridge = EnrollmentControllerBridge::new(client, config());
        EnrollmentOperationController::start(&bridge, authorization, operation_id, 1_000_000)?;
        let shutdown = ShutdownToken::default();
        let worker_shutdown = shutdown.clone();
        let worker = thread::spawn(move || service.run(&worker_shutdown));
        let mut emitted = Vec::new();
        bridge.relay(1000, operation_id, &management, |update| {
            emitted.push(update);
            Ok(())
        })?;
        assert_eq!(emitted.len(), 3);
        assert!(matches!(
            emitted.last(),
            Some(ManagementUpdate::Completed { result: ManagementResult::Completed, .. })
        ));
        assert!(shutdown.request());
        worker.join().map_err(|_| "worker panicked")??;
        Ok(())
    }
}
