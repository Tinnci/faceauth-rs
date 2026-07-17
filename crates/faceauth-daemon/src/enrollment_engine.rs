//! Single-capacity enrollment worker with shared resource ownership and atomic template commit.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::Duration,
};

use faceauth_enrollment::{EnrollmentConfig, EnrollmentError, EnrollmentSession};
use faceauth_management::{AuthorizedEnrollment, ManagementProgress, OperationId};
use faceauth_storage::{EncryptedTemplateStore, KeyProvider, StorageError, TemplateRecord};
use thiserror::Error;

use crate::{
    BiometricResourceArbiter, BiometricResourceError, BiometricResourceLease,
    BiometricResourceOwner, ShutdownToken,
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
}
