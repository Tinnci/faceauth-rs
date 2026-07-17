//! Single-capacity authentication engine worker and derived-evidence boundary.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::Duration,
};

use faceauth_capture::{CaptureError, FrameSource, PairedFrames, PairingPolicy};
use faceauth_core::CapturePair;
use faceauth_inference::{
    FaceEmbedding, ImageView, InferenceError, InputTensor, OnnxSession, PassiveLivenessScore,
};
use faceauth_liveness::{
    ChallengeError, ChallengeObservation, ChallengeProgress, ChallengeSession,
};
use faceauth_protocol::{ProgressCode, TransactionId};
use faceauth_quality::{
    FaceGeometry, ImageView as QualityImageView, QualityConfig, QualityError, QualityReport,
    assess_cancellable,
};
use faceauth_session::{ConnectionToken, SessionError, SessionManager};
use thiserror::Error;

use crate::{AuthenticationObservation, AuthenticationWorker, ShutdownToken};

const REQUEST_QUEUE_CAPACITY: usize = 1;
const MAX_PROGRESS_UPDATES: usize = 32;
const UPDATE_QUEUE_CAPACITY: usize = MAX_PROGRESS_UPDATES + 1;
const WORKER_POLL_INTERVAL: Duration = Duration::from_millis(25);

/// One authorized, transaction-bound unit of biometric work.
///
/// Construction requires the exact active session connection and transaction. The job contains no
/// caller-supplied authorization claim and exposes cancellation-aware capture/inference helpers.
pub struct AuthenticationJob {
    transaction_id: TransactionId,
    target_uid: u32,
    worker: AuthenticationWorker,
    shutdown: ShutdownToken,
}

impl AuthenticationJob {
    /// Bind a job to the exact active authorized daemon session.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError`] when the connection or transaction does not own the active slot.
    pub fn from_session(
        sessions: &SessionManager,
        connection: ConnectionToken,
        transaction_id: TransactionId,
    ) -> Result<Self, SessionError> {
        Ok(Self {
            transaction_id,
            target_uid: sessions.target_uid(connection, transaction_id)?,
            worker: AuthenticationWorker::new(
                sessions.cancellation_token(connection, transaction_id)?,
            ),
            shutdown: ShutdownToken::default(),
        })
    }

    /// Exact authorized transaction identifier.
    #[must_use]
    pub const fn transaction_id(&self) -> TransactionId {
        self.transaction_id
    }

    /// Numeric target UID copied from the authorization grant.
    #[must_use]
    pub const fn target_uid(&self) -> u32 {
        self.target_uid
    }

    /// Return whether the session ended or the daemon is shutting down.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.worker.is_cancelled() || self.shutdown.is_requested()
    }

    /// Capture one bounded IR/RGB pair with session and daemon cancellation.
    ///
    /// # Errors
    ///
    /// Returns [`CaptureError`] for cancellation, device, timing, or pairing failure.
    pub fn capture_pair(
        &self,
        infrared: &mut impl FrameSource,
        visible: &mut impl FrameSource,
        policy: PairingPolicy,
    ) -> Result<PairedFrames, CaptureError> {
        faceauth_capture::capture_pair_cancellable(infrared, visible, policy, || {
            self.is_cancelled()
        })
    }

    /// Preprocess an ephemeral image with session and daemon cancellation.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError`] for cancellation or malformed/bounded preprocessing failure.
    pub fn preprocess(
        &self,
        session: &OnnxSession,
        image: ImageView<'_>,
    ) -> Result<InputTensor, InferenceError> {
        session.preprocess_cancellable(image, || self.is_cancelled())
    }

    /// Assess one borrowed face region without retaining image pixels, while mapping exact job
    /// cancellation into both bounded image scans.
    ///
    /// # Errors
    ///
    /// Returns [`QualityError`] for cancellation, malformed geometry/image data, invalid
    /// calibration, resource-limit violations, or non-finite computation.
    pub fn assess_quality(
        &self,
        image: QualityImageView<'_>,
        geometry: FaceGeometry,
        config: QualityConfig,
    ) -> Result<QualityReport, QualityError> {
        assess_cancellable(image, geometry, config, || self.is_cancelled())
    }

    /// Process one active-liveness observation with session and daemon cancellation.
    ///
    /// # Errors
    ///
    /// Returns [`ChallengeError`] for cancellation or invalid temporal evidence.
    pub fn observe_liveness(
        &self,
        challenge: &mut ChallengeSession,
        observation: ChallengeObservation,
    ) -> Result<ChallengeProgress, ChallengeError> {
        challenge.observe_cancellable(observation, || self.is_cancelled())
    }
}

/// Complete owned evidence returned by the biometric engine.
///
/// Only derived values cross the worker channel. No raw image, camera buffer, landmark vector,
/// template, key, or model tensor is represented by this type.
pub struct DerivedAuthenticationEvidence {
    transaction_id: TransactionId,
    timing: CapturePair,
    quality: f32,
    embedding: FaceEmbedding,
    passive_liveness: Vec<PassiveLivenessScore>,
    active_challenge: ChallengeProgress,
    completed_at_micros: u64,
}

impl DerivedAuthenticationEvidence {
    /// Assemble owned derived evidence for final template comparison and policy evaluation.
    #[must_use]
    pub const fn new(
        transaction_id: TransactionId,
        timing: CapturePair,
        quality: f32,
        embedding: FaceEmbedding,
        passive_liveness: Vec<PassiveLivenessScore>,
        active_challenge: ChallengeProgress,
        completed_at_micros: u64,
    ) -> Self {
        Self {
            transaction_id,
            timing,
            quality,
            embedding,
            passive_liveness,
            active_challenge,
            completed_at_micros,
        }
    }

    /// Exact authorized job that produced this evidence.
    #[must_use]
    pub const fn transaction_id(&self) -> TransactionId {
        self.transaction_id
    }

    /// Borrow the derived evidence in the existing final policy boundary shape.
    #[must_use]
    pub fn observation(&self) -> AuthenticationObservation<'_> {
        AuthenticationObservation {
            timing: self.timing,
            quality: self.quality,
            embedding: &self.embedding,
            passive_liveness: &self.passive_liveness,
            active_challenge: self.active_challenge,
        }
    }

    /// Monotonic completion time used for the session deadline check.
    #[must_use]
    pub const fn completed_at_micros(&self) -> u64 {
        self.completed_at_micros
    }
}

/// Sanitized failure category returned by the biometric engine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuthenticationEngineFailure {
    /// Session cancellation or daemon shutdown was observed.
    Cancelled,
    /// Required cameras or paired capture were unavailable.
    Capture,
    /// Model loading, preprocessing, or bounded inference failed.
    Inference,
    /// Passive or active liveness processing failed structurally.
    Liveness,
    /// Derived evidence was malformed or incomplete.
    InvalidEvidence,
    /// Internal coordination failed without exposing sensitive detail.
    Internal,
}

/// Long-lived engine implementation owning cameras and mutable inference sessions.
pub trait AuthenticationEngine: Send + 'static {
    /// Execute one already-authorized job and return only owned derived evidence.
    ///
    /// Implementations must use the job's cancellation-aware methods at bounded stage boundaries.
    /// Progress callbacks are bounded and may fail if the owning connection disappeared.
    ///
    /// # Errors
    ///
    /// Returns a sanitized [`AuthenticationEngineFailure`] for capture, inference, liveness,
    /// cancellation, invalid evidence, or internal coordination failure.
    fn authenticate(
        &mut self,
        job: &AuthenticationJob,
        progress: &mut dyn FnMut(ProgressCode) -> Result<(), AuthenticationEngineFailure>,
    ) -> Result<DerivedAuthenticationEvidence, AuthenticationEngineFailure>;
}

/// One bounded update from the engine worker.
pub enum AuthenticationEngineUpdate {
    /// Non-biometric, transaction-bound user guidance.
    Progress {
        /// Exact job transaction.
        transaction_id: TransactionId,
        /// Closed public progress code.
        progress: ProgressCode,
    },
    /// Terminal worker outcome. Final authentication is still decided by template/policy code.
    Completed {
        /// Exact job transaction.
        transaction_id: TransactionId,
        /// Derived evidence or a sanitized internal failure.
        result: Result<DerivedAuthenticationEvidence, AuthenticationEngineFailure>,
    },
}

struct EngineRequest {
    job: AuthenticationJob,
    updates: mpsc::SyncSender<AuthenticationEngineUpdate>,
}

/// Cloneable submission half for the single-capacity engine worker.
#[derive(Clone)]
pub struct AuthenticationEngineClient {
    requests: mpsc::SyncSender<EngineRequest>,
    active: Arc<AtomicBool>,
}

impl AuthenticationEngineClient {
    /// Submit one authorized job without waiting for queue capacity.
    ///
    /// # Errors
    ///
    /// Returns [`AuthenticationEngineSubmitError::Busy`] while another job owns execution
    /// capacity, or [`AuthenticationEngineSubmitError::WorkerStopped`] if supervision ended.
    pub fn submit(
        &self,
        job: AuthenticationJob,
    ) -> Result<AuthenticationEngineHandle, AuthenticationEngineSubmitError> {
        if self.active.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
            return Err(AuthenticationEngineSubmitError::Busy);
        }
        let transaction_id = job.transaction_id;
        let (updates, receiver) = mpsc::sync_channel(UPDATE_QUEUE_CAPACITY);
        if self.requests.try_send(EngineRequest { job, updates }).is_err() {
            self.active.store(false, Ordering::Release);
            return Err(AuthenticationEngineSubmitError::WorkerStopped);
        }
        Ok(AuthenticationEngineHandle { transaction_id, receiver })
    }
}

/// Receiving half for one exact engine job.
pub struct AuthenticationEngineHandle {
    transaction_id: TransactionId,
    receiver: mpsc::Receiver<AuthenticationEngineUpdate>,
}

impl AuthenticationEngineHandle {
    /// Exact submitted transaction.
    #[must_use]
    pub const fn transaction_id(&self) -> TransactionId {
        self.transaction_id
    }

    /// Wait for the next bounded engine update.
    ///
    /// # Errors
    ///
    /// Returns [`mpsc::RecvError`] if the supervised engine ended without a terminal update.
    pub fn recv(&self) -> Result<AuthenticationEngineUpdate, mpsc::RecvError> {
        self.receiver.recv()
    }

    /// Wait for the next update until a caller-provided coordination deadline.
    ///
    /// # Errors
    ///
    /// Returns [`mpsc::RecvTimeoutError`] on timeout or worker disconnection.
    pub fn recv_timeout(
        &self,
        timeout: Duration,
    ) -> Result<AuthenticationEngineUpdate, mpsc::RecvTimeoutError> {
        self.receiver.recv_timeout(timeout)
    }
}

/// Worker half that owns the mutable authentication engine.
pub struct AuthenticationEngineService<E> {
    engine: E,
    requests: mpsc::Receiver<EngineRequest>,
    active: Arc<AtomicBool>,
}

impl<E> AuthenticationEngineService<E>
where
    E: AuthenticationEngine,
{
    /// Create a single-capacity engine client and its supervised worker half.
    #[must_use]
    pub fn new(engine: E) -> (AuthenticationEngineClient, Self) {
        let (requests, receiver) = mpsc::sync_channel(REQUEST_QUEUE_CAPACITY);
        let active = Arc::new(AtomicBool::new(false));
        (
            AuthenticationEngineClient { requests, active: Arc::clone(&active) },
            Self { engine, requests: receiver, active },
        )
    }

    /// Run jobs until daemon shutdown, treating client loss as service-fatal.
    ///
    /// # Errors
    ///
    /// Returns [`AuthenticationEngineServiceError`] if the request channel or an exact job result
    /// receiver disappears before orderly daemon shutdown.
    pub fn run(mut self, shutdown: &ShutdownToken) -> Result<(), AuthenticationEngineServiceError> {
        loop {
            if shutdown.is_requested() {
                return Ok(());
            }
            let request = match self.requests.recv_timeout(WORKER_POLL_INTERVAL) {
                Ok(request) => request,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(AuthenticationEngineServiceError::RequestChannelClosed);
                }
            };
            self.run_job(request, shutdown)?;
        }
    }

    fn run_job(
        &mut self,
        mut request: EngineRequest,
        shutdown: &ShutdownToken,
    ) -> Result<(), AuthenticationEngineServiceError> {
        request.job.shutdown = shutdown.clone();
        let transaction_id = request.job.transaction_id;
        let mut progress_count = 0_usize;
        let mut progress = |progress| {
            if request.job.is_cancelled() {
                return Err(AuthenticationEngineFailure::Cancelled);
            }
            if progress_count >= MAX_PROGRESS_UPDATES {
                return Err(AuthenticationEngineFailure::Internal);
            }
            request
                .updates
                .try_send(AuthenticationEngineUpdate::Progress { transaction_id, progress })
                .map_err(|_| AuthenticationEngineFailure::Internal)?;
            progress_count += 1;
            Ok(())
        };
        let result = if request.job.is_cancelled() {
            Err(AuthenticationEngineFailure::Cancelled)
        } else {
            self.engine.authenticate(&request.job, &mut progress)
        };
        let result = if request.job.is_cancelled() {
            Err(AuthenticationEngineFailure::Cancelled)
        } else {
            result
        };
        let result = match result {
            Ok(evidence) if evidence.transaction_id == transaction_id => Ok(evidence),
            Ok(_) => Err(AuthenticationEngineFailure::InvalidEvidence),
            Err(error) => Err(error),
        };
        let send_result =
            request.updates.send(AuthenticationEngineUpdate::Completed { transaction_id, result });
        self.active.store(false, Ordering::Release);
        send_result.map_err(|_| AuthenticationEngineServiceError::ResultReceiverClosed)
    }
}

/// Non-blocking engine submission failure.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum AuthenticationEngineSubmitError {
    /// Another transaction owns the single camera/model capacity.
    #[error("authentication engine is busy")]
    Busy,
    /// The supervised engine service is no longer available.
    #[error("authentication engine worker has stopped")]
    WorkerStopped,
}

/// Fatal worker lifecycle failure.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum AuthenticationEngineServiceError {
    /// Every submission handle disappeared outside orderly daemon shutdown.
    #[error("authentication engine request channel closed")]
    RequestChannelClosed,
    /// The exact job coordinator disappeared before receiving its terminal result.
    #[error("authentication engine result receiver closed")]
    ResultReceiverClosed,
}

#[cfg(test)]
mod tests {
    use std::{thread, time::Instant};

    use faceauth_authz::{AuthorizationGrant, ExecutableFingerprint};
    use faceauth_core::CapturePair;
    use faceauth_inference::{FaceEmbedding, PassiveLivenessScore};
    use faceauth_model::ModelRole;
    use faceauth_protocol::{AuthenticationPurpose, ServiceName};
    use faceauth_session::SessionConfig;
    use faceauth_transport::PeerIdentity;

    use crate::{ServiceSupervisor, SupervisorConfig};

    use super::*;

    fn active_job()
    -> Result<(SessionManager, ConnectionToken, AuthenticationJob), Box<dyn std::error::Error>>
    {
        let mut sessions = SessionManager::new(SessionConfig::default())?;
        let connection = ConnectionToken::generate()?;
        let transaction_id = TransactionId::generate();
        let grant = AuthorizationGrant {
            transaction_id,
            peer: PeerIdentity { pid: 1, uid: 0, gid: 0 },
            target_uid: 1000,
            service: ServiceName::parse("faceauth-test")?,
            purpose: AuthenticationPurpose::Test,
            executable: ExecutableFingerprint { device: 1, inode: 1 },
        };
        let _started = sessions.start(grant, connection, 1_000_000)?;
        let job = AuthenticationJob::from_session(&sessions, connection, transaction_id)?;
        Ok((sessions, connection, job))
    }

    fn evidence(
        transaction_id: TransactionId,
    ) -> Result<DerivedAuthenticationEvidence, Box<dyn std::error::Error>> {
        let mut values = vec![0.0; 32];
        values[0] = 1.0;
        let embedding = FaceEmbedding::from_normalized_template(&"aa".repeat(32), &values)?;
        let passive_liveness = vec![
            PassiveLivenessScore::from_validated_output(
                ModelRole::PassiveLivenessInfrared,
                &"11".repeat(32),
                0.9,
            )?,
            PassiveLivenessScore::from_validated_output(
                ModelRole::PassiveLivenessVisible,
                &"22".repeat(32),
                0.9,
            )?,
        ];
        Ok(DerivedAuthenticationEvidence::new(
            transaction_id,
            CapturePair {
                infrared_timestamp_micros: 2_000_000,
                visible_timestamp_micros: Some(2_010_000),
            },
            0.9,
            embedding,
            passive_liveness,
            ChallengeProgress::Passed,
            2_100_000,
        ))
    }

    struct SuccessfulEngine;

    impl AuthenticationEngine for SuccessfulEngine {
        fn authenticate(
            &mut self,
            job: &AuthenticationJob,
            progress: &mut dyn FnMut(ProgressCode) -> Result<(), AuthenticationEngineFailure>,
        ) -> Result<DerivedAuthenticationEvidence, AuthenticationEngineFailure> {
            assert_eq!(job.target_uid(), 1000);
            progress(ProgressCode::PositionFace)?;
            progress(ProgressCode::Processing)?;
            evidence(job.transaction_id()).map_err(|_| AuthenticationEngineFailure::Internal)
        }
    }

    fn supervisor() -> Result<ServiceSupervisor, crate::SupervisorError> {
        ServiceSupervisor::new(SupervisorConfig {
            poll_interval: Duration::from_millis(1),
            shutdown_grace: Duration::from_secs(1),
            max_services: 2,
        })
    }

    #[test]
    fn worker_returns_only_transaction_bound_progress_and_derived_evidence()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_sessions, _connection, job) = active_job()?;
        let transaction_id = job.transaction_id();
        let (client, service) = AuthenticationEngineService::new(SuccessfulEngine);
        let mut supervisor = supervisor()?;
        let shutdown = supervisor.shutdown_token();
        supervisor.spawn("engine", move |shutdown| {
            service.run(&shutdown).map_err(|error| error.to_string())
        })?;
        let handle = client.submit(job)?;
        assert_eq!(handle.transaction_id(), transaction_id);
        for expected in [ProgressCode::PositionFace, ProgressCode::Processing] {
            assert!(matches!(
                handle.recv_timeout(Duration::from_secs(1))?,
                AuthenticationEngineUpdate::Progress {
                    transaction_id: actual,
                    progress,
                } if actual == transaction_id && progress == expected
            ));
        }
        let completed = handle.recv_timeout(Duration::from_secs(1))?;
        let AuthenticationEngineUpdate::Completed { transaction_id: actual, result } = completed
        else {
            return Err("terminal engine update expected".into());
        };
        assert_eq!(actual, transaction_id);
        let result = result.map_err(|_| "engine returned failure")?;
        assert_eq!(result.completed_at_micros(), 2_100_000);
        assert!((result.observation().quality - 0.9).abs() < f32::EPSILON);
        let _first_request = shutdown.request();
        supervisor.run_until(|| false)?;
        Ok(())
    }

    struct CancellationEngine;

    impl AuthenticationEngine for CancellationEngine {
        fn authenticate(
            &mut self,
            job: &AuthenticationJob,
            _progress: &mut dyn FnMut(ProgressCode) -> Result<(), AuthenticationEngineFailure>,
        ) -> Result<DerivedAuthenticationEvidence, AuthenticationEngineFailure> {
            let deadline = Instant::now() + Duration::from_secs(1);
            while !job.is_cancelled() && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(1));
            }
            Err(AuthenticationEngineFailure::Cancelled)
        }
    }

    #[test]
    fn session_cancellation_reaches_the_running_engine() -> Result<(), Box<dyn std::error::Error>> {
        let (mut sessions, connection, job) = active_job()?;
        let transaction_id = job.transaction_id();

        let (client, service) = AuthenticationEngineService::new(CancellationEngine);
        let mut supervisor = supervisor()?;
        let shutdown = supervisor.shutdown_token();
        supervisor.spawn("engine", move |shutdown| {
            service.run(&shutdown).map_err(|error| error.to_string())
        })?;
        let handle = client.submit(job)?;
        let _cancelled = sessions.cancel(connection, transaction_id, 1_100_000)?;
        assert!(matches!(
            handle.recv_timeout(Duration::from_secs(1))?,
            AuthenticationEngineUpdate::Completed {
                result: Err(AuthenticationEngineFailure::Cancelled),
                ..
            }
        ));
        let _first_request = shutdown.request();
        supervisor.run_until(|| false)?;
        Ok(())
    }

    struct BlockingEngine {
        started: mpsc::Sender<()>,
        release: mpsc::Receiver<()>,
    }

    impl AuthenticationEngine for BlockingEngine {
        fn authenticate(
            &mut self,
            job: &AuthenticationJob,
            _progress: &mut dyn FnMut(ProgressCode) -> Result<(), AuthenticationEngineFailure>,
        ) -> Result<DerivedAuthenticationEvidence, AuthenticationEngineFailure> {
            self.started.send(()).map_err(|_| AuthenticationEngineFailure::Internal)?;
            self.release.recv().map_err(|_| AuthenticationEngineFailure::Internal)?;
            evidence(job.transaction_id()).map_err(|_| AuthenticationEngineFailure::Internal)
        }
    }

    #[test]
    fn engine_capacity_cannot_queue_a_second_transaction() -> Result<(), Box<dyn std::error::Error>>
    {
        let (_first_sessions, _first_connection, first_job) = active_job()?;
        let (_second_sessions, _second_connection, second_job) = active_job()?;
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (client, service) = AuthenticationEngineService::new(BlockingEngine {
            started: started_tx,
            release: release_rx,
        });
        let mut supervisor = supervisor()?;
        let shutdown = supervisor.shutdown_token();
        supervisor.spawn("engine", move |shutdown| {
            service.run(&shutdown).map_err(|error| error.to_string())
        })?;
        let first = client.submit(first_job)?;
        started_rx.recv_timeout(Duration::from_secs(1))?;
        assert!(matches!(client.submit(second_job), Err(AuthenticationEngineSubmitError::Busy)));
        release_tx.send(())?;
        assert!(matches!(
            first.recv_timeout(Duration::from_secs(1))?,
            AuthenticationEngineUpdate::Completed { result: Ok(_), .. }
        ));
        let _first_request = shutdown.request();
        supervisor.run_until(|| false)?;
        Ok(())
    }
}
