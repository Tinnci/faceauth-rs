//! Privileged daemon authentication boundary orchestration.

mod engine;
mod enrollment_engine;
mod production;
mod resource;
mod supervision;

pub use engine::{
    AuthenticationEngine, AuthenticationEngineClient, AuthenticationEngineFailure,
    AuthenticationEngineHandle, AuthenticationEngineService, AuthenticationEngineServiceError,
    AuthenticationEngineSubmitError, AuthenticationEngineUpdate, AuthenticationJob,
    DerivedAuthenticationEvidence, DetectionPolicy, ProductionAuthenticationEngine,
    ProductionAuthenticationEngineError,
};
pub use enrollment_engine::{
    EnrollmentCancellation, EnrollmentControllerBridge, EnrollmentEngine, EnrollmentEngineClient,
    EnrollmentEngineFailure, EnrollmentEngineHandle, EnrollmentEngineService,
    EnrollmentEngineServiceError, EnrollmentEngineSubmitError, EnrollmentEngineUpdate,
    EnrollmentJob, EnrollmentTemplateSink,
};
pub use production::{
    PRODUCTION_CONFIG_SCHEMA_VERSION, ProductionAuthenticationBoundary, ProductionConfig,
    ProductionConfigError, READINESS_REPORT_SCHEMA_VERSION, ReadinessGate, ReadinessGateKind,
    ReadinessReport, build_production_authentication_boundary,
    build_production_authentication_engine, build_production_template_store,
    inspect_production_readiness, load_production_config, readiness_from_config_path,
};
pub use resource::{
    BiometricResourceArbiter, BiometricResourceError, BiometricResourceLease,
    BiometricResourceOwner,
};
pub use supervision::{
    ServiceSupervisor, ShutdownFuture, ShutdownToken, SupervisorConfig, SupervisorError,
    SupervisorReport, run_supervised_authentication_listener,
};

use faceauth_authz::{
    AuthorizationError, AuthorizationGrant, AuthorizationPolicy, BoundAuthorizationIssuer,
    ExecutableError, VerifiedExecutable,
};
use faceauth_capture::{CaptureError, FrameSource, PairedFrames, PairingPolicy};
use faceauth_core::{
    AuthPolicy, AuthenticationDecision, AuthenticationEvidence, CapturePair, CheckStatus,
    DecisionReason, ObservationStatus, PolicyError,
};
use faceauth_enrollment::{EnrollmentConfig, EnrollmentError, EnrollmentSession};
use faceauth_inference::{
    FaceEmbedding, FaceRegion, FacialLandmarks, ImageFacialLandmarks, ImageView, InferenceError,
    InputTensor, OnnxSession, PassiveLivenessScore,
};
use faceauth_liveness::{
    ChallengeError, ChallengeObservation, ChallengeProgress, ChallengeSession,
};
use faceauth_management::AuthorizedEnrollment;
use faceauth_management_dbus::{
    BackendError as ManagementBackendError, BoundedEnrollmentStateSource, TemplateStateReader,
};
use faceauth_model::ModelRole;
use faceauth_protocol::{
    DecisionCode, ProgressCode, RejectionCode, Request, Response, TransactionId,
};
use faceauth_session::{CancellationToken, ConnectionToken, SessionError, SessionManager};
use faceauth_storage::{
    EncryptedTemplateStore, KeyProvider, StorageError, TEMPLATE_RECORD_SCHEMA_VERSION,
    TemplateRecord,
};
use faceauth_transport::{PeerIdentity, PeerStream, TransportError};
use thiserror::Error;

const CONNECTION_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(25);
const MAX_CANCEL_DRAIN_POLLS: usize = 400;

/// Start enrollment only from an exact root broker grant dedicated to Polkit enrollment.
///
/// # Errors
///
/// Returns [`EnrollmentBoundaryError`] when the grant is for another service/purpose, does not
/// originate from a root peer, or the enrollment configuration is invalid.
pub fn begin_authorized_enrollment(
    grant: &AuthorizationGrant,
    config: EnrollmentConfig,
    now_micros: u64,
) -> Result<EnrollmentSession, EnrollmentBoundaryError> {
    let authorization = AuthorizedEnrollment::from_grant(grant)
        .map_err(|_| EnrollmentBoundaryError::UnauthorizedGrant)?;
    Ok(EnrollmentSession::start(config, authorization.target_uid(), now_micros)?)
}

/// Bind the daemon's exact root broker evidence to the Manager1 enrollment grant issuer.
///
/// Construction performs a policy preflight for an arbitrary target and revalidates the resulting
/// grant through [`AuthorizedEnrollment::from_grant`]. Each real issuance later uses a fresh
/// transaction identifier and the requested target UID.
///
/// # Errors
///
/// Returns [`ManagementBackendError`] unless the peer is root and the policy, executable,
/// `faceauth-enroll` service, and Polkit purpose produce an accepted enrollment grant.
pub fn management_enrollment_grant_issuer(
    policy: AuthorizationPolicy,
    peer: PeerIdentity,
    executable: VerifiedExecutable,
) -> Result<BoundAuthorizationIssuer, ManagementBackendError> {
    if peer.uid != 0 {
        return Err(ManagementBackendError::new("enrollment grant issuer peer is not root"));
    }
    let service = faceauth_protocol::ServiceName::parse(faceauth_management::ENROLLMENT_SERVICE)
        .map_err(|error| ManagementBackendError::new(error.to_string()))?;
    let issuer = BoundAuthorizationIssuer::new(
        policy,
        peer,
        executable,
        service,
        faceauth_protocol::AuthenticationPurpose::Polkit,
    );
    let probe =
        issuer.issue(u32::MAX).map_err(|error| ManagementBackendError::new(error.to_string()))?;
    AuthorizedEnrollment::from_grant(&probe)
        .map_err(|error| ManagementBackendError::new(error.to_string()))?;
    Ok(issuer)
}

/// Enrollment authorization or transaction setup failure.
#[derive(Debug, Error)]
pub enum EnrollmentBoundaryError {
    /// Grant was not issued to the dedicated root Polkit enrollment broker.
    #[error("authorization grant is not valid for enrollment")]
    UnauthorizedGrant,
    /// Enrollment transaction configuration or deadline was invalid.
    #[error("unable to start enrollment transaction: {0}")]
    Enrollment(#[from] EnrollmentError),
}

/// Build the only persistable enrollment payload from a normalized derived embedding.
///
/// Raw images are not accepted or represented by this boundary.
#[must_use]
pub fn template_from_embedding(uid: u32, embedding: &FaceEmbedding) -> TemplateRecord {
    TemplateRecord {
        schema_version: TEMPLATE_RECORD_SCHEMA_VERSION,
        uid,
        model_compatibility_sha256: embedding.compatibility_sha256().to_owned(),
        embedding: embedding.values().to_vec(),
    }
}

/// Compare an observed embedding against one authenticated encrypted template record.
///
/// # Errors
///
/// Returns [`BiometricComparisonError`] when the template is malformed, incompatible with the
/// active model contract, or cannot be reconstructed as a normalized embedding.
pub fn compare_template(
    template: &TemplateRecord,
    observed: &FaceEmbedding,
) -> Result<f32, BiometricComparisonError> {
    template.require_compatibility(observed.compatibility_sha256(), observed.values().len())?;
    let enrolled = FaceEmbedding::from_normalized_template(
        &template.model_compatibility_sha256,
        &template.embedding,
    )?;
    Ok(observed.similarity(&enrolled)?)
}

/// Failure while bridging encrypted templates into role-bound inference evidence.
#[derive(Debug, Error)]
pub enum BiometricComparisonError {
    /// Encrypted template validation or compatibility check failed.
    #[error("template comparison storage check failed: {0}")]
    Storage(#[from] StorageError),
    /// Embedding reconstruction or similarity calculation failed.
    #[error("template comparison inference check failed: {0}")]
    Inference(#[from] InferenceError),
}

/// Exact passive presentation-attack model contracts and calibrated thresholds.
#[derive(Clone, Debug, PartialEq)]
pub struct PassivePadPolicy {
    /// Compatibility digest for the reviewed IR PAD model.
    pub infrared_compatibility_sha256: String,
    /// Minimum calibrated IR live probability.
    pub minimum_infrared_probability: f32,
    /// Compatibility digest for the reviewed visible-light PAD model.
    pub visible_compatibility_sha256: String,
    /// Minimum calibrated visible-light live probability.
    pub minimum_visible_probability: f32,
    /// Optional required fusion model compatibility digest and minimum live probability.
    pub fusion: Option<(String, f32)>,
}

impl PassivePadPolicy {
    /// Validate all model identities and calibrated probability thresholds.
    ///
    /// # Errors
    ///
    /// Returns [`AuthenticationPipelineError::InvalidPassivePadPolicy`] for malformed model
    /// digests or non-finite/out-of-range thresholds.
    pub fn validate(&self) -> Result<(), AuthenticationPipelineError> {
        let valid = valid_digest(&self.infrared_compatibility_sha256)
            && valid_probability(self.minimum_infrared_probability)
            && valid_digest(&self.visible_compatibility_sha256)
            && valid_probability(self.minimum_visible_probability)
            && self.fusion.as_ref().is_none_or(|(digest, minimum)| {
                valid_digest(digest) && valid_probability(*minimum)
            });
        if valid { Ok(()) } else { Err(AuthenticationPipelineError::InvalidPassivePadPolicy) }
    }

    fn accepts(
        &self,
        scores: &[PassiveLivenessScore],
    ) -> Result<bool, AuthenticationPipelineError> {
        self.validate()?;
        let infrared = exact_pad_score(
            scores,
            ModelRole::PassiveLivenessInfrared,
            &self.infrared_compatibility_sha256,
        )?;
        let visible = exact_pad_score(
            scores,
            ModelRole::PassiveLivenessVisible,
            &self.visible_compatibility_sha256,
        )?;
        let fusion_passed = if let Some((digest, minimum)) = &self.fusion {
            exact_pad_score(scores, ModelRole::PassiveLivenessFusion, digest)?.passes(*minimum)?
        } else {
            !scores.iter().any(|score| score.role() == ModelRole::PassiveLivenessFusion)
        };
        Ok(infrared.passes(self.minimum_infrared_probability)?
            && visible.passes(self.minimum_visible_probability)?
            && fusion_passed)
    }
}

/// Ephemeral, derived evidence for one authentication policy evaluation.
#[derive(Clone, Copy)]
pub struct AuthenticationObservation<'a> {
    /// Paired IR/visible monotonic capture timestamps.
    pub timing: CapturePair,
    /// Aggregate image/face quality in 0..=1.
    pub quality: f32,
    /// Role-bound normalized embedding from the current observation.
    pub embedding: &'a FaceEmbedding,
    /// Exact PAD outputs required by [`PassivePadPolicy`].
    pub passive_liveness: &'a [PassiveLivenessScore],
    /// Terminal state of the randomized active challenge.
    pub active_challenge: ChallengeProgress,
}

/// Evaluate a complete derived biometric evidence set against one encrypted template.
///
/// Raw images are not represented by this API. Missing, duplicated, unexpected, incompatible, or
/// malformed evidence returns an error; valid negative evidence returns a rejected policy decision.
///
/// # Errors
///
/// Returns [`AuthenticationPipelineError`] when capture timing, template compatibility, PAD model
/// identity, score structure, or policy evaluation is invalid.
pub fn evaluate_authentication(
    policy: AuthPolicy,
    passive_pad_policy: &PassivePadPolicy,
    template: &TemplateRecord,
    observation: AuthenticationObservation<'_>,
) -> Result<AuthenticationDecision, AuthenticationPipelineError> {
    policy.capture.accepts(observation.timing)?;
    let similarity = compare_template(template, observation.embedding)?;
    let passive_liveness = if passive_pad_policy.accepts(observation.passive_liveness)? {
        CheckStatus::Passed
    } else {
        CheckStatus::Failed
    };
    let active_challenge = if observation.active_challenge == ChallengeProgress::Passed {
        CheckStatus::Passed
    } else {
        CheckStatus::Failed
    };
    Ok(policy.evaluate(AuthenticationEvidence {
        similarity,
        quality: observation.quality,
        infrared: ObservationStatus::Observed,
        visible: ObservationStatus::Observed,
        passive_liveness,
        active_challenge,
    })?)
}

fn exact_pad_score<'a>(
    scores: &'a [PassiveLivenessScore],
    role: ModelRole,
    compatibility_sha256: &str,
) -> Result<&'a PassiveLivenessScore, AuthenticationPipelineError> {
    let mut matching = scores.iter().filter(|score| score.role() == role);
    let score = matching.next().ok_or(AuthenticationPipelineError::PassivePadEvidenceInvalid)?;
    if matching.next().is_some() || score.compatibility_sha256() != compatibility_sha256 {
        return Err(AuthenticationPipelineError::PassivePadEvidenceInvalid);
    }
    Ok(score)
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_hexdigit())
        && !value.bytes().any(|byte| byte.is_ascii_uppercase())
}

fn valid_probability(value: f32) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

/// Failure while assembling and evaluating authentication evidence.
#[derive(Debug, Error)]
pub enum AuthenticationPipelineError {
    /// Passive PAD policy contains an invalid digest or threshold.
    #[error("invalid passive PAD policy")]
    InvalidPassivePadPolicy,
    /// Required PAD evidence is missing, duplicated, unexpected, or model-incompatible.
    #[error("passive PAD evidence does not exactly match the configured model contracts")]
    PassivePadEvidenceInvalid,
    /// Capture or final authentication policy rejected malformed evidence.
    #[error("authentication policy evaluation failed: {0}")]
    Policy(#[from] PolicyError),
    /// Template and observed embedding could not be compared.
    #[error("biometric comparison failed: {0}")]
    Comparison(#[from] BiometricComparisonError),
    /// Passive PAD score or calibrated threshold was invalid.
    #[error("passive PAD inference evidence failed: {0}")]
    Inference(#[from] InferenceError),
}

/// Daemon-owned facade that propagates one transaction cancellation signal through every
/// cancellable biometric stage without coupling those crates to session management.
#[derive(Clone, Debug)]
pub struct AuthenticationWorker {
    cancellation: CancellationToken,
}

impl AuthenticationWorker {
    /// Bind a worker to an already authorized transaction cancellation signal.
    #[must_use]
    pub const fn new(cancellation: CancellationToken) -> Self {
        Self { cancellation }
    }

    /// Return whether the owning transaction has been cancelled or terminally consumed.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }

    /// Pair one IR/RGB observation while mapping the transaction signal to capture.
    ///
    /// # Errors
    ///
    /// Returns the bounded capture/pairing error, including [`CaptureError::Cancelled`].
    pub fn capture_pair(
        &self,
        infrared: &mut impl FrameSource,
        visible: &mut impl FrameSource,
        policy: PairingPolicy,
    ) -> Result<PairedFrames, CaptureError> {
        faceauth_capture::capture_pair_cancellable(infrared, visible, policy, || {
            self.cancellation.is_cancelled()
        })
    }

    /// Preprocess one image while mapping the transaction signal to inference.
    ///
    /// # Errors
    ///
    /// Returns the bounded preprocessing error, including [`InferenceError::Cancelled`].
    pub fn preprocess(
        &self,
        session: &OnnxSession,
        image: ImageView<'_>,
    ) -> Result<InputTensor, InferenceError> {
        session.preprocess_cancellable(image, || self.cancellation.is_cancelled())
    }

    /// Align a full-frame face directly into a zeroizing model tensor with transaction
    /// cancellation and the session's manifest-bound geometry contract.
    ///
    /// # Errors
    ///
    /// Returns the bounded alignment/preprocessing error, including [`InferenceError::Cancelled`].
    pub fn preprocess_aligned(
        &self,
        session: &OnnxSession,
        image: ImageView<'_>,
        landmarks: &ImageFacialLandmarks,
    ) -> Result<InputTensor, InferenceError> {
        session.preprocess_aligned(image, landmarks, || self.cancellation.is_cancelled())
    }

    /// Directly sample one exact detector region into a landmark tensor with transaction
    /// cancellation and no intermediate cropped byte image.
    ///
    /// # Errors
    ///
    /// Returns the bounded landmark preprocessing error, including cancellation and invalid crop
    /// geometry.
    pub fn preprocess_face_region(
        &self,
        session: &OnnxSession,
        image: ImageView<'_>,
        region: &FaceRegion,
    ) -> Result<InputTensor, InferenceError> {
        session.preprocess_face_region(image, region, || self.cancellation.is_cancelled())
    }

    /// Process one fresh active-liveness observation while mapping the transaction signal.
    ///
    /// # Errors
    ///
    /// Returns the bounded challenge error, including [`ChallengeError::Cancelled`].
    pub fn observe_liveness(
        &self,
        challenge: &mut ChallengeSession,
        observation: ChallengeObservation,
    ) -> Result<ChallengeProgress, ChallengeError> {
        challenge.observe_cancellable(observation, || self.cancellation.is_cancelled())
    }

    /// Process active liveness using eye openness and yaw from the exact landmark invocation.
    ///
    /// # Errors
    ///
    /// Returns the bounded challenge error, including cancellation, invalid timing or model
    /// measurements, repeated frames, and deadline expiry.
    pub fn observe_landmark_liveness(
        &self,
        challenge: &mut ChallengeSession,
        timing: CapturePair,
        infrared_sequence: u32,
        visible_sequence: u32,
        landmarks: &FacialLandmarks,
    ) -> Result<ChallengeProgress, ChallengeError> {
        let measurements = landmarks.measurements();
        self.observe_liveness(
            challenge,
            ChallengeObservation {
                timing,
                infrared_sequence,
                visible_sequence,
                face_count: 1,
                eye_openness: measurements.eye_openness(),
                yaw_degrees: measurements.yaw_degrees(),
            },
        )
    }
}

/// Source of authenticated, UID-addressed biometric templates.
pub trait TemplateSource {
    /// Load and authenticate the template belonging to `uid`.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] when no usable authenticated record can be produced.
    fn load_template(&self, uid: u32) -> Result<TemplateRecord, StorageError>;
}

impl<K> TemplateSource for EncryptedTemplateStore<K>
where
    K: KeyProvider,
{
    fn load_template(&self, uid: u32) -> Result<TemplateRecord, StorageError> {
        self.load(uid)
    }
}

/// Authenticated enrollment-state reader suitable for the bounded Manager1 storage worker.
pub struct AuthenticatedTemplateStateReader<S> {
    source: S,
}

impl<S> AuthenticatedTemplateStateReader<S> {
    /// Wrap a daemon template source without exposing template contents to D-Bus types.
    #[must_use]
    pub const fn new(source: S) -> Self {
        Self { source }
    }
}

impl<S> TemplateStateReader for AuthenticatedTemplateStateReader<S>
where
    S: TemplateSource + Send + 'static,
{
    fn is_enrolled(&self, target_uid: u32) -> Result<bool, ManagementBackendError> {
        match self.source.load_template(target_uid) {
            Ok(record) => {
                drop(record);
                Ok(true)
            }
            Err(StorageError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                Ok(false)
            }
            Err(_) => {
                Err(ManagementBackendError::new("unable to authenticate encrypted template state"))
            }
        }
    }
}

/// Build the bounded Manager1 template-state source owned by the daemon.
///
/// Authenticated storage reads run on one dedicated worker rather than the zbus executor. The
/// returned source exposes only a boolean enrollment state.
///
/// # Errors
///
/// Returns [`ManagementBackendError`] for invalid queue bounds or worker creation failure.
pub fn management_template_state_source<S>(
    source: S,
    queue_capacity: usize,
) -> Result<BoundedEnrollmentStateSource, ManagementBackendError>
where
    S: TemplateSource + Send + 'static,
{
    BoundedEnrollmentStateSource::new(AuthenticatedTemplateStateReader::new(source), queue_capacity)
}

/// Inputs bound to one admitted authentication transaction completion.
#[derive(Clone, Copy)]
pub struct AuthenticationCompletion<'a> {
    /// Connection identity created by the daemon for the admitted socket.
    pub connection: ConnectionToken,
    /// Exact active transaction identifier.
    pub transaction_id: TransactionId,
    /// Final authentication policy.
    pub policy: AuthPolicy,
    /// Exact calibrated passive PAD contracts.
    pub passive_pad_policy: &'a PassivePadPolicy,
    /// Complete ephemeral derived evidence.
    pub observation: AuthenticationObservation<'a>,
    /// Current monotonic time used for the terminal deadline check.
    pub now_micros: u64,
}

/// Load the exact active UID's encrypted template, evaluate derived evidence, and consume the
/// connection-bound transaction with one terminal response.
///
/// Storage and malformed-evidence failures deliberately collapse to `InternalError`; only valid
/// biometric policy decisions are exposed. The session manager replaces any result with
/// `TimedOut` once its monotonic deadline has elapsed.
///
/// # Errors
///
/// Returns [`SessionError`] when the connection/transaction binding is invalid or no active
/// transaction exists.
pub fn complete_authentication<S: TemplateSource>(
    sessions: &mut SessionManager,
    templates: &S,
    completion: AuthenticationCompletion<'_>,
) -> Result<Response, SessionError> {
    let target_uid = sessions.target_uid(completion.connection, completion.transaction_id)?;
    let decision =
        templates.load_template(target_uid).map_or(DecisionCode::InternalError, |template| {
            evaluate_authentication(
                completion.policy,
                completion.passive_pad_policy,
                &template,
                completion.observation,
            )
            .map_or(DecisionCode::InternalError, |result| decision_code(result.reason))
        });
    sessions.complete(
        completion.connection,
        completion.transaction_id,
        decision,
        completion.now_micros,
    )
}

const fn decision_code(reason: DecisionReason) -> DecisionCode {
    match reason {
        DecisionReason::Accepted => DecisionCode::Accepted,
        DecisionReason::InfraredMissing => DecisionCode::InfraredMissing,
        DecisionReason::VisibleMissing => DecisionCode::VisibleMissing,
        DecisionReason::InsufficientQuality => DecisionCode::InsufficientQuality,
        DecisionReason::PassiveLivenessFailed => DecisionCode::PassiveLivenessFailed,
        DecisionReason::ActiveChallengeFailed => DecisionCode::ActiveChallengeFailed,
        DecisionReason::FaceMismatch => DecisionCode::FaceMismatch,
    }
}

/// Decodes and admits requests through executable verification, authorization, and session state.
pub struct BoundaryService {
    authorization: AuthorizationPolicy,
    sessions: SessionManager,
}

impl BoundaryService {
    /// Construct a boundary service from validated authorization and session policies.
    #[must_use]
    pub const fn new(authorization: AuthorizationPolicy, sessions: SessionManager) -> Self {
        Self { authorization, sessions }
    }

    /// Read and process one request from an already peer-credentialed connection.
    ///
    /// Successful authentication admission emits `Started`; no biometric success is produced here.
    /// Cancellation is accepted only for the exact connection-bound active transaction.
    /// Rejections are written as closed protocol responses.
    ///
    /// # Errors
    ///
    /// Returns [`BoundaryError`] only when framing or response I/O fails. Authorization and session
    /// failures are converted to fail-closed protocol rejections.
    pub fn handle_one(
        &mut self,
        stream: &mut PeerStream,
        connection: ConnectionToken,
        now_micros: u64,
    ) -> Result<Response, BoundaryError> {
        let request = stream.read_message::<Request>()?.message;
        let response = match request {
            Request::Authenticate { context } => {
                let transaction_id = context.transaction_id;
                let executable = VerifiedExecutable::from_peer(stream.peer(), stream.peer_pidfd());
                match executable.map_err(ExecutableOrAuthorizationError::from).and_then(
                    |executable| {
                        self.authorization
                            .authorize(stream.peer(), Some(executable), &context)
                            .map_err(ExecutableOrAuthorizationError::Authorization)
                    },
                ) {
                    Ok(grant) => match self.sessions.start(grant, connection, now_micros) {
                        Ok(response) => response,
                        Err(SessionError::Busy) => Response::Rejected {
                            transaction_id: Some(transaction_id),
                            reason: RejectionCode::Busy,
                        },
                        Err(_) => Response::Rejected {
                            transaction_id: Some(transaction_id),
                            reason: RejectionCode::ServiceUnavailable,
                        },
                    },
                    Err(ExecutableOrAuthorizationError::Authorization(
                        AuthorizationError::ServiceNotAllowed,
                    )) => Response::Rejected {
                        transaction_id: Some(transaction_id),
                        reason: RejectionCode::ServiceNotAllowed,
                    },
                    Err(_) => Response::Rejected {
                        transaction_id: Some(transaction_id),
                        reason: RejectionCode::UnauthorizedPeer,
                    },
                }
            }
            Request::Cancel { transaction_id } => {
                match self.sessions.cancel(connection, transaction_id, now_micros) {
                    Ok(response) => response,
                    Err(SessionError::WrongConnection | SessionError::WrongTransaction) => {
                        Response::Rejected {
                            transaction_id: Some(transaction_id),
                            reason: RejectionCode::UnauthorizedPeer,
                        }
                    }
                    Err(_) => Response::Rejected {
                        transaction_id: Some(transaction_id),
                        reason: RejectionCode::ServiceUnavailable,
                    },
                }
            }
        };
        stream.write_message(response.clone())?;
        Ok(response)
    }

    /// Borrow the transaction manager for the internal capture/inference pipeline.
    #[must_use]
    pub const fn sessions(&mut self) -> &mut SessionManager {
        &mut self.sessions
    }

    /// Create the only engine job admitted for an exact active socket transaction.
    ///
    /// # Errors
    ///
    /// Returns [`BoundaryError`] when the connection or transaction does not own the active slot.
    pub fn authentication_job(
        &self,
        connection: ConnectionToken,
        transaction_id: TransactionId,
    ) -> Result<AuthenticationJob, BoundaryError> {
        Ok(AuthenticationJob::from_session(&self.sessions, connection, transaction_id)?)
    }

    /// Emit one transaction-bound, non-biometric progress response on the admitted connection.
    ///
    /// If the session deadline has elapsed, the emitted response is terminal `TimedOut` and the
    /// transaction is consumed.
    ///
    /// # Errors
    ///
    /// Returns [`BoundaryError`] for an invalid session binding or response transport failure.
    pub fn send_progress(
        &mut self,
        stream: &mut PeerStream,
        connection: ConnectionToken,
        transaction_id: TransactionId,
        progress: ProgressCode,
        now_micros: u64,
    ) -> Result<Response, BoundaryError> {
        let response = self.sessions.progress(connection, transaction_id, progress, now_micros)?;
        stream.write_message(response.clone())?;
        Ok(response)
    }

    /// Complete one authentication transaction and write its terminal response on the admitted
    /// connection.
    ///
    /// The session is consumed before the response write. A broken client connection therefore
    /// cannot retry or replay a successful terminal result.
    ///
    /// # Errors
    ///
    /// Returns [`BoundaryError`] for an invalid session binding or response transport failure.
    pub fn send_completion<S: TemplateSource>(
        &mut self,
        stream: &mut PeerStream,
        templates: &S,
        completion: AuthenticationCompletion<'_>,
    ) -> Result<Response, BoundaryError> {
        let response = complete_authentication(&mut self.sessions, templates, completion)?;
        stream.write_message(response.clone())?;
        Ok(response)
    }

    /// Evaluate a successful engine result against authenticated storage and emit one terminal
    /// transaction-bound response.
    ///
    /// # Errors
    ///
    /// Returns [`BoundaryError`] for invalid session binding or terminal response delivery failure.
    pub fn send_derived_completion<S: TemplateSource>(
        &mut self,
        stream: &mut PeerStream,
        templates: &S,
        connection: ConnectionToken,
        policy: AuthPolicy,
        passive_pad_policy: &PassivePadPolicy,
        evidence: &DerivedAuthenticationEvidence,
    ) -> Result<Response, BoundaryError> {
        self.send_completion(
            stream,
            templates,
            AuthenticationCompletion {
                connection,
                transaction_id: evidence.transaction_id(),
                policy,
                passive_pad_policy,
                observation: evidence.observation(),
                now_micros: evidence.completed_at_micros(),
            },
        )
    }

    /// Consume an active transaction with a sanitized internal failure and write its terminal
    /// response.
    ///
    /// # Errors
    ///
    /// Returns [`BoundaryError`] for invalid binding or terminal response delivery failure.
    pub fn send_engine_failure(
        &mut self,
        stream: &mut PeerStream,
        connection: ConnectionToken,
        transaction_id: TransactionId,
        now_micros: u64,
    ) -> Result<Response, BoundaryError> {
        let response = self.sessions.complete(
            connection,
            transaction_id,
            DecisionCode::InternalError,
            now_micros,
        )?;
        stream.write_message(response.clone())?;
        Ok(response)
    }
}

/// Policies required to evaluate one engine result at the socket boundary.
#[derive(Clone, Copy)]
pub struct AuthenticationConnectionPolicy<'a> {
    /// Final recognition, capture, quality, and fallback policy.
    pub authentication: AuthPolicy,
    /// Exact calibrated passive PAD model identities and thresholds.
    pub passive_pad: &'a PassivePadPolicy,
}

/// Run one complete authenticated socket transaction through the biometric engine.
///
/// Client cancellation and disconnect signal the exact session token before the coordinator drains
/// the bounded engine terminal update. Only one terminal response is ever written.
///
/// # Errors
///
/// Returns [`BoundaryError`] for transport, session binding, engine submission, or unexpected
/// engine-worker termination.
pub fn coordinate_authentication_connection<S, F>(
    boundary: &mut BoundaryService,
    stream: &mut PeerStream,
    connection: ConnectionToken,
    engine: &AuthenticationEngineClient,
    templates: &S,
    policy: AuthenticationConnectionPolicy<'_>,
    mut now_micros: F,
) -> Result<Response, BoundaryError>
where
    S: TemplateSource,
    F: FnMut() -> u64,
{
    let admission = boundary.handle_one(stream, connection, now_micros())?;
    let Response::Started { transaction_id } = admission else {
        return Ok(admission);
    };
    coordinate_started_authentication(
        boundary,
        stream,
        connection,
        transaction_id,
        engine,
        templates,
        policy,
        now_micros,
    )
}

#[allow(clippy::too_many_arguments)]
fn coordinate_started_authentication<S, F>(
    boundary: &mut BoundaryService,
    stream: &mut PeerStream,
    connection: ConnectionToken,
    transaction_id: TransactionId,
    engine: &AuthenticationEngineClient,
    templates: &S,
    policy: AuthenticationConnectionPolicy<'_>,
    mut now_micros: F,
) -> Result<Response, BoundaryError>
where
    S: TemplateSource,
    F: FnMut() -> u64,
{
    let job = boundary.authentication_job(connection, transaction_id)?;
    let Ok(handle) = engine.submit(job) else {
        return boundary.send_engine_failure(stream, connection, transaction_id, now_micros());
    };

    loop {
        match handle.recv_timeout(CONNECTION_POLL_INTERVAL) {
            Ok(AuthenticationEngineUpdate::Progress {
                transaction_id: update_transaction,
                progress,
            }) if update_transaction == transaction_id => {
                let response = boundary.send_progress(
                    stream,
                    connection,
                    transaction_id,
                    progress,
                    now_micros(),
                )?;
                if matches!(response, Response::Completed { .. }) {
                    drain_cancelled_engine(&handle)?;
                    return Ok(response);
                }
            }
            Ok(AuthenticationEngineUpdate::Completed {
                transaction_id: update_transaction,
                result,
            }) if update_transaction == transaction_id => {
                return match result {
                    Ok(evidence) => boundary.send_derived_completion(
                        stream,
                        templates,
                        connection,
                        policy.authentication,
                        policy.passive_pad,
                        &evidence,
                    ),
                    Err(_) => boundary.send_engine_failure(
                        stream,
                        connection,
                        transaction_id,
                        now_micros(),
                    ),
                };
            }
            Ok(_) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                return boundary.send_engine_failure(
                    stream,
                    connection,
                    transaction_id,
                    now_micros(),
                );
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
        }

        match stream.has_pending_input() {
            Ok(false) => {}
            Ok(true) => match boundary.handle_one(stream, connection, now_micros()) {
                Ok(response @ Response::Completed { .. }) => {
                    drain_cancelled_engine(&handle)?;
                    return Ok(response);
                }
                Ok(_) => {}
                Err(error) => {
                    let _cancel_result =
                        boundary.sessions().cancel(connection, transaction_id, now_micros());
                    drain_cancelled_engine(&handle)?;
                    return Err(error);
                }
            },
            Err(error) => {
                let _cancel_result =
                    boundary.sessions().cancel(connection, transaction_id, now_micros());
                drain_cancelled_engine(&handle)?;
                return Err(BoundaryError::Transport(error));
            }
        }
    }
}

fn drain_cancelled_engine(handle: &AuthenticationEngineHandle) -> Result<(), BoundaryError> {
    for _ in 0..MAX_CANCEL_DRAIN_POLLS {
        match handle.recv_timeout(CONNECTION_POLL_INTERVAL) {
            Ok(AuthenticationEngineUpdate::Completed { .. }) => return Ok(()),
            Ok(AuthenticationEngineUpdate::Progress { .. })
            | Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                return Err(BoundaryError::EngineStopped);
            }
        }
    }
    Err(BoundaryError::EngineStopped)
}

enum ExecutableOrAuthorizationError {
    Executable,
    Authorization(AuthorizationError),
}

impl From<ExecutableError> for ExecutableOrAuthorizationError {
    fn from(_error: ExecutableError) -> Self {
        Self::Executable
    }
}

/// Boundary framing or response delivery failure.
#[derive(Debug, Error)]
pub enum BoundaryError {
    /// Local protocol transport failed.
    #[error("daemon authentication boundary transport failed: {0}")]
    Transport(#[from] TransportError),
    /// Authentication transaction binding or lifecycle failed.
    #[error("daemon authentication session failed: {0}")]
    Session(#[from] SessionError),
    /// The engine could not admit this exact transaction.
    #[error("daemon authentication engine submission failed: {0}")]
    EngineSubmit(AuthenticationEngineSubmitError),
    /// The engine worker ended or failed to terminate within its cancellation bound.
    #[error("daemon authentication engine stopped unexpectedly")]
    EngineStopped,
    /// An engine update was not bound to the submitted transaction.
    #[error("daemon authentication engine returned a foreign transaction update")]
    UnexpectedEngineUpdate,
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, os::unix::net::UnixStream, thread};

    use faceauth_authz::{
        AuthorizationGrant, AuthorizationRule, CallerRelation, ExecutableFingerprint,
    };
    use faceauth_management_dbus::EnrollmentStateSource;
    use faceauth_protocol::{
        AuthenticationPurpose, Envelope, RequestContext, ServiceName, TransactionId,
    };
    use faceauth_session::SessionConfig;
    use faceauth_transport::TransportConfig;

    use super::*;

    fn test_embedding(digest_byte: u8) -> Result<FaceEmbedding, InferenceError> {
        let digest = format!("{digest_byte:02x}").repeat(32);
        let mut values = vec![0.0; 32];
        values[0] = 1.0;
        FaceEmbedding::from_normalized_template(&digest, &values)
    }

    fn enrollment_config() -> EnrollmentConfig {
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

    fn pad_policy() -> PassivePadPolicy {
        PassivePadPolicy {
            infrared_compatibility_sha256: "11".repeat(32),
            minimum_infrared_probability: 0.8,
            visible_compatibility_sha256: "22".repeat(32),
            minimum_visible_probability: 0.75,
            fusion: None,
        }
    }

    fn pad_scores(
        infrared: f32,
        visible: f32,
    ) -> Result<Vec<PassiveLivenessScore>, InferenceError> {
        Ok(vec![
            PassiveLivenessScore::from_validated_output(
                ModelRole::PassiveLivenessInfrared,
                &"11".repeat(32),
                infrared,
            )?,
            PassiveLivenessScore::from_validated_output(
                ModelRole::PassiveLivenessVisible,
                &"22".repeat(32),
                visible,
            )?,
        ])
    }

    struct CoordinatorEngine;

    impl AuthenticationEngine for CoordinatorEngine {
        fn authenticate(
            &mut self,
            job: &AuthenticationJob,
            progress: &mut dyn FnMut(ProgressCode) -> Result<(), AuthenticationEngineFailure>,
        ) -> Result<DerivedAuthenticationEvidence, AuthenticationEngineFailure> {
            progress(ProgressCode::HoldStill)?;
            progress(ProgressCode::Processing)?;
            let embedding =
                test_embedding(0xaa).map_err(|_| AuthenticationEngineFailure::Internal)?;
            let scores =
                pad_scores(0.99, 0.99).map_err(|_| AuthenticationEngineFailure::Internal)?;
            Ok(DerivedAuthenticationEvidence::new(
                job.transaction_id(),
                CapturePair {
                    infrared_timestamp_micros: 2_000_000,
                    visible_timestamp_micros: Some(2_010_000),
                },
                0.95,
                embedding,
                scores,
                ChallengeProgress::Passed,
                2_100_000,
            ))
        }
    }

    struct TestTemplateSource {
        record: Option<TemplateRecord>,
        requested_uid: Cell<Option<u32>>,
    }

    impl TemplateSource for TestTemplateSource {
        fn load_template(&self, uid: u32) -> Result<TemplateRecord, StorageError> {
            self.requested_uid.set(Some(uid));
            self.record.clone().ok_or(StorageError::InvalidRecord("test template unavailable"))
        }
    }

    enum StateTemplateMode {
        Present,
        Missing,
        Corrupt,
    }

    struct StateTemplateSource(StateTemplateMode);

    impl TemplateSource for StateTemplateSource {
        fn load_template(&self, uid: u32) -> Result<TemplateRecord, StorageError> {
            match self.0 {
                StateTemplateMode::Present => Ok(TemplateRecord {
                    schema_version: TEMPLATE_RECORD_SCHEMA_VERSION,
                    uid,
                    model_compatibility_sha256: "aa".repeat(32),
                    embedding: vec![1.0, 0.0, 0.0],
                }),
                StateTemplateMode::Missing => Err(StorageError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "missing test template",
                ))),
                StateTemplateMode::Corrupt => Err(StorageError::AuthenticationFailed),
            }
        }
    }

    fn active_session(
        target_uid: u32,
    ) -> Result<(SessionManager, ConnectionToken, TransactionId), Box<dyn std::error::Error>> {
        let mut sessions = SessionManager::new(SessionConfig::default())?;
        let connection = ConnectionToken::generate()?;
        let transaction_id = TransactionId::generate();
        let grant = AuthorizationGrant {
            transaction_id,
            peer: faceauth_transport::PeerIdentity { pid: 123, uid: 0, gid: 0 },
            target_uid,
            service: ServiceName::parse("faceauth-test")?,
            purpose: AuthenticationPurpose::Test,
            executable: ExecutableFingerprint { device: 1, inode: 1 },
        };
        let _ = sessions.start(grant, connection, 1_000_000)?;
        Ok((sessions, connection, transaction_id))
    }

    #[test]
    fn enrollment_requires_exact_root_polkit_broker_grant() -> Result<(), Box<dyn std::error::Error>>
    {
        let service = ServiceName::parse(faceauth_management::ENROLLMENT_SERVICE)?;
        let grant = AuthorizationGrant {
            transaction_id: TransactionId::generate(),
            peer: faceauth_transport::PeerIdentity { pid: 10, uid: 0, gid: 0 },
            target_uid: 1000,
            service,
            purpose: AuthenticationPurpose::Polkit,
            executable: ExecutableFingerprint { device: 1, inode: 1 },
        };
        let session = begin_authorized_enrollment(&grant, enrollment_config(), 1_000_000)?;
        assert_eq!(session.accepted_samples(), 0);

        let mut wrong_peer = grant;
        wrong_peer.peer.uid = 1000;
        assert!(matches!(
            begin_authorized_enrollment(&wrong_peer, enrollment_config(), 1_000_000),
            Err(EnrollmentBoundaryError::UnauthorizedGrant)
        ));
        Ok(())
    }

    #[test]
    fn management_state_reader_authenticates_without_exposing_templates()
    -> Result<(), Box<dyn std::error::Error>> {
        let source =
            management_template_state_source(StateTemplateSource(StateTemplateMode::Present), 2)?;
        assert!(futures_lite::future::block_on(source.is_enrolled(1000))?);
        assert!(
            !AuthenticatedTemplateStateReader::new(StateTemplateSource(StateTemplateMode::Missing))
                .is_enrolled(1000)?
        );
        assert!(
            AuthenticatedTemplateStateReader::new(StateTemplateSource(StateTemplateMode::Corrupt))
                .is_enrolled(1000)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn enrollment_template_round_trips_into_compatible_comparison()
    -> Result<(), Box<dyn std::error::Error>> {
        let observed = test_embedding(0xaa)?;
        let template = template_from_embedding(1000, &observed);
        template.validate()?;
        assert_eq!(template.uid, 1000);
        assert!((compare_template(&template, &observed)? - 1.0).abs() < 1.0e-6);
        Ok(())
    }

    #[test]
    fn comparison_rejects_a_different_manifest_digest() -> Result<(), Box<dyn std::error::Error>> {
        let enrolled = test_embedding(0xaa)?;
        let observed = test_embedding(0xbb)?;
        let template = template_from_embedding(1000, &enrolled);
        assert!(matches!(
            compare_template(&template, &observed),
            Err(BiometricComparisonError::Storage(StorageError::IncompatibleTemplate))
        ));
        Ok(())
    }

    #[test]
    fn complete_bound_evidence_can_be_accepted() -> Result<(), Box<dyn std::error::Error>> {
        let embedding = test_embedding(0xaa)?;
        let template = template_from_embedding(1000, &embedding);
        let scores = pad_scores(0.95, 0.9)?;
        let decision = evaluate_authentication(
            AuthPolicy::default(),
            &pad_policy(),
            &template,
            AuthenticationObservation {
                timing: CapturePair {
                    infrared_timestamp_micros: 1_000_000,
                    visible_timestamp_micros: Some(1_050_000),
                },
                quality: 0.9,
                embedding: &embedding,
                passive_liveness: &scores,
                active_challenge: ChallengeProgress::Passed,
            },
        )?;
        assert!(decision.accepted);
        assert_eq!(decision.reason, faceauth_core::DecisionReason::Accepted);
        Ok(())
    }

    #[test]
    fn pad_failure_and_incomplete_active_challenge_reject() -> Result<(), Box<dyn std::error::Error>>
    {
        let embedding = test_embedding(0xaa)?;
        let template = template_from_embedding(1000, &embedding);
        let scores = pad_scores(0.79, 0.99)?;
        let decision = evaluate_authentication(
            AuthPolicy::default(),
            &pad_policy(),
            &template,
            AuthenticationObservation {
                timing: CapturePair {
                    infrared_timestamp_micros: 1_000_000,
                    visible_timestamp_micros: Some(1_010_000),
                },
                quality: 0.9,
                embedding: &embedding,
                passive_liveness: &scores,
                active_challenge: ChallengeProgress::BaselineRequired,
            },
        )?;
        assert!(!decision.accepted);
        assert_eq!(decision.reason, faceauth_core::DecisionReason::PassiveLivenessFailed);
        Ok(())
    }

    #[test]
    fn missing_duplicate_or_wrong_contract_pad_evidence_fails_closed()
    -> Result<(), Box<dyn std::error::Error>> {
        let embedding = test_embedding(0xaa)?;
        let template = template_from_embedding(1000, &embedding);
        let mut scores = pad_scores(0.99, 0.99)?;
        scores.push(PassiveLivenessScore::from_validated_output(
            ModelRole::PassiveLivenessInfrared,
            &"11".repeat(32),
            0.99,
        )?);
        let result = evaluate_authentication(
            AuthPolicy::default(),
            &pad_policy(),
            &template,
            AuthenticationObservation {
                timing: CapturePair {
                    infrared_timestamp_micros: 1_000_000,
                    visible_timestamp_micros: Some(1_010_000),
                },
                quality: 0.9,
                embedding: &embedding,
                passive_liveness: &scores,
                active_challenge: ChallengeProgress::Passed,
            },
        );
        assert!(matches!(result, Err(AuthenticationPipelineError::PassivePadEvidenceInvalid)));
        Ok(())
    }

    #[test]
    fn completion_loads_only_the_session_authorized_uid() -> Result<(), Box<dyn std::error::Error>>
    {
        let embedding = test_embedding(0xaa)?;
        let source = TestTemplateSource {
            record: Some(template_from_embedding(4242, &embedding)),
            requested_uid: Cell::new(None),
        };
        let scores = pad_scores(0.99, 0.99)?;
        let (mut sessions, connection, transaction_id) = active_session(4242)?;
        let response = complete_authentication(
            &mut sessions,
            &source,
            AuthenticationCompletion {
                connection,
                transaction_id,
                policy: AuthPolicy::default(),
                passive_pad_policy: &pad_policy(),
                observation: AuthenticationObservation {
                    timing: CapturePair {
                        infrared_timestamp_micros: 2_000_000,
                        visible_timestamp_micros: Some(2_010_000),
                    },
                    quality: 0.9,
                    embedding: &embedding,
                    passive_liveness: &scores,
                    active_challenge: ChallengeProgress::Passed,
                },
                now_micros: 3_000_000,
            },
        )?;
        assert_eq!(source.requested_uid.get(), Some(4242));
        assert_eq!(
            response,
            Response::Completed { transaction_id, decision: DecisionCode::Accepted }
        );
        assert!(!sessions.is_busy());
        Ok(())
    }

    #[test]
    fn admitted_socket_receives_started_progress_and_terminal_sequence()
    -> Result<(), Box<dyn std::error::Error>> {
        let embedding = test_embedding(0xaa)?;
        let source = TestTemplateSource {
            record: Some(template_from_embedding(1000, &embedding)),
            requested_uid: Cell::new(None),
        };
        let scores = pad_scores(0.99, 0.99)?;
        let rule = AuthorizationRule {
            service: ServiceName::parse("faceauth-test")?,
            purpose: AuthenticationPurpose::Test,
            caller: CallerRelation::RootOnly,
            executables: vec![ExecutableFingerprint { device: 1, inode: 1 }],
        };
        let mut service = BoundaryService::new(
            AuthorizationPolicy::new(vec![rule])?,
            SessionManager::new(SessionConfig::default())?,
        );
        let connection = ConnectionToken::generate()?;
        let transaction_id = TransactionId::generate();
        let grant = AuthorizationGrant {
            transaction_id,
            peer: faceauth_transport::PeerIdentity { pid: 123, uid: 0, gid: 0 },
            target_uid: 1000,
            service: ServiceName::parse("faceauth-test")?,
            purpose: AuthenticationPurpose::Test,
            executable: ExecutableFingerprint { device: 1, inode: 1 },
        };
        let started = service.sessions().start(grant, connection, 1_000_000)?;
        let job = service.authentication_job(connection, transaction_id)?;
        assert_eq!(job.transaction_id(), transaction_id);
        assert_eq!(job.target_uid(), 1000);
        let evidence = DerivedAuthenticationEvidence::new(
            transaction_id,
            CapturePair {
                infrared_timestamp_micros: 2_000_000,
                visible_timestamp_micros: Some(2_010_000),
            },
            0.9,
            embedding,
            scores,
            ChallengeProgress::Passed,
            3_000_000,
        );
        let (client, server) = UnixStream::pair()?;
        let mut client = PeerStream::connect(client, TransportConfig::default())?;
        let mut server = PeerStream::connect(server, TransportConfig::default())?;
        server.write_message(started)?;
        service.send_progress(
            &mut server,
            connection,
            transaction_id,
            ProgressCode::Processing,
            2_000_000,
        )?;
        service.send_derived_completion(
            &mut server,
            &source,
            connection,
            AuthPolicy::default(),
            &pad_policy(),
            &evidence,
        )?;

        assert_eq!(
            client.read_message::<Response>()?.message,
            Response::Started { transaction_id }
        );
        assert_eq!(
            client.read_message::<Response>()?.message,
            Response::Progress { transaction_id, progress: ProgressCode::Processing }
        );
        assert_eq!(
            client.read_message::<Response>()?.message,
            Response::Completed { transaction_id, decision: DecisionCode::Accepted }
        );
        assert!(!service.sessions().is_busy());
        Ok(())
    }

    #[test]
    fn connection_coordinator_relays_engine_updates_and_one_terminal()
    -> Result<(), Box<dyn std::error::Error>> {
        let embedding = test_embedding(0xaa)?;
        let source = TestTemplateSource {
            record: Some(template_from_embedding(1000, &embedding)),
            requested_uid: Cell::new(None),
        };
        let rule = AuthorizationRule {
            service: ServiceName::parse("faceauth-test")?,
            purpose: AuthenticationPurpose::Test,
            caller: CallerRelation::RootOnly,
            executables: vec![ExecutableFingerprint { device: 1, inode: 1 }],
        };
        let mut boundary = BoundaryService::new(
            AuthorizationPolicy::new(vec![rule])?,
            SessionManager::new(SessionConfig::default())?,
        );
        let connection = ConnectionToken::generate()?;
        let transaction_id = TransactionId::generate();
        let grant = AuthorizationGrant {
            transaction_id,
            peer: faceauth_transport::PeerIdentity { pid: 123, uid: 0, gid: 0 },
            target_uid: 1000,
            service: ServiceName::parse("faceauth-test")?,
            purpose: AuthenticationPurpose::Test,
            executable: ExecutableFingerprint { device: 1, inode: 1 },
        };
        let _started = boundary.sessions().start(grant, connection, 1_000_000)?;
        let (client, server) = UnixStream::pair()?;
        let mut client = PeerStream::connect(client, TransportConfig::default())?;
        let mut server = PeerStream::connect(server, TransportConfig::default())?;
        let (engine, service) = AuthenticationEngineService::new(CoordinatorEngine);
        let shutdown = ShutdownToken::default();
        let worker_shutdown = shutdown.clone();
        let worker = thread::spawn(move || service.run(&worker_shutdown));
        let terminal = coordinate_started_authentication(
            &mut boundary,
            &mut server,
            connection,
            transaction_id,
            &engine,
            &source,
            AuthenticationConnectionPolicy {
                authentication: AuthPolicy::default(),
                passive_pad: &pad_policy(),
            },
            || 2_050_000,
        )?;
        assert_eq!(
            terminal,
            Response::Completed { transaction_id, decision: DecisionCode::Accepted }
        );
        assert_eq!(
            client.read_message::<Response>()?.message,
            Response::Progress { transaction_id, progress: ProgressCode::HoldStill }
        );
        assert_eq!(
            client.read_message::<Response>()?.message,
            Response::Progress { transaction_id, progress: ProgressCode::Processing }
        );
        assert_eq!(client.read_message::<Response>()?.message, terminal);
        assert!(!boundary.sessions().is_busy());
        let _first = shutdown.request();
        worker.join().map_err(|_| "engine worker panicked")??;
        Ok(())
    }

    #[test]
    fn storage_failure_consumes_session_as_internal_error() -> Result<(), Box<dyn std::error::Error>>
    {
        let embedding = test_embedding(0xaa)?;
        let source = TestTemplateSource { record: None, requested_uid: Cell::new(None) };
        let scores = pad_scores(0.99, 0.99)?;
        let (mut sessions, connection, transaction_id) = active_session(1000)?;
        let response = complete_authentication(
            &mut sessions,
            &source,
            AuthenticationCompletion {
                connection,
                transaction_id,
                policy: AuthPolicy::default(),
                passive_pad_policy: &pad_policy(),
                observation: AuthenticationObservation {
                    timing: CapturePair {
                        infrared_timestamp_micros: 2_000_000,
                        visible_timestamp_micros: Some(2_010_000),
                    },
                    quality: 0.9,
                    embedding: &embedding,
                    passive_liveness: &scores,
                    active_challenge: ChallengeProgress::Passed,
                },
                now_micros: 3_000_000,
            },
        )?;
        assert_eq!(
            response,
            Response::Completed { transaction_id, decision: DecisionCode::InternalError }
        );
        assert!(!sessions.is_busy());
        Ok(())
    }

    #[test]
    fn worker_maps_the_bound_session_token_into_liveness() -> Result<(), Box<dyn std::error::Error>>
    {
        let (mut sessions, connection, transaction_id) = active_session(1000)?;
        let worker =
            AuthenticationWorker::new(sessions.cancellation_token(connection, transaction_id)?);
        assert!(!worker.is_cancelled());
        let _ = sessions.cancel(connection, transaction_id, 2_000_000)?;
        assert!(worker.is_cancelled());
        let mut challenge = ChallengeSession::begin(
            faceauth_liveness::ChallengeConfig {
                max_pair_skew_micros: 100_000,
                duration_micros: 10_000_000,
                max_observations: 90,
                open_eye_threshold: 0.65,
                closed_eye_threshold: 0.25,
                neutral_yaw_degrees: 10.0,
                turn_yaw_degrees: 20.0,
                action_consecutive_observations: 2,
                minimum_action_duration_micros: 50_000,
                recovery_consecutive_observations: 2,
            },
            1_000_000,
        )?;
        let result = worker.observe_liveness(
            &mut challenge,
            ChallengeObservation {
                timing: CapturePair {
                    infrared_timestamp_micros: 1_100_000,
                    visible_timestamp_micros: Some(1_110_000),
                },
                infrared_sequence: 1,
                visible_sequence: 1,
                face_count: 1,
                eye_openness: 0.9,
                yaw_degrees: 0.0,
            },
        );
        assert!(matches!(result, Err(ChallengeError::Cancelled)));
        assert_eq!(challenge.progress(), ChallengeProgress::BaselineRequired);
        Ok(())
    }

    #[test]
    fn session_deadline_overrides_a_valid_biometric_result()
    -> Result<(), Box<dyn std::error::Error>> {
        let embedding = test_embedding(0xaa)?;
        let source = TestTemplateSource {
            record: Some(template_from_embedding(1000, &embedding)),
            requested_uid: Cell::new(None),
        };
        let scores = pad_scores(0.99, 0.99)?;
        let (mut sessions, connection, transaction_id) = active_session(1000)?;
        let response = complete_authentication(
            &mut sessions,
            &source,
            AuthenticationCompletion {
                connection,
                transaction_id,
                policy: AuthPolicy::default(),
                passive_pad_policy: &pad_policy(),
                observation: AuthenticationObservation {
                    timing: CapturePair {
                        infrared_timestamp_micros: 2_000_000,
                        visible_timestamp_micros: Some(2_010_000),
                    },
                    quality: 0.9,
                    embedding: &embedding,
                    passive_liveness: &scores,
                    active_challenge: ChallengeProgress::Passed,
                },
                now_micros: 16_000_000,
            },
        )?;
        assert_eq!(
            response,
            Response::Completed { transaction_id, decision: DecisionCode::TimedOut }
        );
        Ok(())
    }

    #[test]
    fn untrusted_development_executable_fails_before_session_start()
    -> Result<(), Box<dyn std::error::Error>> {
        let rule = AuthorizationRule {
            service: ServiceName::parse("faceauth-test")?,
            purpose: AuthenticationPurpose::Test,
            caller: CallerRelation::RootOrTargetUser,
            executables: vec![ExecutableFingerprint { device: 1, inode: 1 }],
        };
        let mut service = BoundaryService::new(
            AuthorizationPolicy::new(vec![rule])?,
            SessionManager::new(SessionConfig::default())?,
        );
        let connection = ConnectionToken::generate()?;
        let (client, server) = UnixStream::pair()?;
        let mut client = PeerStream::connect(client, TransportConfig::default())?;
        let mut server = PeerStream::connect(server, TransportConfig::default())?;
        let transaction_id = TransactionId::generate();
        client.write_message(Request::Authenticate {
            context: RequestContext {
                transaction_id,
                target_uid: server.peer().uid,
                service: ServiceName::parse("faceauth-test")?,
                purpose: AuthenticationPurpose::Test,
            },
        })?;

        assert_eq!(
            service.handle_one(&mut server, connection, 1_000_000)?,
            Response::Rejected {
                transaction_id: Some(transaction_id),
                reason: RejectionCode::UnauthorizedPeer,
            }
        );
        let wire: Envelope<Response> = client.read_message()?;
        assert_eq!(
            wire.message,
            Response::Rejected {
                transaction_id: Some(transaction_id),
                reason: RejectionCode::UnauthorizedPeer,
            }
        );
        assert!(!service.sessions().is_busy());
        Ok(())
    }

    #[test]
    fn cancellation_without_an_active_session_never_succeeds()
    -> Result<(), Box<dyn std::error::Error>> {
        let rule = AuthorizationRule {
            service: ServiceName::parse("faceauth-test")?,
            purpose: AuthenticationPurpose::Test,
            caller: CallerRelation::RootOnly,
            executables: vec![ExecutableFingerprint { device: 1, inode: 1 }],
        };
        let mut service = BoundaryService::new(
            AuthorizationPolicy::new(vec![rule])?,
            SessionManager::new(SessionConfig::default())?,
        );
        let connection = ConnectionToken::generate()?;
        let (client, server) = UnixStream::pair()?;
        let mut client = PeerStream::connect(client, TransportConfig::default())?;
        let mut server = PeerStream::connect(server, TransportConfig::default())?;
        let transaction_id = TransactionId::generate();
        client.write_message(Request::Cancel { transaction_id })?;

        assert_eq!(
            service.handle_one(&mut server, connection, 1_000_000)?,
            Response::Rejected {
                transaction_id: Some(transaction_id),
                reason: RejectionCode::ServiceUnavailable,
            }
        );
        Ok(())
    }

    #[test]
    fn exact_connection_can_cancel_an_internally_started_session()
    -> Result<(), Box<dyn std::error::Error>> {
        let service_name = ServiceName::parse("faceauth-test")?;
        let rule = AuthorizationRule {
            service: service_name.clone(),
            purpose: AuthenticationPurpose::Test,
            caller: CallerRelation::RootOnly,
            executables: vec![ExecutableFingerprint { device: 1, inode: 1 }],
        };
        let mut service = BoundaryService::new(
            AuthorizationPolicy::new(vec![rule])?,
            SessionManager::new(SessionConfig::default())?,
        );
        let connection = ConnectionToken::generate()?;
        let transaction_id = TransactionId::generate();
        let grant = AuthorizationGrant {
            transaction_id,
            peer: faceauth_transport::PeerIdentity { pid: 123, uid: 0, gid: 0 },
            target_uid: 1000,
            service: service_name,
            purpose: AuthenticationPurpose::Test,
            executable: ExecutableFingerprint { device: 1, inode: 1 },
        };
        let _ = service.sessions().start(grant, connection, 1_000_000)?;
        let (client, server) = UnixStream::pair()?;
        let mut client = PeerStream::connect(client, TransportConfig::default())?;
        let mut server = PeerStream::connect(server, TransportConfig::default())?;
        client.write_message(Request::Cancel { transaction_id })?;

        assert_eq!(
            service.handle_one(&mut server, connection, 2_000_000)?,
            Response::Completed {
                transaction_id,
                decision: faceauth_protocol::DecisionCode::Cancelled
            }
        );
        let wire: Envelope<Response> = client.read_message()?;
        assert_eq!(
            wire.message,
            Response::Completed {
                transaction_id,
                decision: faceauth_protocol::DecisionCode::Cancelled
            }
        );
        assert!(!service.sessions().is_busy());
        Ok(())
    }
}
