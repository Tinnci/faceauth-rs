//! Versioned production configuration and fail-closed readiness reporting.

use std::{
    fs::{self, File},
    io::{self, Read},
    os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt},
    path::{Path, PathBuf},
    time::Duration,
};

use faceauth_authz::{AuthorizationPolicy, AuthorizationRule, CallerRelation};
use faceauth_camera::{CameraPairSelector, CameraSelector};
use faceauth_capture::{CaptureSpec, PairingPolicy, V4l2CaptureDevice};
use faceauth_core::CaptureModality;
use faceauth_core::{AuthPolicy, LivenessLevel};
use faceauth_enrollment::EnrollmentConfig;
use faceauth_inference::{
    OnnxSession, RuntimeConfig, verify_model_installation, verify_runtime_installation,
};
use faceauth_liveness::ChallengeConfig;
use faceauth_management::{
    ENROLLMENT_POLKIT_ACTION, ENROLLMENT_SERVICE, MANAGER_BUS_NAME, MANAGER_OBJECT_PATH,
    ManagementConfig,
};
use faceauth_model::{ModelManifest, ModelRole};
use faceauth_protocol::AuthenticationPurpose;
use faceauth_quality::QualityConfig;
use faceauth_session::SessionConfig;
use faceauth_storage::{EncryptedTemplateStore, TpmKeyProvider};
use faceauth_transport::{SecureListener, TransportConfig};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    DetectionPolicy, ProductionAuthenticationEngine, ProductionAuthenticationEngineError,
    supervision::SupervisorConfig,
};

/// Current daemon production-configuration schema.
pub const PRODUCTION_CONFIG_SCHEMA_VERSION: u16 = 4;
/// Current machine-readable readiness-report schema.
pub const READINESS_REPORT_SCHEMA_VERSION: u16 = 1;

const MAX_CONFIG_BYTES: u64 = 1024 * 1024;
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
const MAX_REVIEW_EVIDENCE_BYTES: u64 = 16 * 1024 * 1024;
const REQUIRED_MODEL_ROLES: [ModelRole; 6] = [
    ModelRole::FaceDetector,
    ModelRole::FaceLandmarks,
    ModelRole::FaceEmbedding,
    ModelRole::PassiveLivenessInfrared,
    ModelRole::PassiveLivenessVisible,
    ModelRole::PassiveLivenessFusion,
];

/// Strict, administrator-controlled configuration for a future production daemon composition.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProductionConfig {
    /// Exact configuration schema understood by this build.
    pub schema_version: u16,
    /// Stable selectors and exact bounded capture/pairing requirements.
    pub cameras: ProductionCameraConfig,
    /// Trusted dynamic ONNX Runtime policy.
    pub runtime: RuntimePolicyConfig,
    /// Complete role-bound model installation set.
    pub models: Vec<ModelInstallation>,
    /// Thresholds and reviewed measurements bound to exact model compatibility identities.
    pub calibration: CalibrationConfig,
    /// Encrypted-template and machine-key policy.
    pub storage: StorageConfig,
    /// Exact local caller policy and its review record.
    pub authorization: AuthorizationConfig,
    /// Stable Manager1 lifecycle policy.
    pub management: ManagementServiceConfig,
    /// Fail-fast service supervision and bounded shutdown policy.
    pub supervision: SupervisionPolicyConfig,
    /// Root-owned authentication socket and password-recovery invariants.
    pub authentication_boundary: AuthenticationBoundaryConfig,
}

/// Exact production camera selection, negotiation, and pairing policy.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProductionCameraConfig {
    /// Infrared camera selector and required stream format.
    pub infrared: CameraStreamConfig,
    /// Visible-light camera selector and required stream format.
    pub visible: CameraStreamConfig,
    /// Maximum temporal skew and replacement budget for one observation.
    pub pairing: PairingPolicy,
}

impl ProductionCameraConfig {
    fn selectors(&self) -> CameraPairSelector {
        CameraPairSelector {
            infrared: self.infrared.selector.clone(),
            visible: self.visible.selector.clone(),
        }
    }
}

/// One stable camera selector and exact V4L2 negotiation contract.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CameraStreamConfig {
    /// Stable udev identity and physical location selector.
    pub selector: CameraSelector,
    /// Required dimensions, encoding, frame rate, buffers, warmup, and timeout.
    pub capture: CaptureSpec,
}

/// Serializable ONNX Runtime resource bounds.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimePolicyConfig {
    /// Exact root-controlled ONNX Runtime shared library.
    pub library_path: PathBuf,
    /// Sequential graph intra-op worker count.
    pub intra_threads: usize,
    /// Inter-op count; production requires exactly one.
    pub inter_threads: usize,
    /// Maximum admitted model artifact length.
    pub max_model_bytes: u64,
    /// Watchdog deadline for one graph invocation.
    pub max_run_millis: u32,
}

impl RuntimePolicyConfig {
    fn runtime(&self) -> RuntimeConfig {
        RuntimeConfig {
            library_path: self.library_path.clone(),
            intra_threads: self.intra_threads,
            inter_threads: self.inter_threads,
            max_model_bytes: self.max_model_bytes,
            max_run_millis: self.max_run_millis,
        }
    }
}

/// One exact manifest, model artifact, and security review record.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelInstallation {
    /// Pipeline role this entry must provide.
    pub role: ModelRole,
    /// Root-controlled schema-v9 manifest.
    pub manifest_path: PathBuf,
    /// Root-controlled ONNX artifact.
    pub artifact_path: PathBuf,
    /// Immutable review report covering provenance, license, evaluation, and attack testing.
    pub review_evidence: ReviewedEvidence,
}

/// Immutable, root-controlled review artifact identified by SHA-256.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReviewedEvidence {
    /// Absolute path to the review or isolated-test report.
    pub path: PathBuf,
    /// Lowercase SHA-256 of the exact report bytes.
    pub sha256: String,
}

/// All calibrated biometric and enrollment policy.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationConfig {
    /// Recognition, quality, capture, and fallback policy.
    pub recognition: RecognitionCalibration,
    /// Image/face quality policy bound to detector and landmark contracts.
    pub quality: QualityCalibration,
    /// Exact multi-modal passive PAD thresholds.
    pub passive_pad: PassivePadCalibration,
    /// Randomized active-liveness bounds.
    pub active_liveness: ActiveLivenessCalibration,
    /// Multi-observation enrollment bounds.
    pub enrollment: EnrollmentCalibration,
}

/// Quality algorithm calibration bound to exact detector and landmark contracts.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualityCalibration {
    /// Full compatibility digest of the admitted face detector manifest.
    pub detector_compatibility_sha256: String,
    /// Full compatibility digest of the admitted landmark manifest.
    pub landmarks_compatibility_sha256: String,
    /// Minimum calibrated detector confidence admitted before NMS.
    pub minimum_detection_confidence: f32,
    /// Maximum calibrated `IoU` treated as the same detected face.
    pub maximum_detection_iou: f32,
    /// Hard ceiling for distinct faces after NMS.
    pub maximum_detected_faces: usize,
    /// Resource-bounded image and geometry quality policy.
    pub policy: QualityConfig,
    /// Camera, lighting, pose, occlusion, and operating-threshold evaluation report.
    pub evidence: ReviewedEvidence,
}

impl QualityCalibration {
    /// Convert reviewed detector calibration into the production engine policy.
    #[must_use]
    pub const fn detection_policy(&self) -> DetectionPolicy {
        DetectionPolicy {
            minimum_confidence: self.minimum_detection_confidence,
            maximum_iou: self.maximum_detection_iou,
            maximum_faces: self.maximum_detected_faces,
        }
    }
}

/// Recognition policy bound to one embedding model and measured operating point.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecognitionCalibration {
    /// Full compatibility digest of the admitted embedding manifest.
    pub embedding_compatibility_sha256: String,
    /// Production authentication policy; IR, RGB, active liveness, and fallback are mandatory.
    pub policy: AuthPolicy,
    /// FAR/FRR, lighting, demographic, latency, and hardware calibration report.
    pub evidence: ReviewedEvidence,
}

/// Passive PAD identities and hardware-calibrated live thresholds.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PassivePadCalibration {
    /// IR PAD manifest compatibility digest.
    pub infrared_compatibility_sha256: String,
    /// Minimum calibrated IR live probability.
    pub minimum_infrared_probability: f32,
    /// Visible PAD manifest compatibility digest.
    pub visible_compatibility_sha256: String,
    /// Minimum calibrated visible live probability.
    pub minimum_visible_probability: f32,
    /// Fusion PAD manifest compatibility digest.
    pub fusion_compatibility_sha256: String,
    /// Minimum calibrated fusion live probability.
    pub minimum_fusion_probability: f32,
    /// Print, replay, mask, occlusion, desynchronization, and modality-loss report.
    pub evidence: ReviewedEvidence,
}

/// Active-liveness thresholds bound to the exact landmark model.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ActiveLivenessCalibration {
    /// Full compatibility digest of the admitted landmark manifest.
    pub landmarks_compatibility_sha256: String,
    /// Bounded randomized challenge policy.
    pub challenge: ChallengeConfig,
    /// Hardware-specific temporal challenge calibration report.
    pub evidence: ReviewedEvidence,
}

/// Enrollment policy and isolated enrollment/replacement testing evidence.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EnrollmentCalibration {
    /// Bounded multi-sample enrollment policy.
    pub policy: EnrollmentConfig,
    /// Enrollment consistency, cancellation, replacement, and no-raw-image report.
    pub evidence: ReviewedEvidence,
}

/// Template directory and selected machine-key protection.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StorageConfig {
    /// Root-owned encrypted template directory.
    pub template_directory: PathBuf,
    /// Machine-key source. File keys remain diagnostic-only and fail production readiness.
    pub key: StorageKeyConfig,
    /// TPM loss, corruption, restart, and password-recovery test report.
    pub recovery_evidence: ReviewedEvidence,
}

/// Configurable key sources; readiness admits only the TPM-bound variant.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum StorageKeyConfig {
    /// TPM-sealed machine key blob.
    Tpm {
        /// Root-owned sealed public/private blob.
        sealed_key_blob: PathBuf,
    },
    /// Explicit development fallback that can never make production readiness pass.
    RootOnlyFile {
        /// Root-only raw machine-key file.
        key_file: PathBuf,
    },
}

/// Exact executable-bound authorization rules and review evidence.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AuthorizationConfig {
    /// Bounded exact-match caller rules.
    pub rules: Vec<AuthorizationRule>,
    /// Review of installed executable fingerprints and PAM/locker/Polkit service mapping.
    pub review_evidence: ReviewedEvidence,
}

/// Manager1 timing and packaging-policy review.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ManagementServiceConfig {
    /// Manager1 must be enabled in the eventual supervised composition.
    pub enabled: bool,
    /// Maximum lifetime of one enrollment operation.
    pub operation_duration_micros: u64,
    /// Reviewed D-Bus and `PolicyKit` packaging/activation report.
    pub policy_review_evidence: ReviewedEvidence,
}

/// Serializable daemon supervision resource bounds.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SupervisionPolicyConfig {
    /// Maximum delay between external shutdown checks.
    pub poll_interval_millis: u64,
    /// Grace period for all service threads to cooperate and join.
    pub shutdown_grace_millis: u64,
    /// Maximum independently supervised service tasks.
    pub max_services: usize,
}

impl SupervisionPolicyConfig {
    const fn supervisor(&self) -> SupervisorConfig {
        SupervisorConfig {
            poll_interval: Duration::from_millis(self.poll_interval_millis),
            shutdown_grace: Duration::from_millis(self.shutdown_grace_millis),
            max_services: self.max_services,
        }
    }
}

/// Authentication socket bounds and mandatory recovery behavior.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AuthenticationBoundaryConfig {
    /// Absolute path under an existing root-owned runtime directory.
    pub socket_path: PathBuf,
    /// Socket permissions, encoded as an integer (`384` for 0600 or `432` for 0660).
    pub socket_mode: u32,
    /// Maximum framed protocol payload.
    pub max_message_bytes: usize,
    /// Connected-stream read/write deadline.
    pub io_timeout_millis: u64,
    /// End-to-end authentication transaction deadline.
    pub transaction_duration_micros: u64,
    /// Must remain true for every PAM, locker, and Polkit integration.
    pub password_fallback_mandatory: bool,
    /// Isolated PAM/locker/Polkit failure and password-recovery test report.
    pub recovery_test_evidence: ReviewedEvidence,
}

/// Stable category in a readiness report.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ReadinessGateKind {
    /// Trusted, supported, and internally valid configuration.
    Configuration,
    /// Exact configured IR/RGB devices are present and uniquely resolved.
    Cameras,
    /// Trusted ONNX Runtime library is installed.
    Runtime,
    /// Every required role has a reviewed manifest and verified artifact.
    Models,
    /// Thresholds match exact model contracts and reviewed evidence.
    Calibration,
    /// TPM and encrypted-template storage policy is provisioned.
    Storage,
    /// Exact executable-bound service policy is reviewed.
    Authorization,
    /// Stable Manager1/PolicyKit lifecycle is configured and reviewed.
    Management,
    /// Local socket and password fallback/recovery invariants are proven.
    AuthenticationBoundary,
    /// Complete capture/inference/enrollment/service supervision is implemented.
    ServiceComposition,
}

/// One independently reportable production gate.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ReadinessGate {
    /// Stable gate identifier.
    pub gate: ReadinessGateKind,
    /// Whether current evidence proves the gate.
    pub ready: bool,
    /// Non-biometric diagnostic explanation.
    pub detail: String,
}

/// Itemized, fail-closed production readiness result.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ReadinessReport {
    /// Report schema for automation consumers.
    pub schema_version: u16,
    /// True only when every gate is proven.
    pub production_ready: bool,
    /// Stable ordered gate results.
    pub gates: Vec<ReadinessGate>,
}

impl ReadinessReport {
    fn new(gates: Vec<ReadinessGate>) -> Self {
        let production_ready = gates.iter().all(|gate| gate.ready);
        Self { schema_version: READINESS_REPORT_SCHEMA_VERSION, production_ready, gates }
    }

    /// Return stable names for all currently blocking gates.
    #[must_use]
    pub fn blocking_gates(&self) -> Vec<ReadinessGateKind> {
        self.gates.iter().filter(|gate| !gate.ready).map(|gate| gate.gate).collect()
    }
}

/// Invalid or untrusted production configuration.
#[derive(Debug, Error)]
pub enum ProductionConfigError {
    /// Configuration or a referenced file could not be read.
    #[error("unable to read production configuration or evidence: {0}")]
    Io(#[from] io::Error),
    /// Strict JSON decoding failed.
    #[error("invalid production configuration JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// A root-controlled file or directory failed ownership/permission policy.
    #[error("unsafe production path: {0}")]
    UnsafePath(PathBuf),
    /// A file exceeded its hard diagnostic size limit.
    #[error("production file {path} has invalid length {actual} (maximum {maximum})")]
    InvalidFileLength {
        /// Rejected path.
        path: PathBuf,
        /// Observed bytes.
        actual: u64,
        /// Maximum admitted bytes.
        maximum: u64,
    },
    /// The configuration violates a production invariant.
    #[error("invalid production configuration: {0}")]
    Invalid(String),
}

impl ProductionConfig {
    /// Parse strict JSON and validate all configuration-only invariants.
    ///
    /// # Errors
    ///
    /// Returns [`ProductionConfigError`] for malformed/unknown fields, unsupported schema,
    /// incomplete model roles, unsafe fallback policy, or invalid component bounds.
    pub fn from_slice(bytes: &[u8]) -> Result<Self, ProductionConfigError> {
        let config: Self = serde_json::from_slice(bytes)?;
        config.validate()?;
        Ok(config)
    }

    /// Validate configuration without probing hardware or referenced files.
    ///
    /// # Errors
    ///
    /// Returns [`ProductionConfigError`] when any production invariant is absent or contradictory.
    pub fn validate(&self) -> Result<(), ProductionConfigError> {
        if self.schema_version != PRODUCTION_CONFIG_SCHEMA_VERSION {
            return invalid(format!(
                "unsupported schema {}; this build supports {}",
                self.schema_version, PRODUCTION_CONFIG_SCHEMA_VERSION
            ));
        }
        validate_camera_selector(&self.cameras.infrared.selector, "infrared")?;
        validate_camera_selector(&self.cameras.visible.selector, "visible")?;
        if self.cameras.infrared.selector == self.cameras.visible.selector {
            return invalid("infrared and visible camera selectors must differ");
        }
        self.cameras.infrared.capture.validate(CaptureModality::Infrared).map_err(|error| {
            ProductionConfigError::Invalid(format!("invalid IR capture: {error}"))
        })?;
        self.cameras.visible.capture.validate(CaptureModality::Visible).map_err(|error| {
            ProductionConfigError::Invalid(format!("invalid visible capture: {error}"))
        })?;
        self.cameras.pairing.validate().map_err(|error| {
            ProductionConfigError::Invalid(format!("invalid camera pairing: {error}"))
        })?;
        if !self.runtime.library_path.is_absolute() {
            return invalid("runtime library path must be absolute");
        }
        self.runtime.runtime().validate().map_err(|error| {
            ProductionConfigError::Invalid(format!("invalid runtime policy: {error}"))
        })?;
        if self.models.len() != REQUIRED_MODEL_ROLES.len() {
            return invalid("the exact six-role production model suite is required");
        }
        let mut roles = Vec::with_capacity(self.models.len());
        for model in &self.models {
            if !REQUIRED_MODEL_ROLES.contains(&model.role) || roles.contains(&model.role) {
                return invalid("model roles must be unique and exactly match the required suite");
            }
            roles.push(model.role);
            require_absolute(&model.manifest_path, "model manifest")?;
            require_absolute(&model.artifact_path, "model artifact")?;
            validate_evidence_config(&model.review_evidence)?;
        }
        if !REQUIRED_MODEL_ROLES.iter().all(|role| roles.contains(role)) {
            return invalid("the production model suite is incomplete");
        }
        validate_calibration(&self.calibration)?;
        require_absolute(&self.storage.template_directory, "template directory")?;
        match &self.storage.key {
            StorageKeyConfig::Tpm { sealed_key_blob } => {
                require_absolute(sealed_key_blob, "TPM sealed-key blob")?;
            }
            StorageKeyConfig::RootOnlyFile { key_file } => {
                require_absolute(key_file, "file-key path")?;
            }
        }
        validate_evidence_config(&self.storage.recovery_evidence)?;
        let policy =
            AuthorizationPolicy::new(self.authorization.rules.clone()).map_err(|error| {
                ProductionConfigError::Invalid(format!("invalid authorization policy: {error}"))
            })?;
        drop(policy);
        validate_required_authorization_rules(&self.authorization.rules)?;
        validate_evidence_config(&self.authorization.review_evidence)?;
        if !self.management.enabled {
            return invalid("Manager1 must be enabled in production configuration");
        }
        ManagementConfig { operation_duration_micros: self.management.operation_duration_micros }
            .validate()
            .map_err(|error| {
                ProductionConfigError::Invalid(format!("invalid Manager1 policy: {error}"))
            })?;
        validate_evidence_config(&self.management.policy_review_evidence)?;
        self.supervision.supervisor().validate().map_err(|error| {
            ProductionConfigError::Invalid(format!("invalid service supervision policy: {error}"))
        })?;
        validate_authentication_boundary(&self.authentication_boundary)?;
        Ok(())
    }
}

/// Construct the real dual-camera, six-model authentication engine from a validated production
/// configuration.
///
/// This function opens V4L2 nodes and loads ONNX Runtime/model code. Callers must invoke it only
/// during explicit service startup, never from readiness inspection.
///
/// # Errors
///
/// Returns [`ProductionConfigError`] for discovery, device negotiation, manifest/model loading,
/// role mismatch, or calibration identity mismatch.
pub fn build_production_authentication_engine(
    config: &ProductionConfig,
) -> Result<ProductionAuthenticationEngine, ProductionConfigError> {
    config.validate()?;
    let model_identities = inspect_models(config).map_err(ProductionConfigError::Invalid)?;
    inspect_calibration(config, Some(&model_identities)).map_err(ProductionConfigError::Invalid)?;
    let devices = faceauth_camera::discover().map_err(|error| {
        ProductionConfigError::Invalid(format!("camera discovery failed: {error}"))
    })?;
    let resolved =
        faceauth_camera::resolve_pair(&devices, &config.cameras.selectors()).map_err(|error| {
            ProductionConfigError::Invalid(format!("camera selection failed: {error}"))
        })?;
    let infrared_device = V4l2CaptureDevice::open(
        &resolved.infrared.node,
        CaptureModality::Infrared,
        config.cameras.infrared.capture,
    )
    .map_err(|error| ProductionConfigError::Invalid(format!("IR camera open failed: {error}")))?;
    let visible_device = V4l2CaptureDevice::open(
        &resolved.visible.node,
        CaptureModality::Visible,
        config.cameras.visible.capture,
    )
    .map_err(|error| {
        ProductionConfigError::Invalid(format!("visible camera open failed: {error}"))
    })?;

    let runtime = config.runtime.runtime();
    let mut sessions = Vec::with_capacity(REQUIRED_MODEL_ROLES.len());
    for installation in &config.models {
        let manifest_path = trusted_regular_file(&installation.manifest_path, MAX_MANIFEST_BYTES)?;
        let bytes = fs::read(manifest_path)?;
        let manifest: ModelManifest = serde_json::from_slice(&bytes)?;
        if manifest.role != installation.role {
            return invalid("configured model role differs from its manifest role");
        }
        let session = OnnxSession::load(&runtime, &manifest, &installation.artifact_path).map_err(
            |error| ProductionConfigError::Invalid(format!("model load failed: {error}")),
        )?;
        sessions.push((installation.role, session));
    }
    let mut take = |role| {
        let index = sessions
            .iter()
            .position(|(candidate, _)| *candidate == role)
            .ok_or_else(|| ProductionConfigError::Invalid(format!("missing {role:?} session")))?;
        Ok::<_, ProductionConfigError>(sessions.swap_remove(index).1)
    };
    let detector = take(ModelRole::FaceDetector)?;
    let landmarks = take(ModelRole::FaceLandmarks)?;
    let embedding = take(ModelRole::FaceEmbedding)?;
    let passive_infrared = take(ModelRole::PassiveLivenessInfrared)?;
    let passive_visible = take(ModelRole::PassiveLivenessVisible)?;
    let passive_fusion = take(ModelRole::PassiveLivenessFusion)?;
    validate_loaded_calibration(
        config,
        &detector,
        &landmarks,
        &embedding,
        &passive_infrared,
        &passive_visible,
        &passive_fusion,
    )?;
    ProductionAuthenticationEngine::new(
        infrared_device,
        visible_device,
        detector,
        landmarks,
        embedding,
        passive_infrared,
        passive_visible,
        passive_fusion,
        config.cameras.pairing,
        config.calibration.quality.detection_policy(),
        config.calibration.quality.policy,
        config.calibration.active_liveness.challenge,
    )
    .map_err(|error| match error {
        ProductionAuthenticationEngineError::InvalidPolicy
        | ProductionAuthenticationEngineError::ModelRoleMismatch { .. } => {
            ProductionConfigError::Invalid(format!(
                "authentication engine assembly failed: {error}"
            ))
        }
    })
}

/// Construct and self-test the TPM-backed encrypted template store.
///
/// # Errors
///
/// Returns [`ProductionConfigError`] when storage evidence, TPM access, the sealed blob, or the
/// template directory fails production policy. Diagnostic file keys are always rejected.
pub fn build_production_template_store(
    config: &ProductionConfig,
) -> Result<EncryptedTemplateStore<TpmKeyProvider>, ProductionConfigError> {
    config.validate()?;
    inspect_storage(config).map_err(ProductionConfigError::Invalid)?;
    let StorageKeyConfig::Tpm { sealed_key_blob } = &config.storage.key else {
        return invalid("production template storage requires a TPM key");
    };
    let provider = TpmKeyProvider::new(sealed_key_blob);
    provider.self_test().map_err(|error| {
        ProductionConfigError::Invalid(format!("TPM key self-test failed: {error}"))
    })?;
    Ok(EncryptedTemplateStore::new(&config.storage.template_directory, provider))
}

/// Bound listener plus exact authorization, framing, and session policies for production.
pub struct ProductionAuthenticationBoundary {
    /// Root-owned Unix listener that never replaces an existing path.
    pub listener: SecureListener,
    /// Connected-peer framing and I/O limits.
    pub transport: TransportConfig,
    /// Exact executable-bound caller authorization policy.
    pub authorization: AuthorizationPolicy,
    /// Single-slot transaction deadline manager.
    pub sessions: SessionConfig,
}

/// Bind the root-owned authentication socket and construct its exact boundary policies.
///
/// # Errors
///
/// Returns [`ProductionConfigError`] when policy evidence, authorization, socket ownership/mode,
/// transport limits, or session limits are invalid. Existing socket paths are never removed.
pub fn build_production_authentication_boundary(
    config: &ProductionConfig,
) -> Result<ProductionAuthenticationBoundary, ProductionConfigError> {
    config.validate()?;
    inspect_authorization(config).map_err(ProductionConfigError::Invalid)?;
    inspect_authentication_boundary(config).map_err(ProductionConfigError::Invalid)?;
    let boundary = &config.authentication_boundary;
    let listener = SecureListener::bind_root_owned(&boundary.socket_path, boundary.socket_mode)
        .map_err(|error| {
            ProductionConfigError::Invalid(format!("authentication socket bind failed: {error}"))
        })?;
    let transport = TransportConfig {
        max_message_bytes: boundary.max_message_bytes,
        io_timeout: Duration::from_millis(boundary.io_timeout_millis),
    };
    let sessions =
        SessionConfig { transaction_duration_micros: boundary.transaction_duration_micros };
    let authorization =
        AuthorizationPolicy::new(config.authorization.rules.clone()).map_err(|error| {
            ProductionConfigError::Invalid(format!("authorization assembly failed: {error}"))
        })?;
    Ok(ProductionAuthenticationBoundary { listener, transport, authorization, sessions })
}

#[allow(clippy::too_many_arguments)]
fn validate_loaded_calibration(
    config: &ProductionConfig,
    detector: &OnnxSession,
    landmarks: &OnnxSession,
    embedding: &OnnxSession,
    passive_infrared: &OnnxSession,
    passive_visible: &OnnxSession,
    passive_fusion: &OnnxSession,
) -> Result<(), ProductionConfigError> {
    let calibration = &config.calibration;
    if detector.compatibility_sha256() != calibration.quality.detector_compatibility_sha256
        || landmarks.compatibility_sha256() != calibration.quality.landmarks_compatibility_sha256
        || landmarks.compatibility_sha256()
            != calibration.active_liveness.landmarks_compatibility_sha256
        || embedding.compatibility_sha256()
            != calibration.recognition.embedding_compatibility_sha256
        || passive_infrared.compatibility_sha256()
            != calibration.passive_pad.infrared_compatibility_sha256
        || passive_visible.compatibility_sha256()
            != calibration.passive_pad.visible_compatibility_sha256
        || passive_fusion.compatibility_sha256()
            != calibration.passive_pad.fusion_compatibility_sha256
    {
        return invalid("loaded model compatibility identities differ from calibration");
    }
    Ok(())
}

/// Load a bounded, root-controlled configuration file and validate it strictly.
///
/// # Errors
///
/// Returns [`ProductionConfigError`] for unsafe ownership, excessive size, read failure, strict
/// JSON decoding failure, or any invalid production invariant.
pub fn load_production_config(path: &Path) -> Result<ProductionConfig, ProductionConfigError> {
    let canonical = trusted_regular_file(path, MAX_CONFIG_BYTES)?;
    let bytes = fs::read(canonical)?;
    ProductionConfig::from_slice(&bytes)
}

/// Inspect current hardware, trusted files, policy evidence, and implementation readiness.
///
/// The inspection is read-only: it never loads ONNX Runtime, opens cameras, creates a TPM key,
/// binds a socket, claims a D-Bus name, or installs host policy.
#[must_use]
pub fn inspect_production_readiness(config: &ProductionConfig) -> ReadinessReport {
    if let Err(error) = config.validate() {
        return invalid_configuration_report(error.to_string());
    }

    let mut gates = vec![ready(
        ReadinessGateKind::Configuration,
        "configuration schema and invariants are valid",
    )];
    gates.push(result_gate(ReadinessGateKind::Cameras, inspect_cameras(config)));
    gates.push(result_gate(ReadinessGateKind::Runtime, inspect_runtime(config)));
    let model_result = inspect_models(config);
    let model_identities = model_result.as_ref().ok().cloned();
    gates.push(result_gate(
        ReadinessGateKind::Models,
        model_result.map(|_| {
            "all required manifests, artifacts, and model reviews are verified".to_owned()
        }),
    ));
    gates.push(result_gate(
        ReadinessGateKind::Calibration,
        inspect_calibration(config, model_identities.as_deref()),
    ));
    gates.push(result_gate(ReadinessGateKind::Storage, inspect_storage(config)));
    gates.push(result_gate(ReadinessGateKind::Authorization, inspect_authorization(config)));
    gates.push(result_gate(ReadinessGateKind::Management, inspect_management(config)));
    gates.push(result_gate(
        ReadinessGateKind::AuthenticationBoundary,
        inspect_authentication_boundary(config),
    ));
    gates.push(blocked(
        ReadinessGateKind::ServiceComposition,
        "full supervised capture/inference/authentication/enrollment composition is not implemented",
    ));
    ReadinessReport::new(gates)
}

/// Load a configured path and always return an itemized fail-closed report.
#[must_use]
pub fn readiness_from_config_path(path: &Path) -> ReadinessReport {
    match load_production_config(path) {
        Ok(config) => inspect_production_readiness(&config),
        Err(error) => invalid_configuration_report(error.to_string()),
    }
}

fn validate_camera_selector(
    selector: &CameraSelector,
    role: &'static str,
) -> Result<(), ProductionConfigError> {
    let valid_id = |value: &str| {
        value.len() == 4
            && value.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    };
    if !valid_id(&selector.vendor_id) || !valid_id(&selector.product_id) {
        return invalid(format!("{role} USB identifiers must be four lowercase hexadecimal bytes"));
    }
    if selector.serial.as_ref().is_none_or(String::is_empty)
        && selector.physical_path.as_ref().is_none_or(String::is_empty)
    {
        return invalid(format!("{role} camera requires a non-empty serial or physical path"));
    }
    Ok(())
}

fn validate_calibration(config: &CalibrationConfig) -> Result<(), ProductionConfigError> {
    if !valid_digest(&config.recognition.embedding_compatibility_sha256) {
        return invalid("recognition embedding compatibility digest is invalid");
    }
    config.recognition.policy.validate().map_err(|error| {
        ProductionConfigError::Invalid(format!("invalid recognition policy: {error}"))
    })?;
    if !config.recognition.policy.capture.require_visible
        || config.recognition.policy.liveness != LivenessLevel::PassiveAndActive
        || !config.recognition.policy.password_fallback
    {
        return invalid(
            "production recognition requires IR, visible, active liveness, and password fallback",
        );
    }
    validate_evidence_config(&config.recognition.evidence)?;
    if !valid_digest(&config.quality.detector_compatibility_sha256)
        || !valid_digest(&config.quality.landmarks_compatibility_sha256)
    {
        return invalid("quality calibration model compatibility digests are invalid");
    }
    config
        .quality
        .detection_policy()
        .validate()
        .map_err(|_| ProductionConfigError::Invalid("invalid detector calibration".to_owned()))?;
    config.quality.policy.validate().map_err(|error| {
        ProductionConfigError::Invalid(format!("invalid quality calibration: {error}"))
    })?;
    validate_evidence_config(&config.quality.evidence)?;
    let pad = &config.passive_pad;
    if !valid_digest(&pad.infrared_compatibility_sha256)
        || !valid_digest(&pad.visible_compatibility_sha256)
        || !valid_digest(&pad.fusion_compatibility_sha256)
        || !valid_probability(pad.minimum_infrared_probability)
        || !valid_probability(pad.minimum_visible_probability)
        || !valid_probability(pad.minimum_fusion_probability)
    {
        return invalid("passive PAD model identities or calibrated thresholds are invalid");
    }
    validate_evidence_config(&pad.evidence)?;
    if !valid_digest(&config.active_liveness.landmarks_compatibility_sha256) {
        return invalid("active-liveness landmark compatibility digest is invalid");
    }
    config.active_liveness.challenge.validate().map_err(|error| {
        ProductionConfigError::Invalid(format!("invalid active-liveness calibration: {error}"))
    })?;
    validate_evidence_config(&config.active_liveness.evidence)?;
    config.enrollment.policy.validate().map_err(|error| {
        ProductionConfigError::Invalid(format!("invalid enrollment calibration: {error}"))
    })?;
    validate_evidence_config(&config.enrollment.evidence)?;
    Ok(())
}

fn validate_required_authorization_rules(
    rules: &[AuthorizationRule],
) -> Result<(), ProductionConfigError> {
    for purpose in [
        AuthenticationPurpose::ScreenUnlock,
        AuthenticationPurpose::Login,
        AuthenticationPurpose::Polkit,
    ] {
        if !rules.iter().any(|rule| rule.purpose == purpose) {
            return invalid(format!("authorization policy has no {purpose:?} boundary"));
        }
    }
    if !rules.iter().any(|rule| {
        rule.service.as_str() == ENROLLMENT_SERVICE
            && rule.purpose == AuthenticationPurpose::Polkit
            && rule.caller == CallerRelation::RootOnly
    }) {
        return invalid("authorization policy lacks the exact root faceauth-enroll Polkit rule");
    }
    Ok(())
}

fn validate_authentication_boundary(
    config: &AuthenticationBoundaryConfig,
) -> Result<(), ProductionConfigError> {
    require_absolute(&config.socket_path, "authentication socket")?;
    if !matches!(config.socket_mode, 0o600 | 0o660) {
        return invalid("authentication socket mode must be 0600 or 0660");
    }
    TransportConfig {
        max_message_bytes: config.max_message_bytes,
        io_timeout: Duration::from_millis(config.io_timeout_millis),
    }
    .validate()
    .map_err(|error| {
        ProductionConfigError::Invalid(format!("invalid transport policy: {error}"))
    })?;
    SessionConfig { transaction_duration_micros: config.transaction_duration_micros }
        .validate()
        .map_err(|error| {
            ProductionConfigError::Invalid(format!("invalid session policy: {error}"))
        })?;
    if !config.password_fallback_mandatory {
        return invalid("password fallback must be mandatory");
    }
    validate_evidence_config(&config.recovery_test_evidence)
}

fn validate_evidence_config(evidence: &ReviewedEvidence) -> Result<(), ProductionConfigError> {
    require_absolute(&evidence.path, "review evidence")?;
    if !valid_digest(&evidence.sha256) {
        return invalid("review evidence SHA-256 is invalid");
    }
    Ok(())
}

fn inspect_cameras(config: &ProductionConfig) -> Result<String, String> {
    let devices = faceauth_camera::discover().map_err(|error| error.to_string())?;
    faceauth_camera::resolve_pair(&devices, &config.cameras.selectors())
        .map_err(|error| error.to_string())?;
    Ok("configured IR and visible selectors resolve uniquely to distinct capture nodes".to_owned())
}

fn inspect_runtime(config: &ProductionConfig) -> Result<String, String> {
    verify_runtime_installation(&config.runtime.runtime()).map_err(|error| error.to_string())?;
    Ok("ONNX Runtime library and ancestor permissions satisfy trusted-file policy".to_owned())
}

fn inspect_models(config: &ProductionConfig) -> Result<Vec<(ModelRole, String)>, String> {
    let runtime = config.runtime.runtime();
    let mut identities = Vec::with_capacity(config.models.len());
    for model in &config.models {
        verify_reviewed_evidence(&model.review_evidence)?;
        let manifest_path = trusted_regular_file(&model.manifest_path, MAX_MANIFEST_BYTES)
            .map_err(|error| error.to_string())?;
        let bytes = fs::read(manifest_path).map_err(|error| error.to_string())?;
        let manifest: ModelManifest =
            serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
        manifest.validate().map_err(|error| error.to_string())?;
        if manifest.role != model.role {
            return Err(format!(
                "manifest role {:?} does not match configured role {:?}",
                manifest.role, model.role
            ));
        }
        verify_model_installation(&runtime, &manifest, &model.artifact_path)
            .map_err(|error| error.to_string())?;
        let compatibility = manifest.compatibility_sha256().map_err(|error| error.to_string())?;
        identities.push((model.role, compatibility));
    }
    Ok(identities)
}

fn inspect_calibration(
    config: &ProductionConfig,
    identities: Option<&[(ModelRole, String)]>,
) -> Result<String, String> {
    let identities = identities.ok_or_else(|| "model identities are unavailable".to_owned())?;
    let expected = |role| {
        identities
            .iter()
            .find(|(candidate, _)| *candidate == role)
            .map(|(_, digest)| digest.as_str())
            .ok_or_else(|| format!("missing {role:?} model identity"))
    };
    let calibration = &config.calibration;
    if expected(ModelRole::FaceEmbedding)? != calibration.recognition.embedding_compatibility_sha256
        || expected(ModelRole::FaceDetector)? != calibration.quality.detector_compatibility_sha256
        || expected(ModelRole::FaceLandmarks)? != calibration.quality.landmarks_compatibility_sha256
        || expected(ModelRole::FaceLandmarks)?
            != calibration.active_liveness.landmarks_compatibility_sha256
        || expected(ModelRole::PassiveLivenessInfrared)?
            != calibration.passive_pad.infrared_compatibility_sha256
        || expected(ModelRole::PassiveLivenessVisible)?
            != calibration.passive_pad.visible_compatibility_sha256
        || expected(ModelRole::PassiveLivenessFusion)?
            != calibration.passive_pad.fusion_compatibility_sha256
    {
        return Err("calibration is not bound to the exact installed model contracts".to_owned());
    }
    for evidence in [
        &calibration.recognition.evidence,
        &calibration.quality.evidence,
        &calibration.passive_pad.evidence,
        &calibration.active_liveness.evidence,
        &calibration.enrollment.evidence,
    ] {
        verify_reviewed_evidence(evidence)?;
    }
    Ok("all thresholds are bound to installed model contracts and immutable review evidence"
        .to_owned())
}

fn inspect_storage(config: &ProductionConfig) -> Result<String, String> {
    trusted_directory(&config.storage.template_directory).map_err(|error| error.to_string())?;
    verify_reviewed_evidence(&config.storage.recovery_evidence)?;
    let StorageKeyConfig::Tpm { sealed_key_blob } = &config.storage.key else {
        return Err(
            "root-only file keys are diagnostic-only; production requires TPM binding".to_owned()
        );
    };
    trusted_character_device(Path::new("/dev/tpmrm0")).map_err(|error| error.to_string())?;
    trusted_regular_file(sealed_key_blob, MAX_CONFIG_BYTES).map_err(|error| error.to_string())?;
    Ok("TPM resource manager, sealed machine key, template directory, and recovery evidence are verified".to_owned())
}

fn inspect_authorization(config: &ProductionConfig) -> Result<String, String> {
    verify_reviewed_evidence(&config.authorization.review_evidence)?;
    Ok("exact executable-bound screen-unlock, login, and Polkit rules are configured and reviewed"
        .to_owned())
}

fn inspect_management(config: &ProductionConfig) -> Result<String, String> {
    verify_reviewed_evidence(&config.management.policy_review_evidence)?;
    Ok(format!(
        "stable {MANAGER_BUS_NAME} at {MANAGER_OBJECT_PATH} and {ENROLLMENT_POLKIT_ACTION} policy are reviewed"
    ))
}

fn inspect_authentication_boundary(config: &ProductionConfig) -> Result<String, String> {
    let parent = config
        .authentication_boundary
        .socket_path
        .parent()
        .ok_or_else(|| "authentication socket has no parent".to_owned())?;
    trusted_directory(parent).map_err(|error| error.to_string())?;
    match fs::symlink_metadata(&config.authentication_boundary.socket_path) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Ok(_) => {
            return Err(
                "authentication socket path already exists; stale entries are never replaced"
                    .to_owned(),
            );
        }
        Err(error) => return Err(error.to_string()),
    }
    verify_reviewed_evidence(&config.authentication_boundary.recovery_test_evidence)?;
    Ok("root-owned socket boundary and isolated password-recovery evidence are verified".to_owned())
}

fn verify_reviewed_evidence(evidence: &ReviewedEvidence) -> Result<(), String> {
    let path = trusted_regular_file(&evidence.path, MAX_REVIEW_EVIDENCE_BYTES)
        .map_err(|error| error.to_string())?;
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    let actual = hex_lower(&hasher.finalize());
    if actual != evidence.sha256 {
        return Err(format!("review evidence digest mismatch for {}", evidence.path.display()));
    }
    Ok(())
}

fn trusted_regular_file(path: &Path, maximum: u64) -> Result<PathBuf, ProductionConfigError> {
    let canonical = fs::canonicalize(path)?;
    let metadata = fs::metadata(&canonical)?;
    if !metadata.is_file() || metadata.uid() != 0 || metadata.permissions().mode() & 0o022 != 0 {
        return Err(ProductionConfigError::UnsafePath(canonical));
    }
    if metadata.len() == 0 || metadata.len() > maximum {
        return Err(ProductionConfigError::InvalidFileLength {
            path: canonical,
            actual: metadata.len(),
            maximum,
        });
    }
    verify_ancestors(&canonical)?;
    Ok(canonical)
}

fn trusted_directory(path: &Path) -> Result<(), ProductionConfigError> {
    let canonical = fs::canonicalize(path)?;
    let metadata = fs::metadata(&canonical)?;
    if !metadata.is_dir() || metadata.uid() != 0 || metadata.permissions().mode() & 0o022 != 0 {
        return Err(ProductionConfigError::UnsafePath(canonical));
    }
    verify_ancestors(&canonical)
}

fn trusted_character_device(path: &Path) -> Result<(), ProductionConfigError> {
    let canonical = fs::canonicalize(path)?;
    let metadata = fs::metadata(&canonical)?;
    if !metadata.file_type().is_char_device()
        || metadata.uid() != 0
        || metadata.permissions().mode() & 0o002 != 0
    {
        return Err(ProductionConfigError::UnsafePath(canonical));
    }
    verify_ancestors(&canonical)
}

fn verify_ancestors(path: &Path) -> Result<(), ProductionConfigError> {
    for ancestor in path.ancestors().skip(1) {
        let metadata = fs::metadata(ancestor)?;
        if !metadata.is_dir() || metadata.uid() != 0 || metadata.permissions().mode() & 0o022 != 0 {
            return Err(ProductionConfigError::UnsafePath(ancestor.to_owned()));
        }
    }
    Ok(())
}

fn require_absolute(path: &Path, label: &str) -> Result<(), ProductionConfigError> {
    if path.is_absolute() { Ok(()) } else { invalid(format!("{label} path must be absolute")) }
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn valid_probability(value: f32) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn invalid<T>(message: impl Into<String>) -> Result<T, ProductionConfigError> {
    Err(ProductionConfigError::Invalid(message.into()))
}

fn ready(gate: ReadinessGateKind, detail: impl Into<String>) -> ReadinessGate {
    ReadinessGate { gate, ready: true, detail: detail.into() }
}

fn blocked(gate: ReadinessGateKind, detail: impl Into<String>) -> ReadinessGate {
    ReadinessGate { gate, ready: false, detail: detail.into() }
}

fn result_gate(gate: ReadinessGateKind, result: Result<String, String>) -> ReadinessGate {
    match result {
        Ok(detail) => ready(gate, detail),
        Err(detail) => blocked(gate, detail),
    }
}

fn invalid_configuration_report(detail: String) -> ReadinessReport {
    let mut gates = vec![blocked(ReadinessGateKind::Configuration, detail)];
    for gate in [
        ReadinessGateKind::Cameras,
        ReadinessGateKind::Runtime,
        ReadinessGateKind::Models,
        ReadinessGateKind::Calibration,
        ReadinessGateKind::Storage,
        ReadinessGateKind::Authorization,
        ReadinessGateKind::Management,
        ReadinessGateKind::AuthenticationBoundary,
    ] {
        gates.push(blocked(gate, "not evaluated because configuration is not trusted and valid"));
    }
    gates.push(blocked(
        ReadinessGateKind::ServiceComposition,
        "full supervised capture/inference/authentication/enrollment composition is not implemented",
    ));
    ReadinessReport::new(gates)
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

#[cfg(test)]
mod tests {
    use faceauth_authz::{CallerRelation, ExecutableFingerprint};
    use faceauth_core::CapturePolicy;
    use faceauth_protocol::ServiceName;

    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn evidence(name: &str) -> ReviewedEvidence {
        ReviewedEvidence {
            path: PathBuf::from(format!("/var/lib/faceauth/reviews/{name}")),
            sha256: digest('a'),
        }
    }

    fn rule(
        service: &str,
        purpose: AuthenticationPurpose,
        caller: CallerRelation,
    ) -> Result<AuthorizationRule, Box<dyn std::error::Error>> {
        Ok(AuthorizationRule {
            service: ServiceName::parse(service)?,
            purpose,
            caller,
            executables: vec![ExecutableFingerprint { device: 1, inode: 2 }],
        })
    }

    fn calibration() -> CalibrationConfig {
        CalibrationConfig {
            recognition: RecognitionCalibration {
                embedding_compatibility_sha256: digest('b'),
                policy: AuthPolicy {
                    capture: CapturePolicy {
                        require_infrared: true,
                        require_visible: true,
                        max_pair_skew_micros: 100_000,
                    },
                    liveness: LivenessLevel::PassiveAndActive,
                    minimum_similarity: 0.7,
                    minimum_quality: 0.7,
                    password_fallback: true,
                },
                evidence: evidence("recognition.json"),
            },
            quality: QualityCalibration {
                detector_compatibility_sha256: digest('1'),
                landmarks_compatibility_sha256: digest('f'),
                minimum_detection_confidence: 0.8,
                maximum_detection_iou: 0.4,
                maximum_detected_faces: 4,
                policy: QualityConfig::engineering_baseline(),
                evidence: evidence("quality.json"),
            },
            passive_pad: PassivePadCalibration {
                infrared_compatibility_sha256: digest('c'),
                minimum_infrared_probability: 0.8,
                visible_compatibility_sha256: digest('d'),
                minimum_visible_probability: 0.8,
                fusion_compatibility_sha256: digest('e'),
                minimum_fusion_probability: 0.8,
                evidence: evidence("pad.json"),
            },
            active_liveness: ActiveLivenessCalibration {
                landmarks_compatibility_sha256: digest('f'),
                challenge: ChallengeConfig {
                    max_pair_skew_micros: 100_000,
                    duration_micros: 10_000_000,
                    max_observations: 60,
                    open_eye_threshold: 0.7,
                    closed_eye_threshold: 0.3,
                    neutral_yaw_degrees: 8.0,
                    turn_yaw_degrees: 20.0,
                    action_consecutive_observations: 3,
                    minimum_action_duration_micros: 100_000,
                    recovery_consecutive_observations: 3,
                },
                evidence: evidence("active.json"),
            },
            enrollment: EnrollmentCalibration {
                policy: EnrollmentConfig {
                    duration_micros: 30_000_000,
                    minimum_samples: 5,
                    maximum_samples: 8,
                    minimum_quality: 0.7,
                    minimum_sample_similarity: 0.75,
                    minimum_sample_interval_micros: 250_000,
                    minimum_yaw_span_degrees: 15.0,
                },
                evidence: evidence("enrollment.json"),
            },
        }
    }

    fn valid_config() -> Result<ProductionConfig, Box<dyn std::error::Error>> {
        let selector = |product_id: &str, physical_path: &str| CameraSelector {
            vendor_id: "04f2".to_owned(),
            product_id: product_id.to_owned(),
            serial: None,
            physical_path: Some(physical_path.to_owned()),
        };
        let models = REQUIRED_MODEL_ROLES
            .iter()
            .enumerate()
            .map(|(index, role)| ModelInstallation {
                role: *role,
                manifest_path: PathBuf::from(format!("/usr/share/faceauth/models/{index}.json")),
                artifact_path: PathBuf::from(format!("/usr/share/faceauth/models/{index}.onnx")),
                review_evidence: evidence(&format!("model-{index}.json")),
            })
            .collect();
        Ok(ProductionConfig {
            schema_version: PRODUCTION_CONFIG_SCHEMA_VERSION,
            cameras: ProductionCameraConfig {
                infrared: CameraStreamConfig {
                    selector: selector("b769", "pci-ir"),
                    capture: CaptureSpec {
                        width: 640,
                        height: 480,
                        pixel_format: faceauth_capture::PixelFormat::Gray8,
                        frames_per_second: 30,
                        buffer_count: 4,
                        warmup_frames: 3,
                        frame_timeout_millis: 250,
                        max_frame_bytes: 640 * 480,
                    },
                },
                visible: CameraStreamConfig {
                    selector: selector("b768", "pci-rgb"),
                    capture: CaptureSpec {
                        width: 640,
                        height: 480,
                        pixel_format: faceauth_capture::PixelFormat::Yuyv,
                        frames_per_second: 30,
                        buffer_count: 4,
                        warmup_frames: 3,
                        frame_timeout_millis: 250,
                        max_frame_bytes: 640 * 480 * 2,
                    },
                },
                pairing: PairingPolicy { max_skew_micros: 100_000, max_replacements: 4 },
            },
            runtime: RuntimePolicyConfig {
                library_path: PathBuf::from("/usr/lib/libonnxruntime.so"),
                intra_threads: 2,
                inter_threads: 1,
                max_model_bytes: 512 * 1024 * 1024,
                max_run_millis: 1_000,
            },
            models,
            calibration: calibration(),
            storage: StorageConfig {
                template_directory: PathBuf::from("/var/lib/faceauth/templates"),
                key: StorageKeyConfig::Tpm {
                    sealed_key_blob: PathBuf::from("/var/lib/faceauth/machine-key.tpm"),
                },
                recovery_evidence: evidence("storage-recovery.json"),
            },
            authorization: AuthorizationConfig {
                rules: vec![
                    rule(
                        "faceauth-locker",
                        AuthenticationPurpose::ScreenUnlock,
                        CallerRelation::RootOnly,
                    )?,
                    rule("faceauth-login", AuthenticationPurpose::Login, CallerRelation::RootOnly)?,
                    rule(
                        ENROLLMENT_SERVICE,
                        AuthenticationPurpose::Polkit,
                        CallerRelation::RootOnly,
                    )?,
                ],
                review_evidence: evidence("authorization.json"),
            },
            management: ManagementServiceConfig {
                enabled: true,
                operation_duration_micros: 120_000_000,
                policy_review_evidence: evidence("management-policy.json"),
            },
            supervision: SupervisionPolicyConfig {
                poll_interval_millis: 25,
                shutdown_grace_millis: 5_000,
                max_services: 4,
            },
            authentication_boundary: AuthenticationBoundaryConfig {
                socket_path: PathBuf::from("/run/faceauth/auth.sock"),
                socket_mode: 0o660,
                max_message_bytes: 16 * 1024,
                io_timeout_millis: 2_000,
                transaction_duration_micros: 15_000_000,
                password_fallback_mandatory: true,
                recovery_test_evidence: evidence("password-recovery.json"),
            },
        })
    }

    #[test]
    fn complete_configuration_is_structurally_valid() -> Result<(), Box<dyn std::error::Error>> {
        valid_config()?.validate()?;
        Ok(())
    }

    #[test]
    fn packaged_example_matches_the_current_strict_schema() -> Result<(), Box<dyn std::error::Error>>
    {
        ProductionConfig::from_slice(include_bytes!(
            "../../../contrib/config/faceauth.json.example"
        ))?;
        Ok(())
    }

    #[test]
    fn strict_json_rejects_unknown_top_level_and_nested_fields()
    -> Result<(), Box<dyn std::error::Error>> {
        let value = serde_json::to_value(valid_config()?)?;
        let mut top_level = value.clone();
        top_level
            .as_object_mut()
            .ok_or("object expected")?
            .insert("future".to_owned(), serde_json::json!(true));
        assert!(ProductionConfig::from_slice(&serde_json::to_vec(&top_level)?).is_err());

        let mut nested = value;
        nested["cameras"]["infrared"]["future"] = serde_json::json!(true);
        assert!(ProductionConfig::from_slice(&serde_json::to_vec(&nested)?).is_err());
        Ok(())
    }

    #[test]
    fn incomplete_model_suite_and_disabled_fallback_are_rejected()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut missing_model = valid_config()?;
        missing_model.models.pop();
        assert!(missing_model.validate().is_err());

        let mut no_fallback = valid_config()?;
        no_fallback.authentication_boundary.password_fallback_mandatory = false;
        assert!(no_fallback.validate().is_err());
        Ok(())
    }

    #[test]
    fn no_partial_configuration_can_report_production_ready()
    -> Result<(), Box<dyn std::error::Error>> {
        let report = inspect_production_readiness(&valid_config()?);
        assert!(!report.production_ready);
        assert_eq!(report.gates.len(), 10);
        assert!(
            report
                .gates
                .iter()
                .any(|gate| { gate.gate == ReadinessGateKind::ServiceComposition && !gate.ready })
        );
        Ok(())
    }

    #[test]
    fn hard_resource_ceilings_are_not_weakened_by_config() -> Result<(), Box<dyn std::error::Error>>
    {
        let mut config = valid_config()?;
        config.runtime.max_model_bytes = faceauth_inference::ABSOLUTE_MAX_MODEL_BYTES + 1;
        assert!(config.validate().is_err());
        config.runtime.max_model_bytes = 512 * 1024 * 1024;
        config.runtime.max_run_millis = faceauth_inference::ABSOLUTE_MAX_RUN_MILLIS + 1;
        assert!(config.validate().is_err());
        config.runtime.max_run_millis = 1_000;
        config.authentication_boundary.max_message_bytes =
            faceauth_transport::ABSOLUTE_MAX_MESSAGE_BYTES + 1;
        assert!(config.validate().is_err());
        config.authentication_boundary.max_message_bytes = 16 * 1024;
        config.supervision.shutdown_grace_millis = 9;
        assert!(config.validate().is_err());
        Ok(())
    }
}
