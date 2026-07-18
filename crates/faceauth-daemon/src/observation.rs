//! Shared production IR/RGB capture and six-model observation pipeline.

use std::{
    ops::ControlFlow,
    sync::{Arc, Mutex},
};

use faceauth_capture::{
    CapturedFrame, PairedFrames, PairingPolicy, PixelFormat, V4l2CaptureDevice,
    capture_pair_cancellable,
};
use faceauth_core::CapturePair;
use faceauth_inference::{
    FaceEmbedding, FaceRegion, FacialLandmarks, ImageFacialLandmarks, ImageFormat, ImageView,
    InferenceError, InputTensor, OnnxSession, PassiveLivenessScore,
};
use faceauth_liveness::{
    ChallengeAction, ChallengeConfig, ChallengeObservation, ChallengeProgress, ChallengeSession,
};
use faceauth_model::ModelRole;
use faceauth_quality::{
    FaceGeometry, Gray8View, ImageView as QualityImageView, QualityConfig, QualityError, YuyvView,
    assess_cancellable,
};
use thiserror::Error;

use crate::DetectionPolicy;

const MAX_DERIVED_OBSERVATION_ATTEMPTS: usize = 32;

/// Shared single owner of the real cameras and mutable model sessions.
pub type SharedProductionObservationPipeline = Arc<Mutex<ProductionObservationPipeline>>;

/// Closed, desktop-independent progress emitted by the shared vision pipeline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObservationProgress {
    PositionFace,
    HoldStill,
    Blink,
    TurnLeft,
    TurnRight,
    ReturnToCenter,
    Processing,
}

/// One fully derived, model-bound observation.
///
/// Raw images, tensors, detector boxes, and landmarks are deliberately absent. The value remains
/// inside the owning worker and is immediately converted into authentication or enrollment
/// evidence.
pub struct DerivedBiometricObservation {
    pub(crate) timing: CapturePair,
    pub(crate) quality: f32,
    pub(crate) yaw_degrees: f32,
    pub(crate) embedding: FaceEmbedding,
    pub(crate) passive_liveness: Vec<PassiveLivenessScore>,
    pub(crate) completed_at_micros: u64,
}

/// Real V4L2/ONNX pipeline shared by authentication and enrollment adapters.
pub struct ProductionObservationPipeline {
    infrared_device: V4l2CaptureDevice,
    visible_device: V4l2CaptureDevice,
    detector: OnnxSession,
    landmarks: OnnxSession,
    embedding: OnnxSession,
    passive_infrared: OnnxSession,
    passive_visible: OnnxSession,
    passive_fusion: OnnxSession,
    pairing: PairingPolicy,
    detection: DetectionPolicy,
    quality: QualityConfig,
}

impl ProductionObservationPipeline {
    /// Assemble the unique camera/model owner from already admitted resources.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        infrared_device: V4l2CaptureDevice,
        visible_device: V4l2CaptureDevice,
        detector: OnnxSession,
        landmarks: OnnxSession,
        embedding: OnnxSession,
        passive_infrared: OnnxSession,
        passive_visible: OnnxSession,
        passive_fusion: OnnxSession,
        pairing: PairingPolicy,
        detection: DetectionPolicy,
        quality: QualityConfig,
    ) -> Result<Self, ProductionObservationPipelineError> {
        pairing.validate().map_err(|_| ProductionObservationPipelineError::InvalidPolicy)?;
        detection.validate().map_err(|_| ProductionObservationPipelineError::InvalidPolicy)?;
        quality.validate().map_err(|_| ProductionObservationPipelineError::InvalidPolicy)?;
        for (session, expected) in [
            (&detector, ModelRole::FaceDetector),
            (&landmarks, ModelRole::FaceLandmarks),
            (&embedding, ModelRole::FaceEmbedding),
            (&passive_infrared, ModelRole::PassiveLivenessInfrared),
            (&passive_visible, ModelRole::PassiveLivenessVisible),
            (&passive_fusion, ModelRole::PassiveLivenessFusion),
        ] {
            if session.role() != expected {
                return Err(ProductionObservationPipelineError::ModelRoleMismatch {
                    expected,
                    actual: session.role(),
                });
            }
        }
        Ok(Self {
            infrared_device,
            visible_device,
            detector,
            landmarks,
            embedding,
            passive_infrared,
            passive_visible,
            passive_fusion,
            pairing,
            detection,
            quality,
        })
    }

    /// Capture one operation-bound active challenge followed by fresh derived observations.
    ///
    /// The callback decides when enough derived observations have been collected. All frames and
    /// intermediate tensors remain on this stack and are dropped before the result leaves.
    pub(crate) fn run_session<T>(
        &mut self,
        challenge_config: ChallengeConfig,
        cancelled: &mut dyn FnMut() -> bool,
        progress: &mut dyn FnMut(ObservationProgress) -> Result<(), ObservationFailure>,
        observe: &mut dyn FnMut(
            DerivedBiometricObservation,
        ) -> Result<ControlFlow<T>, ObservationFailure>,
    ) -> Result<T, ObservationFailure> {
        self.pairing.validate().map_err(|_| ObservationFailure::Capture)?;
        self.detection.validate().map_err(|_| ObservationFailure::InvalidEvidence)?;
        self.quality.validate().map_err(|_| ObservationFailure::InvalidEvidence)?;
        challenge_config.validate().map_err(|_| ObservationFailure::Liveness)?;
        ensure_active(cancelled)?;

        let Self {
            infrared_device,
            visible_device,
            detector,
            landmarks,
            embedding,
            passive_infrared,
            passive_visible,
            passive_fusion,
            pairing,
            detection,
            quality,
        } = self;
        let mut infrared = infrared_device.stream().map_err(|_| capture_failure(cancelled))?;
        let mut visible = visible_device.stream().map_err(|_| capture_failure(cancelled))?;
        progress(ObservationProgress::PositionFace)?;

        let mut challenge: Option<ChallengeSession> = None;
        let mut challenge_passed = false;
        let mut last_progress = None;
        let mut previous_pair: Option<FreshPairIdentity> = None;
        let mut derived_attempts = 0_usize;
        let maximum_capture_attempts =
            usize::from(challenge_config.max_observations) * 2 + MAX_DERIVED_OBSERVATION_ATTEMPTS;
        let mut capture_attempts = 0_usize;
        loop {
            ensure_active(cancelled)?;
            if capture_attempts >= maximum_capture_attempts {
                return Err(ObservationFailure::AttemptLimit);
            }
            let pair =
                capture_pair_cancellable(&mut infrared, &mut visible, *pairing, &mut *cancelled)
                    .map_err(|_| capture_failure(cancelled))?;
            capture_attempts += 1;
            let identity = FreshPairIdentity::from_pair(&pair);
            if previous_pair.is_some_and(|previous| !identity.is_fresh_after(previous)) {
                return Err(ObservationFailure::InvalidEvidence);
            }
            previous_pair = Some(identity);

            let landmarks = match process_pair(&pair, detector, landmarks, *detection, cancelled) {
                Ok(landmarks) => landmarks,
                Err(AttemptFailure::RetryPosition) => {
                    progress(ObservationProgress::PositionFace)?;
                    continue;
                }
                Err(AttemptFailure::Fatal(error)) => return Err(error),
            };
            let timing = pair.timing();
            let observation_time = identity.timestamp_micros;

            if !challenge_passed {
                let state = advance_challenge(
                    &mut challenge,
                    challenge_config,
                    observation_time,
                    timing,
                    &pair,
                    &landmarks,
                    cancelled,
                )?;
                if state != ChallengeProgress::Passed {
                    let public = observation_progress(state);
                    if last_progress != Some(public) {
                        progress(public)?;
                        last_progress = Some(public);
                    }
                    continue;
                }
                challenge_passed = true;
            }

            progress(ObservationProgress::Processing)?;
            let derived = derive_observation(
                &pair,
                &landmarks,
                embedding,
                passive_infrared,
                passive_visible,
                passive_fusion,
                *quality,
                cancelled,
            )?;
            derived_attempts += 1;
            match observe(derived)? {
                ControlFlow::Break(result) => return Ok(result),
                ControlFlow::Continue(())
                    if derived_attempts < MAX_DERIVED_OBSERVATION_ATTEMPTS =>
                {
                    progress(ObservationProgress::HoldStill)?;
                }
                ControlFlow::Continue(()) => return Err(ObservationFailure::AttemptLimit),
            }
        }
    }
}

fn advance_challenge(
    challenge: &mut Option<ChallengeSession>,
    config: ChallengeConfig,
    observation_time: u64,
    timing: CapturePair,
    pair: &PairedFrames,
    landmarks: &FacialLandmarks,
    cancelled: &mut dyn FnMut() -> bool,
) -> Result<ChallengeProgress, ObservationFailure> {
    let active = match challenge.as_mut() {
        Some(active) => active,
        None => challenge.insert(
            ChallengeSession::begin(config, observation_time)
                .map_err(|_| ObservationFailure::Liveness)?,
        ),
    };
    let measurements = landmarks.measurements();
    active
        .observe_cancellable(
            ChallengeObservation {
                timing,
                infrared_sequence: pair.infrared.summary().sequence,
                visible_sequence: pair.visible.summary().sequence,
                face_count: 1,
                eye_openness: measurements.eye_openness(),
                yaw_degrees: measurements.yaw_degrees(),
            },
            &mut *cancelled,
        )
        .map_err(|_| liveness_failure(cancelled))
}

#[derive(Clone, Copy)]
struct FreshPairIdentity {
    infrared_sequence: u32,
    visible_sequence: u32,
    timestamp_micros: u64,
}

impl FreshPairIdentity {
    fn from_pair(pair: &PairedFrames) -> Self {
        let timing = pair.timing();
        Self {
            infrared_sequence: pair.infrared.summary().sequence,
            visible_sequence: pair.visible.summary().sequence,
            timestamp_micros: timing
                .visible_timestamp_micros
                .map_or(timing.infrared_timestamp_micros, |visible| {
                    timing.infrared_timestamp_micros.max(visible)
                }),
        }
    }

    const fn is_fresh_after(self, previous: Self) -> bool {
        self.infrared_sequence != previous.infrared_sequence
            && self.visible_sequence != previous.visible_sequence
            && self.timestamp_micros > previous.timestamp_micros
    }
}

enum AttemptFailure {
    RetryPosition,
    Fatal(ObservationFailure),
}

fn process_pair(
    pair: &PairedFrames,
    detector: &mut OnnxSession,
    landmarks_session: &mut OnnxSession,
    detection: DetectionPolicy,
    cancelled: &mut dyn FnMut() -> bool,
) -> Result<FacialLandmarks, AttemptFailure> {
    let detector_input = preprocess_frame(detector, &pair.visible, cancelled)
        .map_err(|_| AttemptFailure::Fatal(inference_failure(cancelled)))?;
    ensure_active(cancelled).map_err(AttemptFailure::Fatal)?;
    let detector_outputs = detector
        .run(&detector_input)
        .map_err(|_| AttemptFailure::Fatal(inference_failure(cancelled)))?;
    ensure_active(cancelled).map_err(AttemptFailure::Fatal)?;
    let detections = detector
        .extract_face_detections(&detector_outputs, detection.minimum_confidence)
        .and_then(|faces| faces.suppress_overlaps(detection.maximum_iou, detection.maximum_faces))
        .map_err(|_| AttemptFailure::Fatal(inference_failure(cancelled)))?;
    let face = match detections.require_single_face() {
        Ok(face) => face,
        Err(InferenceError::NoFaceDetected | InferenceError::MultipleFacesDetected { .. }) => {
            return Err(AttemptFailure::RetryPosition);
        }
        Err(_) => return Err(AttemptFailure::Fatal(inference_failure(cancelled))),
    };
    let region = match landmarks_session.derive_face_region(face.bounds()) {
        Ok(region) => region,
        Err(InferenceError::FaceRegionOutsideImage) => {
            return Err(AttemptFailure::RetryPosition);
        }
        Err(_) => return Err(AttemptFailure::Fatal(inference_failure(cancelled))),
    };
    let visible_image = inference_frame_view(&pair.visible)
        .map_err(|_| AttemptFailure::Fatal(inference_failure(cancelled)))?;
    let landmark_input =
        preprocess_face_region(landmarks_session, visible_image, &region, cancelled).map_err(
            |error| match error {
                InferenceError::FaceRegionOutsideImage => AttemptFailure::RetryPosition,
                _ => AttemptFailure::Fatal(inference_failure(cancelled)),
            },
        )?;
    ensure_active(cancelled).map_err(AttemptFailure::Fatal)?;
    let outputs = landmarks_session
        .run(&landmark_input)
        .map_err(|_| AttemptFailure::Fatal(inference_failure(cancelled)))?;
    ensure_active(cancelled).map_err(AttemptFailure::Fatal)?;
    landmarks_session
        .extract_face_landmarks(&outputs, &region)
        .map_err(|_| AttemptFailure::Fatal(inference_failure(cancelled)))
}

#[allow(clippy::too_many_arguments)]
fn derive_observation(
    pair: &PairedFrames,
    landmarks: &FacialLandmarks,
    embedding_session: &mut OnnxSession,
    passive_infrared: &mut OnnxSession,
    passive_visible: &mut OnnxSession,
    passive_fusion: &mut OnnxSession,
    quality_config: QualityConfig,
    cancelled: &mut dyn FnMut() -> bool,
) -> Result<DerivedBiometricObservation, ObservationFailure> {
    let quality =
        assess_landmark_frame_quality(&pair.visible, landmarks, quality_config, cancelled)
            .map_err(|_| invalid_evidence_failure(cancelled))?;
    let image_landmarks = landmarks.map_to_image().map_err(|_| inference_failure(cancelled))?;
    let visible_image =
        inference_frame_view(&pair.visible).map_err(|_| inference_failure(cancelled))?;
    let embedding_input =
        preprocess_aligned(embedding_session, visible_image, &image_landmarks, cancelled)
            .map_err(|_| inference_failure(cancelled))?;
    ensure_active(cancelled)?;
    let embedding_outputs =
        embedding_session.run(&embedding_input).map_err(|_| inference_failure(cancelled))?;
    ensure_active(cancelled)?;
    let embedding = embedding_session
        .extract_embedding(&embedding_outputs)
        .map_err(|_| inference_failure(cancelled))?;

    let infrared_input = preprocess_frame(passive_infrared, &pair.infrared, cancelled)
        .map_err(|_| inference_failure(cancelled))?;
    let visible_input = preprocess_frame(passive_visible, &pair.visible, cancelled)
        .map_err(|_| inference_failure(cancelled))?;
    let fusion_infrared =
        preprocess_named_frame(passive_fusion, "infrared", &pair.infrared, cancelled)
            .map_err(|_| inference_failure(cancelled))?;
    let fusion_visible =
        preprocess_named_frame(passive_fusion, "visible", &pair.visible, cancelled)
            .map_err(|_| inference_failure(cancelled))?;
    ensure_active(cancelled)?;
    let infrared_outputs =
        passive_infrared.run(&infrared_input).map_err(|_| inference_failure(cancelled))?;
    ensure_active(cancelled)?;
    let visible_outputs =
        passive_visible.run(&visible_input).map_err(|_| inference_failure(cancelled))?;
    ensure_active(cancelled)?;
    let fusion_outputs = passive_fusion
        .run_named(&[("infrared", &fusion_infrared), ("visible", &fusion_visible)])
        .map_err(|_| inference_failure(cancelled))?;
    ensure_active(cancelled)?;
    let passive_liveness = vec![
        passive_infrared
            .extract_passive_liveness(&infrared_outputs)
            .map_err(|_| inference_failure(cancelled))?,
        passive_visible
            .extract_passive_liveness(&visible_outputs)
            .map_err(|_| inference_failure(cancelled))?,
        passive_fusion
            .extract_passive_liveness(&fusion_outputs)
            .map_err(|_| inference_failure(cancelled))?,
    ];
    let identity = FreshPairIdentity::from_pair(pair);
    Ok(DerivedBiometricObservation {
        timing: pair.timing(),
        quality: quality.aggregate,
        yaw_degrees: landmarks.measurements().yaw_degrees(),
        embedding,
        passive_liveness,
        completed_at_micros: identity.timestamp_micros,
    })
}

fn preprocess_frame(
    session: &OnnxSession,
    frame: &CapturedFrame,
    cancelled: &mut dyn FnMut() -> bool,
) -> Result<InputTensor, InferenceError> {
    session.preprocess_cancellable(inference_frame_view(frame)?, &mut *cancelled)
}

fn preprocess_named_frame(
    session: &OnnxSession,
    name: &str,
    frame: &CapturedFrame,
    cancelled: &mut dyn FnMut() -> bool,
) -> Result<InputTensor, InferenceError> {
    session.preprocess_named_cancellable(name, inference_frame_view(frame)?, &mut *cancelled)
}

fn preprocess_aligned(
    session: &OnnxSession,
    image: ImageView<'_>,
    landmarks: &ImageFacialLandmarks,
    cancelled: &mut dyn FnMut() -> bool,
) -> Result<InputTensor, InferenceError> {
    session.preprocess_aligned(image, landmarks, &mut *cancelled)
}

fn preprocess_face_region(
    session: &OnnxSession,
    image: ImageView<'_>,
    region: &FaceRegion,
    cancelled: &mut dyn FnMut() -> bool,
) -> Result<InputTensor, InferenceError> {
    session.preprocess_face_region(image, region, &mut *cancelled)
}

fn assess_landmark_frame_quality(
    frame: &CapturedFrame,
    landmarks: &FacialLandmarks,
    config: QualityConfig,
    cancelled: &mut dyn FnMut() -> bool,
) -> Result<faceauth_quality::QualityReport, QualityError> {
    let bounds = landmarks.region().bounds();
    let measurements = landmarks.measurements();
    let geometry = FaceGeometry {
        center_x: (bounds.left() + bounds.right()) * 0.5,
        center_y: (bounds.top() + bounds.bottom()) * 0.5,
        width: bounds.right() - bounds.left(),
        height: bounds.bottom() - bounds.top(),
        yaw_degrees: measurements.yaw_degrees(),
        pitch_degrees: measurements.pitch_degrees(),
        roll_degrees: measurements.roll_degrees(),
        landmark_confidence: measurements.landmark_confidence(),
        visible_fraction: measurements.visible_fraction(),
    };
    assess_cancellable(quality_frame_view(frame)?, geometry, config, &mut *cancelled)
}

fn inference_frame_view(frame: &CapturedFrame) -> Result<ImageView<'_>, InferenceError> {
    let summary = frame.summary();
    let format = match summary.format.pixel_format {
        PixelFormat::Gray8 => ImageFormat::Gray8,
        PixelFormat::Yuyv => ImageFormat::Yuyv,
        PixelFormat::Mjpeg => return Err(InferenceError::SourceImageInvalid),
    };
    Ok(ImageView {
        width: summary.format.width,
        height: summary.format.height,
        format,
        bytes: frame.bytes(),
    })
}

fn quality_frame_view(frame: &CapturedFrame) -> Result<QualityImageView<'_>, QualityError> {
    let summary = frame.summary();
    match summary.format.pixel_format {
        PixelFormat::Gray8 => {
            Ok(Gray8View::new(summary.format.width, summary.format.height, frame.bytes())?.into())
        }
        PixelFormat::Yuyv => {
            Ok(YuyvView::new(summary.format.width, summary.format.height, frame.bytes())?.into())
        }
        PixelFormat::Mjpeg => Err(QualityError::UnsupportedImageFormat),
    }
}

const fn observation_progress(progress: ChallengeProgress) -> ObservationProgress {
    match progress {
        ChallengeProgress::BaselineRequired => ObservationProgress::HoldStill,
        ChallengeProgress::ActionRequired(ChallengeAction::Blink) => ObservationProgress::Blink,
        ChallengeProgress::ActionRequired(ChallengeAction::TurnLeft) => {
            ObservationProgress::TurnLeft
        }
        ChallengeProgress::ActionRequired(ChallengeAction::TurnRight) => {
            ObservationProgress::TurnRight
        }
        ChallengeProgress::RecoveryRequired => ObservationProgress::ReturnToCenter,
        ChallengeProgress::Passed => ObservationProgress::Processing,
    }
}

fn ensure_active(cancelled: &mut dyn FnMut() -> bool) -> Result<(), ObservationFailure> {
    if cancelled() { Err(ObservationFailure::Cancelled) } else { Ok(()) }
}

fn capture_failure(cancelled: &mut dyn FnMut() -> bool) -> ObservationFailure {
    if cancelled() { ObservationFailure::Cancelled } else { ObservationFailure::Capture }
}

fn inference_failure(cancelled: &mut dyn FnMut() -> bool) -> ObservationFailure {
    if cancelled() { ObservationFailure::Cancelled } else { ObservationFailure::Inference }
}

fn liveness_failure(cancelled: &mut dyn FnMut() -> bool) -> ObservationFailure {
    if cancelled() { ObservationFailure::Cancelled } else { ObservationFailure::Liveness }
}

fn invalid_evidence_failure(cancelled: &mut dyn FnMut() -> bool) -> ObservationFailure {
    if cancelled() { ObservationFailure::Cancelled } else { ObservationFailure::InvalidEvidence }
}

/// Construction failure before cameras are streamed.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum ProductionObservationPipelineError {
    #[error("production observation pipeline policy is invalid")]
    InvalidPolicy,
    #[error("production model role mismatch: expected {expected:?}, received {actual:?}")]
    ModelRoleMismatch { expected: ModelRole, actual: ModelRole },
}

/// Sanitized runtime failure from the shared production pipeline.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum ObservationFailure {
    #[error("observation was cancelled")]
    Cancelled,
    #[error("observation capture failed")]
    Capture,
    #[error("observation inference failed")]
    Inference,
    #[error("observation liveness failed")]
    Liveness,
    #[error("observation evidence was invalid")]
    InvalidEvidence,
    #[error("observation attempt limit was reached")]
    AttemptLimit,
    #[error("observation coordination failed")]
    Internal,
}

#[cfg(test)]
mod tests {
    use super::FreshPairIdentity;

    #[test]
    fn fresh_pair_guard_accepts_sequence_wrap_but_rejects_reuse_and_old_time() {
        let previous = FreshPairIdentity {
            infrared_sequence: u32::MAX,
            visible_sequence: u32::MAX,
            timestamp_micros: 10,
        };
        assert!(
            FreshPairIdentity { infrared_sequence: 0, visible_sequence: 0, timestamp_micros: 11 }
                .is_fresh_after(previous)
        );
        assert!(
            !FreshPairIdentity {
                infrared_sequence: u32::MAX,
                visible_sequence: 0,
                timestamp_micros: 11,
            }
            .is_fresh_after(previous)
        );
        assert!(
            !FreshPairIdentity { infrared_sequence: 0, visible_sequence: 0, timestamp_micros: 10 }
                .is_fresh_after(previous)
        );
    }
}
