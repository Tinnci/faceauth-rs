//! Model provenance and integrity checks independent of an inference runtime.

use std::{
    fs::File,
    io::{self, Read},
    path::Path,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Current model-manifest schema version.
pub const MODEL_MANIFEST_SCHEMA_VERSION: u16 = 8;

/// Maximum artifact name or version length.
pub const MAX_ARTIFACT_LABEL_LENGTH: usize = 128;

/// Maximum source URL length.
pub const MAX_SOURCE_URL_LENGTH: usize = 2048;

/// Maximum image dimension accepted by the static preprocessing contract.
pub const MAX_INPUT_DIMENSION: u32 = 4096;

/// Maximum number of declared output tensors.
pub const MAX_OUTPUT_TENSORS: usize = 8;

/// Maximum rank of one declared output tensor.
pub const MAX_TENSOR_RANK: usize = 8;

/// Maximum number of elements in one input or output tensor.
pub const MAX_TENSOR_ELEMENTS: u64 = 64 * 1024 * 1024;

/// Maximum aggregate elements across all declared output tensors.
pub const MAX_TOTAL_OUTPUT_ELEMENTS: u64 = 64 * 1024 * 1024;

/// Maximum fixed detector candidates admitted by a reviewed graph.
pub const MAX_FACE_DETECTION_CANDIDATES: u32 = 4096;

/// Minimum landmarks required by the geometric alignment boundary.
pub const MIN_FACE_LANDMARKS: u32 = 5;

/// Maximum fixed landmarks admitted from one face crop.
pub const MAX_FACE_LANDMARKS: u32 = 512;

/// Exact number of landmarks admitted by the initial similarity-alignment contract.
pub const FACE_ALIGNMENT_LANDMARKS: usize = 5;

/// Role a model serves in the biometric pipeline.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelRole {
    /// Face bounding-box detector.
    FaceDetector,
    /// Facial landmark detector used for geometric alignment.
    FaceLandmarks,
    /// Face embedding model used after alignment.
    FaceEmbedding,
    /// Passive presentation-attack detector for IR frames.
    PassiveLivenessInfrared,
    /// Passive presentation-attack detector for visible-light frames.
    PassiveLivenessVisible,
    /// Passive presentation-attack detector that fuses paired modalities.
    PassiveLivenessFusion,
}

/// Tensor memory layout expected by the model.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TensorLayout {
    /// Batch, channel, height, width.
    Nchw,
    /// Batch, height, width, channel.
    Nhwc,
}

/// Pixel channel interpretation before model-specific normalization.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ColorSpace {
    /// One-channel grayscale input.
    Grayscale,
    /// Red, green, blue channel order.
    Rgb,
    /// Blue, green, red channel order.
    Bgr,
}

/// Tensor element type admitted by the initial biometric runtime.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TensorElementType {
    /// IEEE 754 single-precision floating point.
    Float32,
}

/// Deterministic image resize operation admitted by the preprocessing boundary.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ResizeFilter {
    /// Bilinear interpolation with half-pixel center coordinates and clamped edges.
    BilinearHalfPixel,
}

/// Manifest-bound detector-box expansion used to create a landmark model input.
///
/// The expanded region is always square and is rejected if any edge would leave the image. This
/// avoids model-dependent implicit padding and aspect-ratio behavior.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FaceCropContract {
    /// Square side length as a multiple of the detector box's larger dimension.
    pub scale: f32,
    /// Horizontal center shift as a multiple of the detector box width; positive moves right.
    pub center_offset_x: f32,
    /// Vertical center shift as a multiple of the detector box height; positive moves down.
    pub center_offset_y: f32,
}

/// Calibrated semantic outputs required from a landmark model used for quality and active PAD.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LandmarkMeasurementContract {
    /// Point confidence at or above this value contributes to the visible fraction.
    pub minimum_visible_confidence: f32,
    /// Maximum admitted absolute yaw in degrees.
    pub maximum_absolute_yaw_degrees: f32,
    /// Maximum admitted absolute pitch in degrees.
    pub maximum_absolute_pitch_degrees: f32,
    /// Maximum admitted absolute roll in degrees.
    pub maximum_absolute_roll_degrees: f32,
}

impl LandmarkMeasurementContract {
    fn validate(self) -> Result<(), ModelError> {
        if !self.minimum_visible_confidence.is_finite()
            || !(0.0..=1.0).contains(&self.minimum_visible_confidence)
            || !self.maximum_absolute_yaw_degrees.is_finite()
            || !(1.0..=90.0).contains(&self.maximum_absolute_yaw_degrees)
            || !self.maximum_absolute_pitch_degrees.is_finite()
            || !(1.0..=90.0).contains(&self.maximum_absolute_pitch_degrees)
            || !self.maximum_absolute_roll_degrees.is_finite()
            || !(1.0..=180.0).contains(&self.maximum_absolute_roll_degrees)
        {
            return Err(ModelError::InvalidLandmarkMeasurementContract);
        }
        Ok(())
    }
}

impl FaceCropContract {
    fn validate(self) -> Result<(), ModelError> {
        if !self.scale.is_finite()
            || !(1.0..=3.0).contains(&self.scale)
            || !self.center_offset_x.is_finite()
            || !self.center_offset_y.is_finite()
            || !(-0.5..=0.5).contains(&self.center_offset_x)
            || !(-0.5..=0.5).contains(&self.center_offset_y)
        {
            return Err(ModelError::InvalidFaceCropContract);
        }
        Ok(())
    }
}

/// One normalized destination point in an aligned model input.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NormalizedPoint {
    /// Horizontal coordinate in `0..=1`.
    pub x: f32,
    /// Vertical coordinate in `0..=1`.
    pub y: f32,
}

/// Exact five-point similarity-alignment contract for an embedding input.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FaceAlignmentContract {
    /// Complete compatibility digest of the landmark model that supplies the source points.
    pub landmarks_compatibility_sha256: String,
    /// Five unique indices in the landmark model's fixed topology.
    pub landmark_indices: [u16; FACE_ALIGNMENT_LANDMARKS],
    /// Corresponding normalized points in the embedding model's fixed input image.
    pub reference_points: [NormalizedPoint; FACE_ALIGNMENT_LANDMARKS],
    /// Maximum root-mean-square normalized landmark residual admitted after fitting.
    pub maximum_normalized_residual: f32,
}

impl FaceAlignmentContract {
    fn validate(&self) -> Result<(), ModelError> {
        if !valid_sha256(&self.landmarks_compatibility_sha256) {
            return Err(ModelError::InvalidAlignmentContract);
        }
        let mut sorted = self.landmark_indices;
        sorted.sort_unstable();
        if sorted.windows(2).any(|pair| pair[0] == pair[1])
            || sorted.iter().any(|index| u32::from(*index) >= MAX_FACE_LANDMARKS)
            || self.reference_points.iter().any(|point| {
                !point.x.is_finite()
                    || !point.y.is_finite()
                    || !(0.0..=1.0).contains(&point.x)
                    || !(0.0..=1.0).contains(&point.y)
            })
            || !self.maximum_normalized_residual.is_finite()
            || !(0.001..=0.25).contains(&self.maximum_normalized_residual)
        {
            return Err(ModelError::InvalidAlignmentContract);
        }
        let center_x = self.reference_points.iter().map(|point| point.x).sum::<f32>() / 5.0;
        let center_y = self.reference_points.iter().map(|point| point.y).sum::<f32>() / 5.0;
        let variance = self.reference_points.iter().try_fold(0.0_f32, |sum, point| {
            let x = point.x - center_x;
            let y = point.y - center_y;
            let next = x.mul_add(x, y.mul_add(y, sum));
            next.is_finite().then_some(next)
        });
        if variance.is_none_or(|value| value <= 1.0e-6) {
            return Err(ModelError::InvalidAlignmentContract);
        }
        Ok(())
    }
}

/// Per-channel affine conversion from an 8-bit pixel to a model tensor value.
///
/// For channel `c`, preprocessing computes `pixel * scale[c] + bias[c]`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChannelNormalization {
    /// Finite, non-zero multiplier for each input channel.
    pub scale: Vec<f32>,
    /// Finite offset for each input channel.
    pub bias: Vec<f32>,
}

/// Static input tensor contract recorded with a model.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InputContract {
    /// Exact ONNX graph input name.
    pub name: String,
    /// Input width in pixels.
    pub width: u32,
    /// Input height in pixels.
    pub height: u32,
    /// Input channel count.
    pub channels: u8,
    /// Tensor memory layout.
    pub layout: TensorLayout,
    /// Pixel channel interpretation.
    pub color_space: ColorSpace,
    /// Required tensor element type.
    pub element_type: TensorElementType,
    /// Exact resize algorithm used when source dimensions differ.
    pub resize_filter: ResizeFilter,
    /// Exact per-channel conversion from 8-bit pixels to float32.
    pub normalization: ChannelNormalization,
}

impl InputContract {
    /// Return the exact fixed batch-1 tensor shape.
    #[must_use]
    pub fn dimensions(&self) -> [u32; 4] {
        match self.layout {
            TensorLayout::Nchw => [1, u32::from(self.channels), self.height, self.width],
            TensorLayout::Nhwc => [1, self.height, self.width, u32::from(self.channels)],
        }
    }
}

/// Exact fixed output tensor contract.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OutputContract {
    /// Exact ONNX graph output name.
    pub name: String,
    /// Fixed tensor dimensions; dynamic outputs are not admitted.
    pub dimensions: Vec<u32>,
    /// Required tensor element type.
    pub element_type: TensorElementType,
    /// Machine-checked meaning consumed by a role-specific adapter.
    pub semantic: OutputSemantic,
}

/// Security-relevant meaning of one model output.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum OutputSemantic {
    /// Normalized detector boxes in `[1, candidates, 4]` `x_min, y_min, x_max, y_max` order.
    FaceDetectionBoxes,
    /// Detector confidence scores in `[1, candidates]`, paired by candidate index with boxes.
    FaceDetectionScores,
    /// Normalized landmark points in `[1, landmarks, 2]` `x, y` order for one face crop.
    FaceLandmarks,
    /// Per-point landmark confidence in `[1, landmarks]`, in topology order.
    FaceLandmarkConfidence,
    /// Model-estimated yaw, pitch, and roll degrees in `[1,3]` order.
    FacePoseDegrees,
    /// Model-estimated left and right eye openness in `[1,2]`, each normalized to `0..=1`.
    EyeOpenness,
    /// Face embedding vector used only by the embedding adapter.
    Embedding,
    /// Scalar probability that the observation is live, in the inclusive range 0..=1.
    LiveProbability,
    /// Bounded output retained for a future reviewed role-specific adapter.
    Auxiliary,
}

/// Auditable metadata required before a model can enter the inference pipeline.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelManifest {
    /// Manifest schema version.
    pub schema_version: u16,
    /// Stable artifact name.
    pub name: String,
    /// Upstream artifact or training release version.
    pub version: String,
    /// Pipeline role assigned to this artifact.
    pub role: ModelRole,
    /// HTTPS page or immutable artifact URL documenting provenance.
    pub source_url: String,
    /// SPDX license expression for the model weights.
    pub license_spdx: String,
    /// Lowercase hexadecimal SHA-256 digest of the exact model file.
    pub sha256: String,
    /// Static input tensor contract.
    pub input: InputContract,
    /// Required detector-region derivation for landmark model inputs.
    pub face_crop: Option<FaceCropContract>,
    /// Required calibrated measurement semantics for landmark roles.
    pub landmark_measurement: Option<LandmarkMeasurementContract>,
    /// Required landmark-bound geometry preprocessing for roles that consume an aligned face.
    pub face_alignment: Option<FaceAlignmentContract>,
    /// Exact bounded output tensors expected from the graph.
    pub outputs: Vec<OutputContract>,
}

impl ModelManifest {
    /// Validate manifest structure without opening the model file.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] when the schema, labels, source, license, digest, or input contract
    /// is invalid.
    pub fn validate(&self) -> Result<(), ModelError> {
        if self.schema_version != MODEL_MANIFEST_SCHEMA_VERSION {
            return Err(ModelError::UnsupportedSchema {
                received: self.schema_version,
                supported: MODEL_MANIFEST_SCHEMA_VERSION,
            });
        }
        validate_label("name", &self.name)?;
        validate_label("version", &self.version)?;
        if self.source_url.len() > MAX_SOURCE_URL_LENGTH || !self.source_url.starts_with("https://")
        {
            return Err(ModelError::InvalidSourceUrl);
        }
        if spdx::Expression::parse(&self.license_spdx).is_err() {
            return Err(ModelError::InvalidLicenseExpression);
        }
        if !valid_sha256(&self.sha256) {
            return Err(ModelError::InvalidSha256);
        }
        validate_tensor_name(&self.input.name)?;
        if self.input.width == 0
            || self.input.height == 0
            || self.input.width > MAX_INPUT_DIMENSION
            || self.input.height > MAX_INPUT_DIMENSION
        {
            return Err(ModelError::InvalidInputDimensions);
        }
        let expected_channels = match self.input.color_space {
            ColorSpace::Grayscale => 1,
            ColorSpace::Rgb | ColorSpace::Bgr => 3,
        };
        if self.input.channels != expected_channels {
            return Err(ModelError::InvalidChannelCount {
                expected: expected_channels,
                actual: self.input.channels,
            });
        }
        match (&self.role, &self.face_alignment) {
            (ModelRole::FaceEmbedding, Some(alignment)) => alignment.validate()?,
            (ModelRole::FaceEmbedding, None) => {
                return Err(ModelError::InvalidAlignmentContract);
            }
            (_, None) => {}
            (_, Some(_)) => return Err(ModelError::InvalidAlignmentContract),
        }
        match (self.role, self.face_crop) {
            (ModelRole::FaceLandmarks, Some(crop)) => crop.validate()?,
            (ModelRole::FaceLandmarks, None) | (_, Some(_)) => {
                return Err(ModelError::InvalidFaceCropContract);
            }
            (_, None) => {}
        }
        match (self.role, self.landmark_measurement) {
            (ModelRole::FaceLandmarks, Some(contract)) => contract.validate()?,
            (ModelRole::FaceLandmarks, None) | (_, Some(_)) => {
                return Err(ModelError::InvalidLandmarkMeasurementContract);
            }
            (_, None) => {}
        }
        let channel_count = usize::from(self.input.channels);
        if self.input.normalization.scale.len() != channel_count
            || self.input.normalization.bias.len() != channel_count
            || self
                .input
                .normalization
                .scale
                .iter()
                .any(|value| !value.is_finite() || *value == 0.0)
            || self.input.normalization.bias.iter().any(|value| !value.is_finite())
        {
            return Err(ModelError::InvalidNormalization);
        }
        validate_dimensions(&self.input.dimensions())?;
        if self.outputs.is_empty() || self.outputs.len() > MAX_OUTPUT_TENSORS {
            return Err(ModelError::InvalidOutputCount { actual: self.outputs.len() });
        }
        let mut total_output_elements = 0_u64;
        for (index, output) in self.outputs.iter().enumerate() {
            validate_tensor_name(&output.name)?;
            validate_dimensions(&output.dimensions)?;
            let output_elements = dimension_product(&output.dimensions)?;
            total_output_elements = total_output_elements
                .checked_add(output_elements)
                .ok_or(ModelError::InvalidTensorDimensions)?;
            if total_output_elements > MAX_TOTAL_OUTPUT_ELEMENTS {
                return Err(ModelError::InvalidTensorDimensions);
            }
            if self.outputs[..index].iter().any(|existing| existing.name == output.name) {
                return Err(ModelError::DuplicateOutputName { name: output.name.clone() });
            }
        }
        validate_output_semantics(self.role, &self.outputs)?;
        Ok(())
    }

    /// Compute the compatibility identity for templates and calibrated thresholds.
    ///
    /// The digest covers the complete validated manifest, including the ONNX hash, role,
    /// preprocessing, exact graph contract, and output semantics.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] when the manifest is invalid or cannot be serialized.
    pub fn compatibility_sha256(&self) -> Result<String, ModelError> {
        self.validate()?;
        let encoded = serde_json::to_vec(self)?;
        let mut hasher = Sha256::new();
        hasher.update(b"faceauth-model-compatibility-v1\0");
        hasher.update(encoded);
        Ok(hex_lower(&hasher.finalize()))
    }

    /// Verify a model file against this manifest.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] when the manifest is invalid, the path is not a regular file, the
    /// file cannot be read, or its SHA-256 digest differs from the manifest.
    pub fn verify_file(&self, path: &Path) -> Result<(), ModelError> {
        self.validate()?;
        let file = File::open(path)?;
        if !file.metadata()?.is_file() {
            return Err(ModelError::NotRegularFile);
        }
        self.verify_reader(file)
    }

    fn verify_reader(&self, mut reader: impl Read) -> Result<(), ModelError> {
        let mut hasher = Sha256::new();
        let mut buffer = [0_u8; 16 * 1024];
        loop {
            let count = reader.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            hasher.update(&buffer[..count]);
        }
        let actual = hex_lower(&hasher.finalize());
        if actual != self.sha256 {
            return Err(ModelError::DigestMismatch { expected: self.sha256.clone(), actual });
        }
        Ok(())
    }
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_hexdigit())
        && !value.bytes().any(|byte| byte.is_ascii_uppercase())
}

/// Invalid model manifest or artifact.
#[derive(Debug, Error)]
pub enum ModelError {
    /// The manifest uses an unsupported schema.
    #[error("unsupported model manifest schema {received}; this build supports {supported}")]
    UnsupportedSchema {
        /// Version found in the manifest.
        received: u16,
        /// Version supported by this build.
        supported: u16,
    },
    /// A model name or version label is invalid.
    #[error("model {field} must be 1 to 128 ASCII alphanumeric, '.', '_', '+', or '-' characters")]
    InvalidLabel {
        /// Invalid manifest field.
        field: &'static str,
    },
    /// The source is not a bounded HTTPS URL.
    #[error("model source must be a bounded HTTPS URL")]
    InvalidSourceUrl,
    /// The model-weight license is not a valid SPDX expression.
    #[error("model license is not a valid SPDX expression")]
    InvalidLicenseExpression,
    /// The digest is not 64 lowercase hexadecimal characters.
    #[error("model SHA-256 must be 64 lowercase hexadecimal characters")]
    InvalidSha256,
    /// Width and height must be fixed and inside hard bounds.
    #[error("model input dimensions are invalid or excessive")]
    InvalidInputDimensions,
    /// Channel count does not match the declared color space.
    #[error("model input has {actual} channels; declared color space requires {expected}")]
    InvalidChannelCount {
        /// Required channel count.
        expected: u8,
        /// Manifest channel count.
        actual: u8,
    },
    /// Normalization vectors must exactly match the channel count and contain finite values.
    #[error("model input normalization is invalid")]
    InvalidNormalization,
    /// Face alignment is absent, attached to the wrong role, malformed, or degenerate.
    #[error("model face-alignment contract is invalid")]
    InvalidAlignmentContract,
    /// Landmark crop policy is absent, attached to the wrong role, or malformed.
    #[error("model face-crop contract is invalid")]
    InvalidFaceCropContract,
    /// Landmark measurement calibration is absent, attached to the wrong role, or malformed.
    #[error("model landmark-measurement contract is invalid")]
    InvalidLandmarkMeasurementContract,
    /// Tensor name is empty, excessive, or contains unsupported bytes.
    #[error("model tensor name is invalid")]
    InvalidTensorName,
    /// Output tensor count must be bounded and non-zero.
    #[error("model has invalid declared output count {actual}")]
    InvalidOutputCount {
        /// Actual declared output count.
        actual: usize,
    },
    /// Tensor rank, dimensions, or element count is invalid.
    #[error("model tensor dimensions are invalid or excessive")]
    InvalidTensorDimensions,
    /// Output names must be unique.
    #[error("duplicate model output tensor name {name}")]
    DuplicateOutputName {
        /// Repeated output name.
        name: String,
    },
    /// Output semantic tags do not match the assigned model role.
    #[error("model output semantics do not match its role")]
    OutputSemanticMismatch,
    /// The model path did not resolve to a regular file.
    #[error("model artifact is not a regular file")]
    NotRegularFile,
    /// The model file digest did not match the manifest.
    #[error("model digest mismatch: expected {expected}, found {actual}")]
    DigestMismatch {
        /// Digest recorded in the manifest.
        expected: String,
        /// Digest computed from the model file.
        actual: String,
    },
    /// The model artifact could not be opened or read.
    #[error("unable to read model artifact: {0}")]
    Io(#[from] io::Error),
    /// Manifest compatibility serialization failed.
    #[error("unable to serialize model compatibility contract: {0}")]
    Serialization(#[from] serde_json::Error),
}

fn validate_label(field: &'static str, value: &str) -> Result<(), ModelError> {
    if value.is_empty()
        || value.len() > MAX_ARTIFACT_LABEL_LENGTH
        || !value.bytes().all(|byte| byte.is_ascii_alphanumeric() || b"._+-".contains(&byte))
    {
        Err(ModelError::InvalidLabel { field })
    } else {
        Ok(())
    }
}

fn validate_tensor_name(value: &str) -> Result<(), ModelError> {
    if value.is_empty()
        || value.len() > MAX_ARTIFACT_LABEL_LENGTH
        || !value.bytes().all(|byte| byte.is_ascii_alphanumeric() || b"._+-/:".contains(&byte))
    {
        Err(ModelError::InvalidTensorName)
    } else {
        Ok(())
    }
}

fn validate_dimensions(dimensions: &[u32]) -> Result<(), ModelError> {
    if dimensions.is_empty() || dimensions.len() > MAX_TENSOR_RANK || dimensions.contains(&0) {
        return Err(ModelError::InvalidTensorDimensions);
    }
    if dimension_product(dimensions)? > MAX_TENSOR_ELEMENTS {
        Err(ModelError::InvalidTensorDimensions)
    } else {
        Ok(())
    }
}

fn validate_output_semantics(
    role: ModelRole,
    outputs: &[OutputContract],
) -> Result<(), ModelError> {
    match role {
        ModelRole::FaceDetector => validate_face_detector_outputs(outputs)?,
        ModelRole::FaceLandmarks => validate_face_landmark_outputs(outputs)?,
        ModelRole::FaceEmbedding
            if semantic_count(outputs, OutputSemantic::Embedding) == 1
                && semantic_count(outputs, OutputSemantic::LiveProbability) == 0
                && detector_and_landmark_semantics_absent(outputs) => {}
        ModelRole::PassiveLivenessInfrared
        | ModelRole::PassiveLivenessVisible
        | ModelRole::PassiveLivenessFusion
            if semantic_count(outputs, OutputSemantic::LiveProbability) == 1
                && semantic_count(outputs, OutputSemantic::Embedding) == 0
                && detector_and_landmark_semantics_absent(outputs) => {}
        _ => return Err(ModelError::OutputSemanticMismatch),
    }
    for output in outputs {
        match output.semantic {
            OutputSemantic::Embedding => {
                let elements = dimension_product(&output.dimensions)?;
                let valid_shape = matches!(output.dimensions.as_slice(), [dimension] if *dimension >= 32 && *dimension <= 4096)
                    || matches!(output.dimensions.as_slice(), [1, dimension] if *dimension >= 32 && *dimension <= 4096);
                if !valid_shape || !(32..=4096).contains(&elements) {
                    return Err(ModelError::OutputSemanticMismatch);
                }
            }
            OutputSemantic::LiveProbability => {
                if dimension_product(&output.dimensions)? != 1 {
                    return Err(ModelError::OutputSemanticMismatch);
                }
            }
            OutputSemantic::FaceDetectionBoxes
            | OutputSemantic::FaceDetectionScores
            | OutputSemantic::FaceLandmarks
            | OutputSemantic::FaceLandmarkConfidence
            | OutputSemantic::FacePoseDegrees
            | OutputSemantic::EyeOpenness
            | OutputSemantic::Auxiliary => {}
        }
    }
    Ok(())
}

fn validate_face_detector_outputs(outputs: &[OutputContract]) -> Result<(), ModelError> {
    if outputs.len() != 2 {
        return Err(ModelError::OutputSemanticMismatch);
    }
    let boxes = unique_semantic_output(outputs, OutputSemantic::FaceDetectionBoxes)?;
    let scores = unique_semantic_output(outputs, OutputSemantic::FaceDetectionScores)?;
    let [1, candidate_count, 4] = boxes.dimensions.as_slice() else {
        return Err(ModelError::OutputSemanticMismatch);
    };
    let [1, score_count] = scores.dimensions.as_slice() else {
        return Err(ModelError::OutputSemanticMismatch);
    };
    if candidate_count != score_count
        || !(1..=MAX_FACE_DETECTION_CANDIDATES).contains(candidate_count)
    {
        return Err(ModelError::OutputSemanticMismatch);
    }
    Ok(())
}

fn validate_face_landmark_outputs(outputs: &[OutputContract]) -> Result<(), ModelError> {
    if outputs.len() != 4 {
        return Err(ModelError::OutputSemanticMismatch);
    }
    let landmarks = unique_semantic_output(outputs, OutputSemantic::FaceLandmarks)?;
    let confidence = unique_semantic_output(outputs, OutputSemantic::FaceLandmarkConfidence)?;
    let pose = unique_semantic_output(outputs, OutputSemantic::FacePoseDegrees)?;
    let eye_openness = unique_semantic_output(outputs, OutputSemantic::EyeOpenness)?;
    let [1, landmark_count, 2] = landmarks.dimensions.as_slice() else {
        return Err(ModelError::OutputSemanticMismatch);
    };
    if !(MIN_FACE_LANDMARKS..=MAX_FACE_LANDMARKS).contains(landmark_count) {
        return Err(ModelError::OutputSemanticMismatch);
    }
    if confidence.dimensions.as_slice() != [1, *landmark_count]
        || pose.dimensions.as_slice() != [1, 3]
        || eye_openness.dimensions.as_slice() != [1, 2]
    {
        return Err(ModelError::OutputSemanticMismatch);
    }
    Ok(())
}

fn unique_semantic_output(
    outputs: &[OutputContract],
    semantic: OutputSemantic,
) -> Result<&OutputContract, ModelError> {
    let mut matching = outputs.iter().filter(|output| output.semantic == semantic);
    let output = matching.next().ok_or(ModelError::OutputSemanticMismatch)?;
    if matching.next().is_some() { Err(ModelError::OutputSemanticMismatch) } else { Ok(output) }
}

fn semantic_count(outputs: &[OutputContract], semantic: OutputSemantic) -> usize {
    outputs.iter().filter(|output| output.semantic == semantic).count()
}

fn detector_and_landmark_semantics_absent(outputs: &[OutputContract]) -> bool {
    outputs.iter().all(|output| {
        !matches!(
            output.semantic,
            OutputSemantic::FaceDetectionBoxes
                | OutputSemantic::FaceDetectionScores
                | OutputSemantic::FaceLandmarks
                | OutputSemantic::FaceLandmarkConfidence
                | OutputSemantic::FacePoseDegrees
                | OutputSemantic::EyeOpenness
        )
    })
}

fn dimension_product(dimensions: &[u32]) -> Result<u64, ModelError> {
    dimensions
        .iter()
        .try_fold(1_u64, |product, dimension| product.checked_mul(u64::from(*dimension)))
        .ok_or(ModelError::InvalidTensorDimensions)
}

fn hex_lower(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    bytes
        .iter()
        .flat_map(|byte| {
            [
                char::from(DIGITS[usize::from(byte >> 4)]),
                char::from(DIGITS[usize::from(byte & 0x0f)]),
            ]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    fn manifest() -> ModelManifest {
        ModelManifest {
            schema_version: MODEL_MANIFEST_SCHEMA_VERSION,
            name: "example-detector".to_owned(),
            version: "1.0.0".to_owned(),
            role: ModelRole::FaceDetector,
            source_url: "https://example.invalid/models/example-detector.onnx".to_owned(),
            license_spdx: "Apache-2.0".to_owned(),
            sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".to_owned(),
            input: InputContract {
                name: "input".to_owned(),
                width: 640,
                height: 360,
                channels: 3,
                layout: TensorLayout::Nchw,
                color_space: ColorSpace::Rgb,
                element_type: TensorElementType::Float32,
                resize_filter: ResizeFilter::BilinearHalfPixel,
                normalization: ChannelNormalization {
                    scale: vec![1.0 / 255.0; 3],
                    bias: vec![0.0; 3],
                },
            },
            face_crop: None,
            landmark_measurement: None,
            face_alignment: None,
            outputs: vec![
                OutputContract {
                    name: "boxes".to_owned(),
                    dimensions: vec![1, 100, 4],
                    element_type: TensorElementType::Float32,
                    semantic: OutputSemantic::FaceDetectionBoxes,
                },
                OutputContract {
                    name: "scores".to_owned(),
                    dimensions: vec![1, 100],
                    element_type: TensorElementType::Float32,
                    semantic: OutputSemantic::FaceDetectionScores,
                },
            ],
        }
    }

    fn landmark_outputs(count: u32) -> Vec<OutputContract> {
        vec![
            OutputContract {
                name: "landmarks".to_owned(),
                dimensions: vec![1, count, 2],
                element_type: TensorElementType::Float32,
                semantic: OutputSemantic::FaceLandmarks,
            },
            OutputContract {
                name: "confidence".to_owned(),
                dimensions: vec![1, count],
                element_type: TensorElementType::Float32,
                semantic: OutputSemantic::FaceLandmarkConfidence,
            },
            OutputContract {
                name: "pose".to_owned(),
                dimensions: vec![1, 3],
                element_type: TensorElementType::Float32,
                semantic: OutputSemantic::FacePoseDegrees,
            },
            OutputContract {
                name: "eyes".to_owned(),
                dimensions: vec![1, 2],
                element_type: TensorElementType::Float32,
                semantic: OutputSemantic::EyeOpenness,
            },
        ]
    }

    #[test]
    fn valid_manifest_and_digest_are_accepted() -> Result<(), ModelError> {
        let manifest = manifest();
        manifest.validate()?;
        manifest.verify_reader(Cursor::new(b"abc"))
    }

    #[test]
    fn modified_artifact_is_rejected() {
        let error = manifest().verify_reader(Cursor::new(b"abd"));
        assert!(matches!(error, Err(ModelError::DigestMismatch { .. })));
    }

    #[test]
    fn insecure_source_is_rejected() {
        let mut manifest = manifest();
        manifest.source_url = "http://example.invalid/model.onnx".to_owned();
        assert!(matches!(manifest.validate(), Err(ModelError::InvalidSourceUrl)));
    }

    #[test]
    fn invalid_spdx_expression_is_rejected() {
        let mut manifest = manifest();
        manifest.license_spdx = "probably-free".to_owned();
        assert!(matches!(manifest.validate(), Err(ModelError::InvalidLicenseExpression)));
    }

    #[test]
    fn color_space_and_channel_count_must_agree() {
        let mut manifest = manifest();
        manifest.input.color_space = ColorSpace::Grayscale;
        assert!(matches!(
            manifest.validate(),
            Err(ModelError::InvalidChannelCount { expected: 1, actual: 3 })
        ));
    }

    #[test]
    fn dynamic_or_excessive_tensor_contracts_are_rejected() {
        let mut invalid_output = manifest();
        invalid_output.outputs[0].dimensions = vec![0, 4];
        assert!(matches!(invalid_output.validate(), Err(ModelError::InvalidTensorDimensions)));

        let mut excessive_input = manifest();
        excessive_input.input.width = MAX_INPUT_DIMENSION + 1;
        assert!(matches!(excessive_input.validate(), Err(ModelError::InvalidInputDimensions)));

        let mut excessive_total = manifest();
        excessive_total.outputs = (0..2)
            .map(|index| OutputContract {
                name: format!("output-{index}"),
                dimensions: vec![u32::try_from(MAX_TENSOR_ELEMENTS / 2 + 1).unwrap_or(u32::MAX)],
                element_type: TensorElementType::Float32,
                semantic: OutputSemantic::Auxiliary,
            })
            .collect();
        assert!(matches!(excessive_total.validate(), Err(ModelError::InvalidTensorDimensions)));
    }

    #[test]
    fn tensor_names_and_outputs_are_bounded_and_unique() {
        let mut duplicate_output = manifest();
        duplicate_output.outputs.push(duplicate_output.outputs[0].clone());
        assert!(matches!(duplicate_output.validate(), Err(ModelError::DuplicateOutputName { .. })));

        let mut invalid_name = manifest();
        invalid_name.input.name = "bad name".to_owned();
        assert!(matches!(invalid_name.validate(), Err(ModelError::InvalidTensorName)));
    }

    #[test]
    fn normalization_must_match_channels_and_be_finite() {
        let mut wrong_count = manifest();
        wrong_count.input.normalization.scale.pop();
        assert!(matches!(wrong_count.validate(), Err(ModelError::InvalidNormalization)));

        let mut non_finite = manifest();
        non_finite.input.normalization.bias[0] = f32::NAN;
        assert!(matches!(non_finite.validate(), Err(ModelError::InvalidNormalization)));

        let mut zero_scale = manifest();
        zero_scale.input.normalization.scale[0] = 0.0;
        assert!(matches!(zero_scale.validate(), Err(ModelError::InvalidNormalization)));
    }

    #[test]
    fn role_requires_matching_output_semantics() {
        let mut embedding = manifest();
        embedding.role = ModelRole::FaceEmbedding;
        embedding.face_alignment = Some(FaceAlignmentContract {
            landmarks_compatibility_sha256:
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
            landmark_indices: [0, 1, 2, 3, 4],
            reference_points: [
                NormalizedPoint { x: 0.3, y: 0.35 },
                NormalizedPoint { x: 0.7, y: 0.35 },
                NormalizedPoint { x: 0.5, y: 0.52 },
                NormalizedPoint { x: 0.36, y: 0.72 },
                NormalizedPoint { x: 0.64, y: 0.72 },
            ],
            maximum_normalized_residual: 0.05,
        });
        assert!(matches!(embedding.validate(), Err(ModelError::OutputSemanticMismatch)));
        embedding.outputs = vec![OutputContract {
            name: "embedding".to_owned(),
            dimensions: vec![1, 128],
            element_type: TensorElementType::Float32,
            semantic: OutputSemantic::Embedding,
        }];
        embedding.outputs[0].dimensions = vec![1, 128];
        assert!(embedding.validate().is_ok());

        let mut passive = manifest();
        passive.role = ModelRole::PassiveLivenessInfrared;
        passive.outputs = vec![OutputContract {
            name: "live".to_owned(),
            dimensions: vec![1],
            element_type: TensorElementType::Float32,
            semantic: OutputSemantic::LiveProbability,
        }];
        assert!(passive.validate().is_ok());
    }

    #[test]
    fn embedding_alignment_is_required_unique_model_bound_and_non_degenerate() {
        let mut embedding = manifest();
        embedding.role = ModelRole::FaceEmbedding;
        embedding.outputs = vec![OutputContract {
            name: "embedding".to_owned(),
            dimensions: vec![1, 128],
            element_type: TensorElementType::Float32,
            semantic: OutputSemantic::Embedding,
        }];
        assert!(matches!(embedding.validate(), Err(ModelError::InvalidAlignmentContract)));

        let valid = FaceAlignmentContract {
            landmarks_compatibility_sha256:
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
            landmark_indices: [0, 1, 2, 3, 4],
            reference_points: [
                NormalizedPoint { x: 0.3, y: 0.35 },
                NormalizedPoint { x: 0.7, y: 0.35 },
                NormalizedPoint { x: 0.5, y: 0.52 },
                NormalizedPoint { x: 0.36, y: 0.72 },
                NormalizedPoint { x: 0.64, y: 0.72 },
            ],
            maximum_normalized_residual: 0.05,
        };
        embedding.face_alignment = Some(valid.clone());
        assert!(embedding.validate().is_ok());

        let mut duplicate = valid.clone();
        duplicate.landmark_indices[4] = 3;
        embedding.face_alignment = Some(duplicate);
        assert!(matches!(embedding.validate(), Err(ModelError::InvalidAlignmentContract)));
        let mut degenerate = valid.clone();
        degenerate.reference_points.fill(NormalizedPoint { x: 0.5, y: 0.5 });
        embedding.face_alignment = Some(degenerate);
        assert!(matches!(embedding.validate(), Err(ModelError::InvalidAlignmentContract)));
        let mut unbound = valid;
        unbound.landmarks_compatibility_sha256 = "bad".into();
        embedding.face_alignment = Some(unbound);
        assert!(matches!(embedding.validate(), Err(ModelError::InvalidAlignmentContract)));
    }

    #[test]
    fn detector_requires_exact_paired_single_batch_outputs() {
        let mut detector = manifest();
        assert!(detector.validate().is_ok());

        detector.outputs[1].dimensions = vec![1, 99];
        assert!(matches!(detector.validate(), Err(ModelError::OutputSemanticMismatch)));

        detector = manifest();
        detector.outputs[0].dimensions = vec![2, 100, 4];
        assert!(matches!(detector.validate(), Err(ModelError::OutputSemanticMismatch)));

        detector = manifest();
        detector.outputs[1].semantic = OutputSemantic::Auxiliary;
        assert!(matches!(detector.validate(), Err(ModelError::OutputSemanticMismatch)));

        detector = manifest();
        detector.outputs[0].dimensions = vec![1, MAX_FACE_DETECTION_CANDIDATES + 1, 4];
        detector.outputs[1].dimensions = vec![1, MAX_FACE_DETECTION_CANDIDATES + 1];
        assert!(matches!(detector.validate(), Err(ModelError::OutputSemanticMismatch)));
    }

    #[test]
    fn landmark_role_requires_bounded_geometry_and_measurement_outputs() {
        let mut landmarks = manifest();
        landmarks.role = ModelRole::FaceLandmarks;
        landmarks.face_crop =
            Some(FaceCropContract { scale: 1.25, center_offset_x: 0.0, center_offset_y: 0.1 });
        landmarks.landmark_measurement = Some(LandmarkMeasurementContract {
            minimum_visible_confidence: 0.5,
            maximum_absolute_yaw_degrees: 75.0,
            maximum_absolute_pitch_degrees: 60.0,
            maximum_absolute_roll_degrees: 90.0,
        });
        landmarks.outputs = landmark_outputs(5);
        assert!(landmarks.validate().is_ok());

        landmarks.outputs[0].dimensions = vec![1, 4, 2];
        assert!(matches!(landmarks.validate(), Err(ModelError::OutputSemanticMismatch)));
        landmarks.outputs[0].dimensions = vec![1, 5, 3];
        assert!(matches!(landmarks.validate(), Err(ModelError::OutputSemanticMismatch)));
        landmarks.outputs[0].dimensions = vec![5, 2];
        assert!(matches!(landmarks.validate(), Err(ModelError::OutputSemanticMismatch)));

        landmarks.outputs[0].dimensions = vec![1, 5, 2];
        landmarks.face_crop =
            Some(FaceCropContract { scale: 3.1, center_offset_x: 0.0, center_offset_y: 0.0 });
        assert!(matches!(landmarks.validate(), Err(ModelError::InvalidFaceCropContract)));

        landmarks.face_crop =
            Some(FaceCropContract { scale: 1.25, center_offset_x: 0.0, center_offset_y: 0.0 });
        landmarks.landmark_measurement = None;
        assert!(matches!(
            landmarks.validate(),
            Err(ModelError::InvalidLandmarkMeasurementContract)
        ));
    }

    #[test]
    fn compatibility_digest_covers_preprocessing_and_semantics() -> Result<(), ModelError> {
        let original = manifest();
        let original_digest = original.compatibility_sha256()?;

        let mut changed_normalization = original.clone();
        changed_normalization.input.normalization.bias[0] = 1.0;
        assert_ne!(changed_normalization.compatibility_sha256()?, original_digest);

        let mut changed_output = original;
        changed_output.outputs[0].name = "different".to_owned();
        assert_ne!(changed_output.compatibility_sha256()?, original_digest);

        let mut embedding = manifest();
        embedding.role = ModelRole::FaceEmbedding;
        embedding.outputs = vec![OutputContract {
            name: "embedding".to_owned(),
            dimensions: vec![1, 128],
            element_type: TensorElementType::Float32,
            semantic: OutputSemantic::Embedding,
        }];
        embedding.face_alignment = Some(FaceAlignmentContract {
            landmarks_compatibility_sha256:
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
            landmark_indices: [0, 1, 2, 3, 4],
            reference_points: [
                NormalizedPoint { x: 0.3, y: 0.35 },
                NormalizedPoint { x: 0.7, y: 0.35 },
                NormalizedPoint { x: 0.5, y: 0.52 },
                NormalizedPoint { x: 0.36, y: 0.72 },
                NormalizedPoint { x: 0.64, y: 0.72 },
            ],
            maximum_normalized_residual: 0.05,
        });
        let embedding_digest = embedding.compatibility_sha256()?;
        if let Some(alignment) = embedding.face_alignment.as_mut() {
            alignment.reference_points[0].x += 0.01;
        }
        assert_ne!(embedding.compatibility_sha256()?, embedding_digest);

        let mut landmarks = manifest();
        landmarks.role = ModelRole::FaceLandmarks;
        landmarks.outputs = landmark_outputs(5);
        landmarks.face_crop =
            Some(FaceCropContract { scale: 1.25, center_offset_x: 0.0, center_offset_y: 0.0 });
        landmarks.landmark_measurement = Some(LandmarkMeasurementContract {
            minimum_visible_confidence: 0.5,
            maximum_absolute_yaw_degrees: 75.0,
            maximum_absolute_pitch_degrees: 60.0,
            maximum_absolute_roll_degrees: 90.0,
        });
        let crop_digest = landmarks.compatibility_sha256()?;
        if let Some(crop) = landmarks.face_crop.as_mut() {
            crop.center_offset_y = 0.1;
        }
        assert_ne!(landmarks.compatibility_sha256()?, crop_digest);
        let measurement_digest = landmarks.compatibility_sha256()?;
        if let Some(measurement) = landmarks.landmark_measurement.as_mut() {
            measurement.minimum_visible_confidence = 0.6;
        }
        assert_ne!(landmarks.compatibility_sha256()?, measurement_digest);
        Ok(())
    }
}
