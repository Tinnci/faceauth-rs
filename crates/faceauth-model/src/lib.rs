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
pub const MODEL_MANIFEST_SCHEMA_VERSION: u16 = 3;

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
        if self.sha256.len() != 64
            || !self.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
            || self.sha256.bytes().any(|byte| byte.is_ascii_uppercase())
        {
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
        Ok(())
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
            outputs: vec![OutputContract {
                name: "boxes".to_owned(),
                dimensions: vec![1, 100, 4],
                element_type: TensorElementType::Float32,
            }],
        }
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
}
