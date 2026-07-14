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
pub const MODEL_MANIFEST_SCHEMA_VERSION: u16 = 1;

/// Maximum artifact name or version length.
pub const MAX_ARTIFACT_LABEL_LENGTH: usize = 128;

/// Maximum source URL length.
pub const MAX_SOURCE_URL_LENGTH: usize = 2048;

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

/// Static input tensor contract recorded with a model.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InputContract {
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
}

/// Auditable metadata required before a model can enter the inference pipeline.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
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
        if self.input.width == 0 || self.input.height == 0 {
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
    /// Width and height must both be positive.
    #[error("model input dimensions must be positive")]
    InvalidInputDimensions,
    /// Channel count does not match the declared color space.
    #[error("model input has {actual} channels; declared color space requires {expected}")]
    InvalidChannelCount {
        /// Required channel count.
        expected: u8,
        /// Manifest channel count.
        actual: u8,
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
                width: 640,
                height: 360,
                channels: 3,
                layout: TensorLayout::Nchw,
                color_space: ColorSpace::Rgb,
            },
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
}
