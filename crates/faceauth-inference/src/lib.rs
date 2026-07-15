//! Dynamically loaded, manifest-bound ONNX Runtime sessions.

use std::{
    fs,
    os::unix::fs::{MetadataExt, PermissionsExt},
    path::{Path, PathBuf},
    sync::Mutex,
};

use faceauth_model::{ModelManifest, TensorElementType as ManifestElementType};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::{TensorElementType, ValueType},
};
use thiserror::Error;

/// Default maximum admitted ONNX artifact size.
pub const DEFAULT_MAX_MODEL_BYTES: u64 = 512 * 1024 * 1024;

/// Absolute model-size ceiling.
pub const ABSOLUTE_MAX_MODEL_BYTES: u64 = 1024 * 1024 * 1024;

static RUNTIME_LIBRARY: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Explicit resource and dynamic-library policy for ONNX Runtime.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeConfig {
    /// Exact root-owned ONNX Runtime shared library.
    pub library_path: PathBuf,
    /// Sequential graph execution intra-op thread count.
    pub intra_threads: usize,
    /// Inter-op pool bound, retained as one because parallel graph execution is disabled.
    pub inter_threads: usize,
    /// Maximum admitted model artifact length.
    pub max_model_bytes: u64,
}

impl RuntimeConfig {
    /// Validate resource bounds before touching a runtime or model artifact.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::InvalidConfig`] for empty paths, zero/excessive threads, or an
    /// invalid model-size bound.
    pub fn validate(&self) -> Result<(), InferenceError> {
        if self.library_path.as_os_str().is_empty()
            || !(1..=16).contains(&self.intra_threads)
            || self.inter_threads != 1
            || !(1..=ABSOLUTE_MAX_MODEL_BYTES).contains(&self.max_model_bytes)
        {
            return Err(InferenceError::InvalidConfig);
        }
        Ok(())
    }
}

/// Loaded ONNX graph whose static I/O contract matches its reviewed manifest.
pub struct OnnxSession {
    session: Session,
}

impl OnnxSession {
    /// Verify runtime and model files, create a CPU session with bounded threading, and require an
    /// exact static float32 I/O contract.
    ///
    /// This crate never downloads models or runtime binaries.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError`] for unsafe files, digest/manifest failure, missing runtime,
    /// session creation failure, or any graph I/O mismatch.
    pub fn load(
        config: &RuntimeConfig,
        manifest: &ModelManifest,
        model_path: &Path,
    ) -> Result<Self, InferenceError> {
        config.validate()?;
        let canonical_model = fs::canonicalize(model_path)?;
        verify_trusted_file(&canonical_model, Some(config.max_model_bytes))?;
        manifest.verify_file(&canonical_model)?;
        initialize_runtime(&config.library_path)?;
        let mut builder = Session::builder()?
            .with_intra_threads(config.intra_threads)
            .map_err(ort::Error::from)?
            .with_inter_threads(config.inter_threads)
            .map_err(ort::Error::from)?
            .with_parallel_execution(false)
            .map_err(ort::Error::from)?
            .with_memory_pattern(true)
            .map_err(ort::Error::from)?
            .with_optimization_level(GraphOptimizationLevel::Level2)
            .map_err(ort::Error::from)?;
        let session = builder.commit_from_file(&canonical_model)?;
        validate_session_contract(manifest, &session)?;
        Ok(Self { session })
    }

    /// Borrow the underlying session for role-specific adapters.
    #[must_use]
    pub const fn session(&mut self) -> &mut Session {
        &mut self.session
    }
}

fn initialize_runtime(path: &Path) -> Result<(), InferenceError> {
    let canonical = fs::canonicalize(path)?;
    verify_trusted_file(&canonical, None)?;
    let mut configured =
        RUNTIME_LIBRARY.lock().map_err(|_| InferenceError::RuntimeStatePoisoned)?;
    if let Some(existing) = configured.as_ref() {
        return if existing == &canonical {
            Ok(())
        } else {
            Err(InferenceError::RuntimeAlreadyInitialized {
                existing: existing.clone(),
                requested: canonical,
            })
        };
    }
    let committed = ort::init_from(&canonical)?.with_name("faceauth").commit();
    if !committed {
        return Err(InferenceError::RuntimeAlreadyInitializedExternally);
    }
    *configured = Some(canonical);
    drop(configured);
    Ok(())
}

fn verify_trusted_file(path: &Path, max_bytes: Option<u64>) -> Result<(), InferenceError> {
    let metadata = fs::metadata(path)?;
    if !metadata.is_file() {
        return Err(InferenceError::NotRegularFile { path: path.to_owned() });
    }
    if metadata.uid() != 0 || metadata.permissions().mode() & 0o022 != 0 {
        return Err(InferenceError::UnsafeFilePermissions {
            path: path.to_owned(),
            owner_uid: metadata.uid(),
            mode: metadata.permissions().mode() & 0o777,
        });
    }
    for ancestor in path.ancestors().skip(1) {
        let ancestor_metadata = fs::metadata(ancestor)?;
        if !ancestor_metadata.is_dir()
            || ancestor_metadata.uid() != 0
            || ancestor_metadata.permissions().mode() & 0o022 != 0
        {
            return Err(InferenceError::UnsafeParentDirectory {
                path: ancestor.to_owned(),
                owner_uid: ancestor_metadata.uid(),
                mode: ancestor_metadata.permissions().mode() & 0o777,
            });
        }
    }
    if let Some(maximum) = max_bytes
        && (metadata.len() == 0 || metadata.len() > maximum)
    {
        return Err(InferenceError::ModelSizeInvalid { actual: metadata.len(), maximum });
    }
    Ok(())
}

fn validate_session_contract(
    manifest: &ModelManifest,
    session: &Session,
) -> Result<(), InferenceError> {
    if session.inputs().len() != 1 || session.outputs().len() != manifest.outputs.len() {
        return Err(InferenceError::GraphContractMismatch);
    }
    let input = &session.inputs()[0];
    let expected_input = manifest.input.dimensions().map(i64::from).to_vec();
    validate_outlet(
        input.name(),
        input.dtype(),
        &manifest.input.name,
        &expected_input,
        manifest.input.element_type,
    )?;
    for (actual, expected) in session.outputs().iter().zip(&manifest.outputs) {
        let expected_dimensions =
            expected.dimensions.iter().copied().map(i64::from).collect::<Vec<_>>();
        validate_outlet(
            actual.name(),
            actual.dtype(),
            &expected.name,
            &expected_dimensions,
            expected.element_type,
        )?;
    }
    Ok(())
}

fn validate_outlet(
    actual_name: &str,
    actual_type: &ValueType,
    expected_name: &str,
    expected_dimensions: &[i64],
    expected_type: ManifestElementType,
) -> Result<(), InferenceError> {
    let ValueType::Tensor { ty, shape, .. } = actual_type else {
        return Err(InferenceError::GraphContractMismatch);
    };
    validate_tensor_contract(
        actual_name,
        *ty,
        shape.as_ref(),
        expected_name,
        expected_dimensions,
        expected_type,
    )
}

fn validate_tensor_contract(
    actual_name: &str,
    actual_type: TensorElementType,
    actual_dimensions: &[i64],
    expected_name: &str,
    expected_dimensions: &[i64],
    expected_type: ManifestElementType,
) -> Result<(), InferenceError> {
    if actual_name != expected_name {
        return Err(InferenceError::GraphContractMismatch);
    }
    let element_matches = matches!(
        (expected_type, actual_type),
        (ManifestElementType::Float32, TensorElementType::Float32)
    );
    if !element_matches || actual_dimensions != expected_dimensions {
        return Err(InferenceError::GraphContractMismatch);
    }
    Ok(())
}

/// ONNX runtime or graph admission failure.
#[derive(Debug, Error)]
pub enum InferenceError {
    /// Runtime resource configuration is invalid.
    #[error("invalid ONNX Runtime configuration")]
    InvalidConfig,
    /// Runtime or model path is not a regular file.
    #[error("inference file is not regular: {path}")]
    NotRegularFile {
        /// Rejected path.
        path: PathBuf,
    },
    /// Runtime or model file ownership/permissions permit untrusted replacement.
    #[error("unsafe inference file {path}: owner {owner_uid}, mode {mode:o}")]
    UnsafeFilePermissions {
        /// Rejected path.
        path: PathBuf,
        /// Actual owner UID.
        owner_uid: u32,
        /// Actual permission bits.
        mode: u32,
    },
    /// A containing directory permits replacement of an otherwise trusted file.
    #[error("unsafe inference parent directory {path}: owner {owner_uid}, mode {mode:o}")]
    UnsafeParentDirectory {
        /// Rejected directory.
        path: PathBuf,
        /// Actual owner UID.
        owner_uid: u32,
        /// Actual permission bits.
        mode: u32,
    },
    /// Model file is empty or excessive.
    #[error("model size {actual} is invalid; maximum is {maximum}")]
    ModelSizeInvalid {
        /// Actual model length.
        actual: u64,
        /// Configured maximum length.
        maximum: u64,
    },
    /// A different ONNX Runtime library was already selected.
    #[error("ONNX Runtime already initialized from {existing}; cannot switch to {requested}")]
    RuntimeAlreadyInitialized {
        /// Existing canonical library path.
        existing: PathBuf,
        /// Requested canonical library path.
        requested: PathBuf,
    },
    /// Another crate initialized `ort` before the faceauth runtime policy.
    #[error("ONNX Runtime was initialized outside the faceauth runtime boundary")]
    RuntimeAlreadyInitializedExternally,
    /// Global runtime state mutex was poisoned.
    #[error("ONNX Runtime state is unavailable")]
    RuntimeStatePoisoned,
    /// Loaded graph does not exactly match the reviewed static manifest.
    #[error("ONNX graph input/output contract does not match its manifest")]
    GraphContractMismatch,
    /// Model manifest or digest verification failed.
    #[error("model admission failed: {0}")]
    Model(#[from] faceauth_model::ModelError),
    /// Filesystem operation failed.
    #[error("unable to inspect inference file: {0}")]
    Io(#[from] std::io::Error),
    /// ONNX Runtime loading or session operation failed.
    #[error("ONNX Runtime operation failed: {0}")]
    Ort(#[from] ort::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_configuration_is_bounded() {
        let valid = RuntimeConfig {
            library_path: PathBuf::from("/usr/lib/libonnxruntime.so"),
            intra_threads: 2,
            inter_threads: 1,
            max_model_bytes: DEFAULT_MAX_MODEL_BYTES,
        };
        assert!(valid.validate().is_ok());
        assert!(RuntimeConfig { intra_threads: 0, ..valid.clone() }.validate().is_err());
        assert!(RuntimeConfig { inter_threads: 2, ..valid }.validate().is_err());
    }

    #[test]
    fn missing_runtime_fails_without_downloading_anything() {
        let missing = PathBuf::from("/definitely-missing-faceauth/libonnxruntime.so");
        assert!(matches!(
            initialize_runtime(&missing),
            Err(InferenceError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound
        ));
    }

    #[test]
    fn graph_contract_rejects_names_shapes_and_types() {
        let dimensions = [1, 3, 112, 112];
        assert!(
            validate_tensor_contract(
                "input",
                TensorElementType::Float32,
                &dimensions,
                "input",
                &dimensions,
                ManifestElementType::Float32
            )
            .is_ok()
        );
        assert!(matches!(
            validate_tensor_contract(
                "wrong",
                TensorElementType::Float32,
                &dimensions,
                "input",
                &dimensions,
                ManifestElementType::Float32
            ),
            Err(InferenceError::GraphContractMismatch)
        ));
        assert!(matches!(
            validate_tensor_contract(
                "input",
                TensorElementType::Int64,
                &dimensions,
                "input",
                &dimensions,
                ManifestElementType::Float32
            ),
            Err(InferenceError::GraphContractMismatch)
        ));
        assert!(matches!(
            validate_tensor_contract(
                "input",
                TensorElementType::Float32,
                &[1, 3, 224, 224],
                "input",
                &dimensions,
                ManifestElementType::Float32
            ),
            Err(InferenceError::GraphContractMismatch)
        ));
    }
}
