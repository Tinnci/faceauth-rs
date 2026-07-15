//! Dynamically loaded, manifest-bound ONNX Runtime sessions.

use std::{
    fs,
    os::unix::fs::{MetadataExt, PermissionsExt},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, mpsc},
    thread,
    time::{Duration, Instant},
};

use faceauth_model::{
    ColorSpace, InputContract, ModelManifest, ModelRole, OutputSemantic, ResizeFilter,
    TensorElementType as ManifestElementType, TensorLayout,
};
use ort::{
    session::{RunOptions, Session, builder::GraphOptimizationLevel},
    value::{TensorElementType, TensorRef, ValueType},
};
use thiserror::Error;
use zeroize::Zeroizing;

/// Default maximum admitted ONNX artifact size.
pub const DEFAULT_MAX_MODEL_BYTES: u64 = 512 * 1024 * 1024;

/// Absolute model-size ceiling.
pub const ABSOLUTE_MAX_MODEL_BYTES: u64 = 1024 * 1024 * 1024;

/// Maximum source image dimension accepted before preprocessing.
pub const MAX_SOURCE_DIMENSION: u32 = 4096;

/// Absolute watchdog deadline for one ONNX Runtime call.
pub const ABSOLUTE_MAX_RUN_MILLIS: u32 = 5_000;

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
    /// Watchdog deadline for one graph execution.
    pub max_run_millis: u32,
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
            || !(1..=ABSOLUTE_MAX_RUN_MILLIS).contains(&self.max_run_millis)
        {
            return Err(InferenceError::InvalidConfig);
        }
        Ok(())
    }
}

/// Loaded ONNX graph whose static I/O contract matches its reviewed manifest.
pub struct OnnxSession {
    session: Session,
    manifest: ModelManifest,
    compatibility_sha256: String,
    max_run_millis: u32,
}

/// Borrowed, tightly packed 8-bit image supplied to deterministic preprocessing.
#[derive(Clone, Copy, Debug)]
pub struct ImageView<'a> {
    /// Source width in pixels.
    pub width: u32,
    /// Source height in pixels.
    pub height: u32,
    /// Source pixel encoding.
    pub format: ImageFormat,
    /// Exact tightly packed pixel bytes; no row padding is admitted.
    pub bytes: &'a [u8],
}

/// Raw pixel encodings accepted by the inference preprocessing boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageFormat {
    /// One intensity byte per pixel.
    Gray8,
    /// Three bytes per pixel in red, green, blue order.
    Rgb8,
    /// Three bytes per pixel in blue, green, red order.
    Bgr8,
    /// Packed YUYV 4:2:2 converted with bounded BT.601 limited-range arithmetic.
    Yuyv,
}

/// Zeroizing float32 input with the exact manifest shape.
pub struct InputTensor {
    dimensions: [u32; 4],
    values: Zeroizing<Vec<f32>>,
}

impl InputTensor {
    /// Exact tensor dimensions.
    #[must_use]
    pub const fn dimensions(&self) -> &[u32; 4] {
        &self.dimensions
    }

    /// Borrow ephemeral tensor values.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

/// One copied and zeroizing output tensor from a bounded inference call.
pub struct OutputTensor {
    name: String,
    dimensions: Vec<u32>,
    values: Zeroizing<Vec<f32>>,
}

/// L2-normalized face embedding bound to one complete model compatibility contract.
pub struct FaceEmbedding {
    compatibility_sha256: String,
    values: Zeroizing<Vec<f32>>,
}

impl FaceEmbedding {
    /// Reconstruct a comparison embedding from an already validated encrypted template.
    ///
    /// The values must already be L2-normalized; this function does not silently renormalize
    /// persisted data because that could hide corruption or a migration mismatch.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::EmbeddingInvalid`] for an invalid digest, dimension, numeric
    /// value, or unit norm.
    pub fn from_normalized_template(
        compatibility_sha256: &str,
        values: &[f32],
    ) -> Result<Self, InferenceError> {
        if compatibility_sha256.len() != 64
            || !compatibility_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
            || compatibility_sha256.bytes().any(|byte| byte.is_ascii_uppercase())
            || !(32..=4096).contains(&values.len())
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(InferenceError::EmbeddingInvalid);
        }
        let copied = Zeroizing::new(values.to_vec());
        let squared_norm = copied.iter().try_fold(0.0_f32, |sum, value| {
            let next = value.mul_add(*value, sum);
            next.is_finite().then_some(next).ok_or(InferenceError::EmbeddingInvalid)
        })?;
        if (squared_norm.sqrt() - 1.0).abs() > 1.0e-3 {
            return Err(InferenceError::EmbeddingInvalid);
        }
        Ok(Self { compatibility_sha256: compatibility_sha256.to_owned(), values: copied })
    }

    /// Complete model and preprocessing compatibility digest.
    #[must_use]
    pub fn compatibility_sha256(&self) -> &str {
        &self.compatibility_sha256
    }

    /// Borrow the normalized embedding for encrypted template persistence.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Compute mapped cosine similarity in the inclusive range 0..=1.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::EmbeddingIncompatible`] when dimensions or compatibility
    /// identities differ, or [`InferenceError::EmbeddingInvalid`] for invalid arithmetic.
    pub fn similarity(&self, other: &Self) -> Result<f32, InferenceError> {
        if self.compatibility_sha256 != other.compatibility_sha256
            || self.values.len() != other.values.len()
        {
            return Err(InferenceError::EmbeddingIncompatible);
        }
        let cosine = self
            .values
            .iter()
            .zip(other.values.iter())
            .try_fold(0.0_f32, |sum, (left, right)| {
                let next = left.mul_add(*right, sum);
                next.is_finite().then_some(next).ok_or(InferenceError::EmbeddingInvalid)
            })?
            .clamp(-1.0, 1.0);
        let mapped = cosine.mul_add(0.5, 0.5);
        mapped.is_finite().then_some(mapped).ok_or(InferenceError::EmbeddingInvalid)
    }
}

/// Scalar live probability bound to one passive-PAD model and preprocessing contract.
#[derive(Clone, Debug, PartialEq)]
pub struct PassiveLivenessScore {
    role: ModelRole,
    compatibility_sha256: String,
    probability: f32,
}

impl PassiveLivenessScore {
    /// Passive liveness model role that produced this score.
    #[must_use]
    pub const fn role(&self) -> ModelRole {
        self.role
    }

    /// Complete model and preprocessing compatibility digest.
    #[must_use]
    pub fn compatibility_sha256(&self) -> &str {
        &self.compatibility_sha256
    }

    /// Calibrated live probability in 0..=1.
    #[must_use]
    pub const fn probability(&self) -> f32 {
        self.probability
    }

    /// Apply an explicitly calibrated inclusive threshold.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::InvalidProbabilityThreshold`] for a non-finite or out-of-range
    /// threshold.
    pub fn passes(&self, minimum: f32) -> Result<bool, InferenceError> {
        if !minimum.is_finite() || !(0.0..=1.0).contains(&minimum) {
            return Err(InferenceError::InvalidProbabilityThreshold);
        }
        Ok(self.probability >= minimum)
    }
}

impl OutputTensor {
    /// Exact manifest output name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Exact output dimensions.
    #[must_use]
    pub fn dimensions(&self) -> &[u32] {
        &self.dimensions
    }

    /// Borrow ephemeral output values.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }
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
        Ok(Self {
            session,
            manifest: manifest.clone(),
            compatibility_sha256: manifest.compatibility_sha256()?,
            max_run_millis: config.max_run_millis,
        })
    }

    /// Convert a tightly packed 8-bit image into the exact manifest tensor.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError`] for malformed source buffers, unsupported color conversion,
    /// excessive dimensions, or non-finite preprocessing results.
    pub fn preprocess(&self, image: ImageView<'_>) -> Result<InputTensor, InferenceError> {
        preprocess_image(&self.manifest.input, image)
    }

    /// Execute one already validated tensor and copy all exact float32 outputs into zeroizing
    /// buffers before returning.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError`] when the input differs from the loaded manifest, ONNX Runtime
    /// fails, or an output is malformed or non-finite.
    pub fn run(&mut self, input: &InputTensor) -> Result<Vec<OutputTensor>, InferenceError> {
        if input.dimensions != self.manifest.input.dimensions()
            || input.values.iter().any(|value| !value.is_finite())
        {
            return Err(InferenceError::InputTensorInvalid);
        }
        let shape = input.dimensions.map(i64::from);
        let tensor = TensorRef::from_array_view((shape, input.values.as_slice()))?;
        let run_options = Arc::new(RunOptions::new()?);
        let watchdog_options = Arc::clone(&run_options);
        let deadline = Duration::from_millis(u64::from(self.max_run_millis));
        let started = Instant::now();
        let (completion_sender, completion_receiver) = mpsc::channel();
        let watchdog =
            thread::Builder::new().name("faceauth-ort-watchdog".to_owned()).spawn(move || {
                match completion_receiver.recv_timeout(deadline) {
                    Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => false,
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        let _termination_result = watchdog_options.terminate();
                        true
                    }
                }
            })?;
        let run_result = self.session.run_with_options(ort::inputs![tensor], &run_options);
        let _completion_result = completion_sender.send(());
        let watchdog_expired = watchdog.join().map_err(|_| InferenceError::WatchdogPanicked)?;
        if watchdog_expired || started.elapsed() >= deadline {
            return Err(InferenceError::RunDeadlineExceeded);
        }
        let outputs = match run_result {
            Ok(outputs) => outputs,
            Err(error) => return Err(InferenceError::Ort(error)),
        };
        let mut copied = Vec::with_capacity(self.manifest.outputs.len());
        for expected in &self.manifest.outputs {
            let value = outputs.get(&expected.name).ok_or(InferenceError::GraphContractMismatch)?;
            let (shape, values) = value.try_extract_tensor::<f32>()?;
            let actual_dimensions = shape
                .iter()
                .map(|dimension| u32::try_from(*dimension))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| InferenceError::OutputTensorInvalid)?;
            if actual_dimensions != expected.dimensions
                || values.len() != tensor_element_count(&expected.dimensions)?
                || values.iter().any(|value| !value.is_finite())
            {
                return Err(InferenceError::OutputTensorInvalid);
            }
            copied.push(OutputTensor {
                name: expected.name.clone(),
                dimensions: actual_dimensions,
                values: Zeroizing::new(values.to_vec()),
            });
        }
        Ok(copied)
    }

    /// Convert the semantic embedding output into a normalized, compatibility-bound template.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError`] when the loaded role, output identity, dimensions, values, or
    /// vector norm are invalid.
    pub fn extract_embedding(
        &self,
        outputs: &[OutputTensor],
    ) -> Result<FaceEmbedding, InferenceError> {
        extract_embedding_output(&self.manifest, &self.compatibility_sha256, outputs)
    }

    /// Convert a semantic passive-PAD output into a compatibility-bound live probability.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError`] when the loaded role, output identity, shape, or probability is
    /// invalid.
    pub fn extract_passive_liveness(
        &self,
        outputs: &[OutputTensor],
    ) -> Result<PassiveLivenessScore, InferenceError> {
        extract_passive_liveness_output(&self.manifest, &self.compatibility_sha256, outputs)
    }
}

fn extract_embedding_output(
    manifest: &ModelManifest,
    compatibility_sha256: &str,
    outputs: &[OutputTensor],
) -> Result<FaceEmbedding, InferenceError> {
    if manifest.role != ModelRole::FaceEmbedding {
        return Err(InferenceError::ModelRoleMismatch);
    }
    let contract = semantic_output(manifest, OutputSemantic::Embedding)?;
    let output = exact_output(outputs, &contract.name, &contract.dimensions)?;
    if !(32..=4096).contains(&output.values.len()) {
        return Err(InferenceError::EmbeddingInvalid);
    }
    let squared_norm = output.values.iter().try_fold(0.0_f32, |sum, value| {
        let next = value.mul_add(*value, sum);
        next.is_finite().then_some(next).ok_or(InferenceError::EmbeddingInvalid)
    })?;
    let norm = squared_norm.sqrt();
    if !norm.is_finite() || norm <= f32::EPSILON {
        return Err(InferenceError::EmbeddingInvalid);
    }
    let values =
        Zeroizing::new(output.values.iter().map(|value| value / norm).collect::<Vec<f32>>());
    if values.iter().any(|value| !value.is_finite()) {
        return Err(InferenceError::EmbeddingInvalid);
    }
    Ok(FaceEmbedding { compatibility_sha256: compatibility_sha256.to_owned(), values })
}

fn extract_passive_liveness_output(
    manifest: &ModelManifest,
    compatibility_sha256: &str,
    outputs: &[OutputTensor],
) -> Result<PassiveLivenessScore, InferenceError> {
    if !matches!(
        manifest.role,
        ModelRole::PassiveLivenessInfrared
            | ModelRole::PassiveLivenessVisible
            | ModelRole::PassiveLivenessFusion
    ) {
        return Err(InferenceError::ModelRoleMismatch);
    }
    let contract = semantic_output(manifest, OutputSemantic::LiveProbability)?;
    let output = exact_output(outputs, &contract.name, &contract.dimensions)?;
    let [probability] = output.values.as_slice() else {
        return Err(InferenceError::PassiveLivenessOutputInvalid);
    };
    if !probability.is_finite() || !(0.0..=1.0).contains(probability) {
        return Err(InferenceError::PassiveLivenessOutputInvalid);
    }
    Ok(PassiveLivenessScore {
        role: manifest.role,
        compatibility_sha256: compatibility_sha256.to_owned(),
        probability: *probability,
    })
}

fn semantic_output(
    manifest: &ModelManifest,
    semantic: OutputSemantic,
) -> Result<&faceauth_model::OutputContract, InferenceError> {
    let mut matching = manifest.outputs.iter().filter(|output| output.semantic == semantic);
    let output = matching.next().ok_or(InferenceError::ModelRoleMismatch)?;
    if matching.next().is_some() {
        return Err(InferenceError::ModelRoleMismatch);
    }
    Ok(output)
}

fn exact_output<'a>(
    outputs: &'a [OutputTensor],
    name: &str,
    dimensions: &[u32],
) -> Result<&'a OutputTensor, InferenceError> {
    let mut matching = outputs.iter().filter(|output| output.name == name);
    let output = matching.next().ok_or(InferenceError::OutputTensorInvalid)?;
    if matching.next().is_some() || output.dimensions != dimensions {
        Err(InferenceError::OutputTensorInvalid)
    } else {
        Ok(output)
    }
}

fn preprocess_image(
    contract: &InputContract,
    image: ImageView<'_>,
) -> Result<InputTensor, InferenceError> {
    let channels = usize::from(contract.channels);
    if contract.width == 0
        || contract.height == 0
        || contract.normalization.scale.len() != channels
        || contract.normalization.bias.len() != channels
    {
        return Err(InferenceError::InputTensorInvalid);
    }
    if image.width == 0
        || image.height == 0
        || image.width > MAX_SOURCE_DIMENSION
        || image.height > MAX_SOURCE_DIMENSION
    {
        return Err(InferenceError::SourceImageInvalid);
    }
    let source_pixels = usize::try_from(u64::from(image.width) * u64::from(image.height))
        .map_err(|_| InferenceError::SourceImageInvalid)?;
    let expected_bytes = match image.format {
        ImageFormat::Gray8 => source_pixels,
        ImageFormat::Rgb8 | ImageFormat::Bgr8 => {
            source_pixels.checked_mul(3).ok_or(InferenceError::SourceImageInvalid)?
        }
        ImageFormat::Yuyv => {
            if !image.width.is_multiple_of(2) {
                return Err(InferenceError::SourceImageInvalid);
            }
            source_pixels.checked_mul(2).ok_or(InferenceError::SourceImageInvalid)?
        }
    };
    if image.bytes.len() != expected_bytes
        || matches!(contract.color_space, ColorSpace::Grayscale)
            != matches!(image.format, ImageFormat::Gray8)
    {
        return Err(InferenceError::SourceImageInvalid);
    }

    let dimensions = contract.dimensions();
    let element_count = tensor_element_count(&dimensions)?;
    let mut values = Zeroizing::new(vec![0.0_f32; element_count]);
    let target_width = contract.width;
    let target_height = contract.height;
    match contract.resize_filter {
        ResizeFilter::BilinearHalfPixel => {}
    }
    for target_y in 0..target_height {
        let (y0, y1, wy) = interpolation_axis(target_y, target_height, image.height);
        for target_x in 0..target_width {
            let (x0, x1, wx) = interpolation_axis(target_x, target_width, image.width);
            for channel in 0..channels {
                let top_left = source_channel(image, x0, y0, channel, contract.color_space)?;
                let top_right = source_channel(image, x1, y0, channel, contract.color_space)?;
                let bottom_left = source_channel(image, x0, y1, channel, contract.color_space)?;
                let bottom_right = source_channel(image, x1, y1, channel, contract.color_space)?;
                let top = top_left.mul_add(1.0 - wx, top_right * wx);
                let bottom = bottom_left.mul_add(1.0 - wx, bottom_right * wx);
                let pixel = top.mul_add(1.0 - wy, bottom * wy);
                let normalized = pixel.mul_add(
                    contract.normalization.scale[channel],
                    contract.normalization.bias[channel],
                );
                if !normalized.is_finite() {
                    return Err(InferenceError::InputTensorInvalid);
                }
                let target_index = match contract.layout {
                    TensorLayout::Nchw => channel
                        .checked_mul(
                            usize::try_from(target_width * target_height)
                                .map_err(|_| InferenceError::InputTensorInvalid)?,
                        )
                        .and_then(|base| {
                            base.checked_add(
                                usize::try_from(target_y * target_width + target_x).ok()?,
                            )
                        }),
                    TensorLayout::Nhwc => usize::try_from(target_y * target_width + target_x)
                        .ok()
                        .and_then(|pixel_index| pixel_index.checked_mul(channels))
                        .and_then(|base| base.checked_add(channel)),
                }
                .ok_or(InferenceError::InputTensorInvalid)?;
                values[target_index] = normalized;
            }
        }
    }
    Ok(InputTensor { dimensions, values })
}

fn interpolation_axis(target: u32, target_extent: u32, source_extent: u32) -> (u32, u32, f32) {
    let denominator = i64::from(target_extent) * 2;
    let numerator =
        (i64::from(target) * 2 + 1) * i64::from(source_extent) - i64::from(target_extent);
    let raw_lower = numerator.div_euclid(denominator);
    if raw_lower < 0 {
        return (0, 0, 0.0);
    }
    let lower = u32::try_from(raw_lower).unwrap_or(source_extent - 1).min(source_extent - 1);
    if lower == source_extent - 1 {
        return (lower, lower, 0.0);
    }
    let upper = lower + 1;
    let remainder = u16::try_from(numerator.rem_euclid(denominator)).unwrap_or_default();
    let denominator = u16::try_from(denominator).unwrap_or(1);
    let weight = f32::from(remainder) / f32::from(denominator);
    (lower, upper, weight)
}

fn source_channel(
    image: ImageView<'_>,
    x: u32,
    y: u32,
    channel: usize,
    target_color: ColorSpace,
) -> Result<f32, InferenceError> {
    let pixel_index = usize::try_from(u64::from(y) * u64::from(image.width) + u64::from(x))
        .map_err(|_| InferenceError::SourceImageInvalid)?;
    let rgb = match image.format {
        ImageFormat::Gray8 => {
            return image
                .bytes
                .get(pixel_index)
                .copied()
                .map(f32::from)
                .ok_or(InferenceError::SourceImageInvalid);
        }
        ImageFormat::Rgb8 | ImageFormat::Bgr8 => {
            let base = pixel_index.checked_mul(3).ok_or(InferenceError::SourceImageInvalid)?;
            let first = *image.bytes.get(base).ok_or(InferenceError::SourceImageInvalid)?;
            let green = *image.bytes.get(base + 1).ok_or(InferenceError::SourceImageInvalid)?;
            let third = *image.bytes.get(base + 2).ok_or(InferenceError::SourceImageInvalid)?;
            if image.format == ImageFormat::Rgb8 {
                [first, green, third]
            } else {
                [third, green, first]
            }
        }
        ImageFormat::Yuyv => yuyv_rgb(image, x, y)?,
    };
    let ordered = match target_color {
        ColorSpace::Rgb => rgb,
        ColorSpace::Bgr => [rgb[2], rgb[1], rgb[0]],
        ColorSpace::Grayscale => return Err(InferenceError::SourceImageInvalid),
    };
    ordered.get(channel).copied().map(f32::from).ok_or(InferenceError::SourceImageInvalid)
}

fn yuyv_rgb(image: ImageView<'_>, column: u32, row: u32) -> Result<[u8; 3], InferenceError> {
    let pair = u64::from(row) * u64::from(image.width / 2) + u64::from(column / 2);
    let base = usize::try_from(pair.checked_mul(4).ok_or(InferenceError::SourceImageInvalid)?)
        .map_err(|_| InferenceError::SourceImageInvalid)?;
    let luma = if column.is_multiple_of(2) {
        *image.bytes.get(base).ok_or(InferenceError::SourceImageInvalid)?
    } else {
        *image.bytes.get(base + 2).ok_or(InferenceError::SourceImageInvalid)?
    };
    let chroma_blue = *image.bytes.get(base + 1).ok_or(InferenceError::SourceImageInvalid)?;
    let chroma_red = *image.bytes.get(base + 3).ok_or(InferenceError::SourceImageInvalid)?;
    let scaled_luma = i32::from(luma).saturating_sub(16).max(0);
    let blue_delta = i32::from(chroma_blue) - 128;
    let red_delta = i32::from(chroma_red) - 128;
    Ok([
        clamp_u8((298 * scaled_luma + 409 * red_delta + 128) >> 8),
        clamp_u8((298 * scaled_luma - 100 * blue_delta - 208 * red_delta + 128) >> 8),
        clamp_u8((298 * scaled_luma + 516 * blue_delta + 128) >> 8),
    ])
}

fn clamp_u8(value: i32) -> u8 {
    u8::try_from(value.clamp(0, 255)).unwrap_or_default()
}

fn tensor_element_count(dimensions: &[u32]) -> Result<usize, InferenceError> {
    dimensions
        .iter()
        .try_fold(1_usize, |product, dimension| {
            product.checked_mul(usize::try_from(*dimension).ok()?)
        })
        .ok_or(InferenceError::InputTensorInvalid)
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
    /// Source image dimensions, encoding, or byte length are invalid.
    #[error("source image is malformed or incompatible with the model color contract")]
    SourceImageInvalid,
    /// Preprocessed input shape or values violate the loaded manifest.
    #[error("inference input tensor is invalid")]
    InputTensorInvalid,
    /// Runtime output shape, count, type, or values violate the loaded manifest.
    #[error("inference output tensor is invalid")]
    OutputTensorInvalid,
    /// ONNX Runtime exceeded the configured watchdog deadline and was terminated.
    #[error("inference run exceeded its watchdog deadline")]
    RunDeadlineExceeded,
    /// The inference deadline watchdog terminated unexpectedly.
    #[error("inference watchdog thread terminated unexpectedly")]
    WatchdogPanicked,
    /// A role-specific adapter was requested from a different model role.
    #[error("model role does not match the requested biometric adapter")]
    ModelRoleMismatch,
    /// Face embedding is empty, excessive, degenerate, or numerically invalid.
    #[error("face embedding output is invalid")]
    EmbeddingInvalid,
    /// Stored and observed embeddings were produced by incompatible contracts.
    #[error("face embeddings are incompatible")]
    EmbeddingIncompatible,
    /// Passive liveness output is not one finite scalar probability in 0..=1.
    #[error("passive liveness output is invalid")]
    PassiveLivenessOutputInvalid,
    /// Passive liveness threshold is not finite or outside 0..=1.
    #[error("passive liveness threshold must be finite and in 0..=1")]
    InvalidProbabilityThreshold,
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
    use faceauth_model::{
        ChannelNormalization, MODEL_MANIFEST_SCHEMA_VERSION, OutputContract, ResizeFilter,
    };

    fn input_contract(
        width: u32,
        height: u32,
        layout: TensorLayout,
        color_space: ColorSpace,
    ) -> InputContract {
        let channels = if color_space == ColorSpace::Grayscale { 1 } else { 3 };
        InputContract {
            name: "input".to_owned(),
            width,
            height,
            channels,
            layout,
            color_space,
            element_type: ManifestElementType::Float32,
            resize_filter: ResizeFilter::BilinearHalfPixel,
            normalization: ChannelNormalization {
                scale: vec![1.0 / 255.0; usize::from(channels)],
                bias: vec![0.0; usize::from(channels)],
            },
        }
    }

    fn model_manifest(
        role: ModelRole,
        semantic: OutputSemantic,
        dimensions: Vec<u32>,
    ) -> ModelManifest {
        ModelManifest {
            schema_version: MODEL_MANIFEST_SCHEMA_VERSION,
            name: "test-model".to_owned(),
            version: "1".to_owned(),
            role,
            source_url: "https://example.invalid/model".to_owned(),
            license_spdx: "Apache-2.0".to_owned(),
            sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
            input: input_contract(1, 1, TensorLayout::Nchw, ColorSpace::Rgb),
            outputs: vec![OutputContract {
                name: "output".to_owned(),
                dimensions,
                element_type: ManifestElementType::Float32,
                semantic,
            }],
        }
    }

    #[test]
    fn runtime_configuration_is_bounded() {
        let valid = RuntimeConfig {
            library_path: PathBuf::from("/usr/lib/libonnxruntime.so"),
            intra_threads: 2,
            inter_threads: 1,
            max_model_bytes: DEFAULT_MAX_MODEL_BYTES,
            max_run_millis: 1_000,
        };
        assert!(valid.validate().is_ok());
        assert!(RuntimeConfig { intra_threads: 0, ..valid.clone() }.validate().is_err());
        assert!(RuntimeConfig { max_run_millis: 0, ..valid.clone() }.validate().is_err());
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

    #[test]
    fn grayscale_preprocessing_resizes_and_normalizes_deterministically()
    -> Result<(), InferenceError> {
        let contract = input_contract(1, 1, TensorLayout::Nchw, ColorSpace::Grayscale);
        let tensor = preprocess_image(
            &contract,
            ImageView {
                width: 2,
                height: 2,
                format: ImageFormat::Gray8,
                bytes: &[0, 64, 128, 255],
            },
        )?;
        assert_eq!(tensor.dimensions(), &[1, 1, 1, 1]);
        assert!((tensor.values()[0] - (111.75 / 255.0)).abs() < 1.0e-6);
        Ok(())
    }

    #[test]
    fn rgb_preprocessing_honors_nchw_and_bgr_contract_order() -> Result<(), InferenceError> {
        let contract = input_contract(1, 1, TensorLayout::Nchw, ColorSpace::Bgr);
        let tensor = preprocess_image(
            &contract,
            ImageView { width: 1, height: 1, format: ImageFormat::Rgb8, bytes: &[10, 20, 30] },
        )?;
        let expected = [30.0 / 255.0, 20.0 / 255.0, 10.0 / 255.0];
        for (actual, expected) in tensor.values().iter().zip(expected) {
            assert!((*actual - expected).abs() < 1.0e-6);
        }
        Ok(())
    }

    #[test]
    fn yuyv_conversion_and_source_bounds_fail_closed() -> Result<(), InferenceError> {
        let contract = input_contract(2, 1, TensorLayout::Nhwc, ColorSpace::Rgb);
        let black = preprocess_image(
            &contract,
            ImageView {
                width: 2,
                height: 1,
                format: ImageFormat::Yuyv,
                bytes: &[16, 128, 16, 128],
            },
        )?;
        assert!(black.values().iter().all(|value| *value == 0.0));

        assert!(matches!(
            preprocess_image(
                &contract,
                ImageView { width: 3, height: 1, format: ImageFormat::Yuyv, bytes: &[0; 6] }
            ),
            Err(InferenceError::SourceImageInvalid)
        ));
        assert!(matches!(
            preprocess_image(
                &contract,
                ImageView { width: 2, height: 1, format: ImageFormat::Yuyv, bytes: &[0; 3] }
            ),
            Err(InferenceError::SourceImageInvalid)
        ));
        Ok(())
    }

    #[test]
    fn embedding_adapter_normalizes_and_binds_compatibility() -> Result<(), InferenceError> {
        let manifest =
            model_manifest(ModelRole::FaceEmbedding, OutputSemantic::Embedding, vec![1, 32]);
        let outputs = vec![OutputTensor {
            name: "output".to_owned(),
            dimensions: vec![1, 32],
            values: Zeroizing::new(vec![1.0; 32]),
        }];
        let first = extract_embedding_output(&manifest, "compatibility-a", &outputs)?;
        let second = extract_embedding_output(&manifest, "compatibility-a", &outputs)?;
        assert!(
            (first.values().iter().map(|value| value * value).sum::<f32>() - 1.0).abs() < 1.0e-5
        );
        assert!((first.similarity(&second)? - 1.0).abs() < 1.0e-6);

        let incompatible = extract_embedding_output(&manifest, "compatibility-b", &outputs)?;
        assert!(matches!(
            first.similarity(&incompatible),
            Err(InferenceError::EmbeddingIncompatible)
        ));
        Ok(())
    }

    #[test]
    fn embedding_adapter_rejects_degenerate_vectors() {
        let manifest =
            model_manifest(ModelRole::FaceEmbedding, OutputSemantic::Embedding, vec![1, 32]);
        let outputs = vec![OutputTensor {
            name: "output".to_owned(),
            dimensions: vec![1, 32],
            values: Zeroizing::new(vec![0.0; 32]),
        }];
        assert!(matches!(
            extract_embedding_output(&manifest, "compatibility", &outputs),
            Err(InferenceError::EmbeddingInvalid)
        ));
    }

    #[test]
    fn stored_embedding_requires_unit_norm_and_digest() -> Result<(), InferenceError> {
        let digest = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let mut normalized = vec![0.0; 32];
        normalized[0] = 1.0;
        let restored = FaceEmbedding::from_normalized_template(digest, &normalized)?;
        assert_eq!(restored.compatibility_sha256(), digest);

        normalized[0] = 0.5;
        assert!(matches!(
            FaceEmbedding::from_normalized_template(digest, &normalized),
            Err(InferenceError::EmbeddingInvalid)
        ));
        assert!(matches!(
            FaceEmbedding::from_normalized_template("not-a-digest", &[1.0; 32]),
            Err(InferenceError::EmbeddingInvalid)
        ));
        Ok(())
    }

    #[test]
    fn passive_liveness_adapter_requires_scalar_probability() -> Result<(), InferenceError> {
        let manifest = model_manifest(
            ModelRole::PassiveLivenessInfrared,
            OutputSemantic::LiveProbability,
            vec![1],
        );
        let valid = vec![OutputTensor {
            name: "output".to_owned(),
            dimensions: vec![1],
            values: Zeroizing::new(vec![0.8]),
        }];
        let score = extract_passive_liveness_output(&manifest, "compatibility", &valid)?;
        assert!(score.passes(0.75)?);
        assert!(!score.passes(0.85)?);

        let invalid = vec![OutputTensor {
            name: "output".to_owned(),
            dimensions: vec![1],
            values: Zeroizing::new(vec![1.1]),
        }];
        assert!(matches!(
            extract_passive_liveness_output(&manifest, "compatibility", &invalid),
            Err(InferenceError::PassiveLivenessOutputInvalid)
        ));
        Ok(())
    }
}
