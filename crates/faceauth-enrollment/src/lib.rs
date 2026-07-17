//! Bounded multi-observation enrollment that persists only a derived face template.

use faceauth_inference::{FaceEmbedding, InferenceError};
use faceauth_storage::{StorageError, TEMPLATE_RECORD_SCHEMA_VERSION, TemplateRecord};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use zeroize::Zeroizing;

/// Absolute maximum number of embeddings retained during one enrollment.
pub const MAX_ENROLLMENT_SAMPLES: u8 = 16;

/// Calibrated bounds for one enrollment transaction.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EnrollmentConfig {
    /// Fixed transaction duration measured using monotonic timestamps.
    pub duration_micros: u64,
    /// Minimum accepted observations required to produce a template.
    pub minimum_samples: u8,
    /// Maximum observations retained before the transaction fails closed.
    pub maximum_samples: u8,
    /// Minimum calibrated image-quality score in 0..=1.
    pub minimum_quality: f32,
    /// Minimum mapped cosine similarity between every accepted sample pair.
    pub minimum_sample_similarity: f32,
    /// Minimum monotonic spacing between accepted samples to reject near-duplicate frames.
    pub minimum_sample_interval_micros: u64,
    /// Minimum yaw range covered by the accepted sample set before template creation.
    pub minimum_yaw_span_degrees: f32,
}

impl EnrollmentConfig {
    /// Validate resource, timing, and calibrated score bounds.
    ///
    /// # Errors
    ///
    /// Returns [`EnrollmentError::InvalidConfig`] for non-finite, contradictory, or excessive
    /// settings.
    pub fn validate(self) -> Result<(), EnrollmentError> {
        if !(5_000_000..=120_000_000).contains(&self.duration_micros)
            || !(3..=MAX_ENROLLMENT_SAMPLES).contains(&self.minimum_samples)
            || self.maximum_samples < self.minimum_samples
            || self.maximum_samples > MAX_ENROLLMENT_SAMPLES
            || !valid_unit_score(self.minimum_quality)
            || !valid_unit_score(self.minimum_sample_similarity)
            || !(50_000..=5_000_000).contains(&self.minimum_sample_interval_micros)
            || !self.minimum_yaw_span_degrees.is_finite()
            || !(5.0..=60.0).contains(&self.minimum_yaw_span_degrees)
        {
            return Err(EnrollmentError::InvalidConfig);
        }
        Ok(())
    }
}

/// Model-derived evidence for one fresh enrollment observation.
#[derive(Clone, Copy)]
pub struct EnrollmentObservation<'a> {
    /// Normalized embedding produced by the admitted embedding session.
    pub embedding: &'a FaceEmbedding,
    /// Calibrated aggregate image-quality score.
    pub quality: f32,
    /// Whether calibrated passive presentation-attack detection passed.
    pub passive_liveness_passed: bool,
    /// Whether the randomized active-liveness challenge passed.
    pub active_liveness_passed: bool,
    /// Fresh paired-frame timestamp from the daemon monotonic clock.
    pub timestamp_micros: u64,
    /// Model-derived yaw in degrees used only to require multi-pose enrollment coverage.
    pub yaw_degrees: f32,
}

/// One capacity-bounded enrollment transaction.
pub struct EnrollmentSession {
    config: EnrollmentConfig,
    uid: u32,
    issued_at_micros: u64,
    deadline_micros: u64,
    last_timestamp_micros: Option<u64>,
    compatibility_sha256: Option<String>,
    dimension: Option<usize>,
    samples: Vec<Zeroizing<Vec<f32>>>,
    sample_qualities: Vec<f32>,
    sample_yaws: Vec<f32>,
    terminal_failure: bool,
}

impl EnrollmentSession {
    /// Start a new transaction for one numeric account.
    ///
    /// # Errors
    ///
    /// Returns [`EnrollmentError`] for invalid configuration or deadline overflow.
    pub fn start(
        config: EnrollmentConfig,
        uid: u32,
        issued_at_micros: u64,
    ) -> Result<Self, EnrollmentError> {
        config.validate()?;
        let deadline_micros = issued_at_micros
            .checked_add(config.duration_micros)
            .ok_or(EnrollmentError::DeadlineOverflow)?;
        Ok(Self {
            config,
            uid,
            issued_at_micros,
            deadline_micros,
            last_timestamp_micros: None,
            compatibility_sha256: None,
            dimension: None,
            samples: Vec::with_capacity(usize::from(config.maximum_samples)),
            sample_qualities: Vec::with_capacity(usize::from(config.maximum_samples)),
            sample_yaws: Vec::with_capacity(usize::from(config.maximum_samples)),
            terminal_failure: false,
        })
    }

    /// Number of accepted derived samples currently retained.
    #[must_use]
    pub const fn accepted_samples(&self) -> usize {
        self.samples.len()
    }

    /// Admit one fresh, live, high-quality observation.
    ///
    /// # Errors
    ///
    /// Returns [`EnrollmentError`] when timing, evidence, model compatibility, capacity, or
    /// pairwise identity consistency fails.
    pub fn observe(
        &mut self,
        observation: EnrollmentObservation<'_>,
    ) -> Result<(), EnrollmentError> {
        if self.terminal_failure {
            return Err(EnrollmentError::SessionFailed);
        }
        let result = self.observe_inner(observation);
        if result.as_ref().is_err_and(EnrollmentError::is_terminal_observation_failure) {
            self.terminal_failure = true;
        }
        result
    }

    fn observe_inner(
        &mut self,
        observation: EnrollmentObservation<'_>,
    ) -> Result<(), EnrollmentError> {
        self.validate_observation(&observation)?;
        if self.samples.len() >= usize::from(self.config.maximum_samples) {
            return Err(EnrollmentError::SampleLimitReached);
        }
        if let (Some(digest), Some(dimension)) = (&self.compatibility_sha256, self.dimension) {
            if digest != observation.embedding.compatibility_sha256()
                || dimension != observation.embedding.values().len()
            {
                return Err(EnrollmentError::EmbeddingIncompatible);
            }
        } else {
            self.compatibility_sha256 =
                Some(observation.embedding.compatibility_sha256().to_owned());
            self.dimension = Some(observation.embedding.values().len());
        }
        for existing in &self.samples {
            let similarity = mapped_cosine(existing, observation.embedding.values())?;
            if similarity < self.config.minimum_sample_similarity {
                return Err(EnrollmentError::InconsistentIdentity);
            }
        }
        self.samples.push(Zeroizing::new(observation.embedding.values().to_vec()));
        self.sample_qualities.push(observation.quality);
        self.sample_yaws.push(observation.yaw_degrees);
        self.last_timestamp_micros = Some(observation.timestamp_micros);
        Ok(())
    }

    /// Consume the transaction and aggregate accepted samples into one normalized template.
    ///
    /// # Errors
    ///
    /// Returns [`EnrollmentError`] when the deadline elapsed, too few samples were accepted, or
    /// aggregation cannot produce a finite non-degenerate unit vector.
    pub fn finish(self, now_micros: u64) -> Result<TemplateRecord, EnrollmentError> {
        if self.terminal_failure {
            return Err(EnrollmentError::SessionFailed);
        }
        if now_micros >= self.deadline_micros {
            return Err(EnrollmentError::DeadlineElapsed);
        }
        if self.samples.len() < usize::from(self.config.minimum_samples) {
            return Err(EnrollmentError::InsufficientSamples {
                actual: self.samples.len(),
                required: self.config.minimum_samples,
            });
        }
        let minimum_yaw = self.sample_yaws.iter().copied().reduce(f32::min);
        let maximum_yaw = self.sample_yaws.iter().copied().reduce(f32::max);
        if minimum_yaw.zip(maximum_yaw).is_none_or(|(minimum, maximum)| {
            maximum - minimum < self.config.minimum_yaw_span_degrees
        }) {
            return Err(EnrollmentError::InsufficientPoseCoverage);
        }
        let dimension = self.dimension.ok_or(EnrollmentError::InsufficientSamples {
            actual: 0,
            required: self.config.minimum_samples,
        })?;
        let mut aggregate = Zeroizing::new(vec![0.0_f32; dimension]);
        let mut total_weight = 0.0_f32;
        for (sample, quality) in self.samples.iter().zip(self.sample_qualities.iter()) {
            let weight = *quality;
            total_weight += weight;
            for (sum, value) in aggregate.iter_mut().zip(sample.iter()) {
                *sum = value.mul_add(weight, *sum);
                if !sum.is_finite() {
                    return Err(EnrollmentError::AggregateInvalid);
                }
            }
        }
        if !total_weight.is_finite() || total_weight <= f32::EPSILON {
            return Err(EnrollmentError::AggregateInvalid);
        }
        normalize(&mut aggregate)?;
        let record = TemplateRecord {
            schema_version: TEMPLATE_RECORD_SCHEMA_VERSION,
            uid: self.uid,
            model_compatibility_sha256: self
                .compatibility_sha256
                .ok_or(EnrollmentError::AggregateInvalid)?,
            embedding: aggregate.to_vec(),
        };
        record.validate()?;
        Ok(record)
    }

    fn validate_observation(
        &self,
        observation: &EnrollmentObservation<'_>,
    ) -> Result<(), EnrollmentError> {
        if observation.timestamp_micros <= self.issued_at_micros
            || observation.timestamp_micros >= self.deadline_micros
            || self
                .last_timestamp_micros
                .is_some_and(|previous| observation.timestamp_micros <= previous)
            || self.last_timestamp_micros.is_some_and(|previous| {
                observation.timestamp_micros - previous < self.config.minimum_sample_interval_micros
            })
        {
            return Err(EnrollmentError::StaleObservation);
        }
        if !valid_unit_score(observation.quality)
            || observation.quality < self.config.minimum_quality
        {
            return Err(EnrollmentError::InsufficientQuality);
        }
        if !observation.yaw_degrees.is_finite()
            || !(-90.0..=90.0).contains(&observation.yaw_degrees)
        {
            return Err(EnrollmentError::InvalidPose);
        }
        if !observation.passive_liveness_passed || !observation.active_liveness_passed {
            return Err(EnrollmentError::LivenessRequired);
        }
        if !(32..=4096).contains(&observation.embedding.values().len()) {
            return Err(EnrollmentError::EmbeddingInvalid);
        }
        Ok(())
    }
}

fn mapped_cosine(left: &[f32], right: &[f32]) -> Result<f32, EnrollmentError> {
    if left.len() != right.len() {
        return Err(EnrollmentError::EmbeddingIncompatible);
    }
    let cosine = left
        .iter()
        .zip(right.iter())
        .try_fold(0.0_f32, |sum, (left, right)| {
            let next = left.mul_add(*right, sum);
            next.is_finite().then_some(next).ok_or(EnrollmentError::EmbeddingInvalid)
        })?
        .clamp(-1.0, 1.0);
    Ok(cosine.mul_add(0.5, 0.5))
}

fn normalize(values: &mut [f32]) -> Result<(), EnrollmentError> {
    let squared_norm = values.iter().try_fold(0.0_f32, |sum, value| {
        let next = value.mul_add(*value, sum);
        next.is_finite().then_some(next).ok_or(EnrollmentError::AggregateInvalid)
    })?;
    let norm = squared_norm.sqrt();
    if !norm.is_finite() || norm <= f32::EPSILON {
        return Err(EnrollmentError::AggregateInvalid);
    }
    for value in values {
        *value /= norm;
        if !value.is_finite() {
            return Err(EnrollmentError::AggregateInvalid);
        }
    }
    Ok(())
}

fn valid_unit_score(value: f32) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

/// Enrollment policy, evidence, compatibility, or aggregation failure.
#[derive(Debug, Error)]
pub enum EnrollmentError {
    /// Configuration is non-finite, contradictory, or outside hard bounds.
    #[error("invalid enrollment configuration")]
    InvalidConfig,
    /// Monotonic transaction deadline overflowed.
    #[error("enrollment deadline overflowed")]
    DeadlineOverflow,
    /// Observation timestamp is stale, duplicated, or outside the transaction window.
    #[error("enrollment observation is stale or outside the deadline")]
    StaleObservation,
    /// Quality score is invalid or below the configured threshold.
    #[error("enrollment observation quality is insufficient")]
    InsufficientQuality,
    /// Model-derived pose is non-finite or outside the admitted physical range.
    #[error("enrollment observation pose is invalid")]
    InvalidPose,
    /// Both passive and randomized active liveness must pass.
    #[error("enrollment requires passive and active liveness")]
    LivenessRequired,
    /// Embedding dimension or arithmetic is invalid.
    #[error("enrollment embedding is invalid")]
    EmbeddingInvalid,
    /// Observation uses a different model contract or dimension.
    #[error("enrollment embedding is incompatible with earlier observations")]
    EmbeddingIncompatible,
    /// Observation differs too much from an already accepted identity sample.
    #[error("enrollment observations are not identity-consistent")]
    InconsistentIdentity,
    /// No more derived samples may be retained.
    #[error("enrollment sample limit reached")]
    SampleLimitReached,
    /// Transaction deadline elapsed before completion.
    #[error("enrollment deadline elapsed")]
    DeadlineElapsed,
    /// Fewer than the configured samples were accepted.
    #[error("enrollment has {actual} samples but requires {required}")]
    InsufficientSamples {
        /// Accepted samples.
        actual: usize,
        /// Required samples.
        required: u8,
    },
    /// Accepted samples do not cover the configured yaw range.
    #[error("enrollment samples do not cover the required pose range")]
    InsufficientPoseCoverage,
    /// Sample aggregation produced a non-finite or degenerate vector.
    #[error("enrollment aggregate is invalid")]
    AggregateInvalid,
    /// A prior security-relevant observation failure permanently closed the transaction.
    #[error("enrollment transaction is terminally failed")]
    SessionFailed,
    /// Inference embedding validation failed.
    #[error("enrollment inference validation failed: {0}")]
    Inference(#[from] InferenceError),
    /// Final template validation failed.
    #[error("enrollment template validation failed: {0}")]
    Storage(#[from] StorageError),
}

impl EnrollmentError {
    const fn is_terminal_observation_failure(&self) -> bool {
        !matches!(self, Self::InsufficientQuality)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn embedding(
        digest_byte: u8,
        first: f32,
        second: f32,
    ) -> Result<FaceEmbedding, InferenceError> {
        let digest = format!("{digest_byte:02x}").repeat(32);
        let norm = first.hypot(second);
        let mut values = vec![0.0; 32];
        values[0] = first / norm;
        values[1] = second / norm;
        FaceEmbedding::from_normalized_template(&digest, &values)
    }

    fn observation(embedding: &FaceEmbedding, timestamp_micros: u64) -> EnrollmentObservation<'_> {
        EnrollmentObservation {
            embedding,
            quality: 0.9,
            passive_liveness_passed: true,
            active_liveness_passed: true,
            timestamp_micros,
            yaw_degrees: match timestamp_micros {
                2_000_000 => -10.0,
                4_000_000 => 10.0,
                _ => 0.0,
            },
        }
    }

    #[test]
    fn three_fresh_consistent_samples_produce_only_a_normalized_template()
    -> Result<(), EnrollmentError> {
        let first = embedding(0xaa, 1.0, 0.0)?;
        let second = embedding(0xaa, 0.99, 0.1)?;
        let third = embedding(0xaa, 0.98, -0.1)?;
        let mut session = EnrollmentSession::start(config(), 1000, 1_000_000)?;
        session.observe(observation(&first, 2_000_000))?;
        session.observe(observation(&second, 3_000_000))?;
        session.observe(observation(&third, 4_000_000))?;
        let record = session.finish(5_000_000)?;

        assert_eq!(record.uid, 1000);
        assert_eq!(record.embedding.len(), 32);
        assert!(
            (record.embedding.iter().map(|value| value * value).sum::<f32>() - 1.0).abs() < 1.0e-3
        );
        record.validate()?;
        Ok(())
    }

    #[test]
    fn stale_low_quality_or_non_live_observations_fail_closed() -> Result<(), EnrollmentError> {
        let face = embedding(0xaa, 1.0, 0.0)?;
        let mut stale = EnrollmentSession::start(config(), 1000, 1_000_000)?;
        assert!(matches!(
            stale.observe(observation(&face, 1_000_000)),
            Err(EnrollmentError::StaleObservation)
        ));
        assert!(matches!(
            stale.observe(observation(&face, 2_000_000)),
            Err(EnrollmentError::SessionFailed)
        ));

        let mut session = EnrollmentSession::start(config(), 1000, 1_000_000)?;
        let mut low_quality = observation(&face, 2_000_000);
        low_quality.quality = 0.2;
        assert!(matches!(session.observe(low_quality), Err(EnrollmentError::InsufficientQuality)));
        let mut spoof = observation(&face, 2_000_000);
        spoof.passive_liveness_passed = false;
        assert!(matches!(session.observe(spoof), Err(EnrollmentError::LivenessRequired)));
        assert!(matches!(
            session.observe(observation(&face, 3_000_000)),
            Err(EnrollmentError::SessionFailed)
        ));
        Ok(())
    }

    #[test]
    fn mixed_contracts_and_inconsistent_identities_are_rejected() -> Result<(), EnrollmentError> {
        let first = embedding(0xaa, 1.0, 0.0)?;
        let incompatible = embedding(0xbb, 1.0, 0.0)?;
        let different = embedding(0xaa, 0.0, 1.0)?;
        let mut session = EnrollmentSession::start(config(), 1000, 1_000_000)?;
        session.observe(observation(&first, 2_000_000))?;
        assert!(matches!(
            session.observe(observation(&incompatible, 3_000_000)),
            Err(EnrollmentError::EmbeddingIncompatible)
        ));

        let mut session = EnrollmentSession::start(config(), 1000, 1_000_000)?;
        session.observe(observation(&first, 2_000_000))?;
        assert!(matches!(
            session.observe(observation(&different, 3_000_000)),
            Err(EnrollmentError::InconsistentIdentity)
        ));
        Ok(())
    }

    #[test]
    fn finish_requires_sample_count_and_unexpired_deadline() -> Result<(), EnrollmentError> {
        let face = embedding(0xaa, 1.0, 0.0)?;
        let mut insufficient = EnrollmentSession::start(config(), 1000, 1_000_000)?;
        insufficient.observe(observation(&face, 2_000_000))?;
        assert!(matches!(
            insufficient.finish(3_000_000),
            Err(EnrollmentError::InsufficientSamples { .. })
        ));

        let expired = EnrollmentSession::start(config(), 1000, 1_000_000)?;
        assert!(matches!(expired.finish(31_000_000), Err(EnrollmentError::DeadlineElapsed)));
        Ok(())
    }

    #[test]
    fn near_duplicate_frames_and_insufficient_pose_coverage_fail_closed()
    -> Result<(), EnrollmentError> {
        let face = embedding(0xaa, 1.0, 0.0)?;
        let mut duplicate = EnrollmentSession::start(config(), 1000, 1_000_000)?;
        duplicate.observe(observation(&face, 2_000_000))?;
        assert!(matches!(
            duplicate.observe(observation(&face, 2_100_000)),
            Err(EnrollmentError::StaleObservation)
        ));

        let mut flat = EnrollmentSession::start(config(), 1000, 1_000_000)?;
        for timestamp in [2_000_000, 3_000_000, 4_000_000] {
            let mut sample = observation(&face, timestamp);
            sample.yaw_degrees = 0.0;
            flat.observe(sample)?;
        }
        assert!(matches!(flat.finish(5_000_000), Err(EnrollmentError::InsufficientPoseCoverage)));
        Ok(())
    }

    #[test]
    fn quality_weighted_centroid_limits_lower_quality_sample_drift() -> Result<(), EnrollmentError>
    {
        let frontal = embedding(0xaa, 1.0, 0.0)?;
        let offset = embedding(0xaa, 0.98, 0.2)?;
        let mut session = EnrollmentSession::start(config(), 1000, 1_000_000)?;
        let mut first = observation(&frontal, 2_000_000);
        first.quality = 1.0;
        session.observe(first)?;
        let mut second = observation(&offset, 3_000_000);
        second.quality = 0.7;
        session.observe(second)?;
        let mut third = observation(&offset, 4_000_000);
        third.quality = 0.7;
        session.observe(third)?;
        let record = session.finish(5_000_000)?;
        assert!(record.embedding[0] > 0.99);
        assert!(record.embedding[1] < 0.13);
        Ok(())
    }
}
