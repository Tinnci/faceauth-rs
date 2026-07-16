use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Camera modality used as authentication evidence.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CaptureModality {
    /// Near-infrared grayscale capture.
    Infrared,
    /// Visible-light capture.
    Visible,
}

/// Timestamp metadata for a paired camera observation.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CapturePair {
    /// Monotonic timestamp of the IR frame, in microseconds.
    pub infrared_timestamp_micros: u64,
    /// Monotonic timestamp of the visible frame, when one is required.
    pub visible_timestamp_micros: Option<u64>,
}

/// Rules for accepting one IR/RGB capture pair.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CapturePolicy {
    /// Require an IR frame for authentication.
    pub require_infrared: bool,
    /// Require a visible-light frame in addition to IR.
    pub require_visible: bool,
    /// Maximum permitted timestamp difference between paired frames.
    pub max_pair_skew_micros: u64,
}

impl Default for CapturePolicy {
    fn default() -> Self {
        Self { require_infrared: true, require_visible: true, max_pair_skew_micros: 100_000 }
    }
}

impl CapturePolicy {
    /// Validate policy invariants.
    ///
    /// # Errors
    ///
    /// Returns [`PolicyError`] when IR is optional or the frame-pair tolerance is zero.
    pub const fn validate(self) -> Result<(), PolicyError> {
        if !self.require_infrared {
            return Err(PolicyError::InfraredRequired);
        }
        if self.max_pair_skew_micros == 0 {
            return Err(PolicyError::PairSkewMustBePositive);
        }
        Ok(())
    }

    /// Check whether timestamp metadata satisfies this capture policy.
    ///
    /// # Errors
    ///
    /// Returns [`PolicyError`] when the policy is invalid, a required visible frame is absent,
    /// or the IR and visible frames are too far apart in monotonic time.
    pub fn accepts(self, pair: CapturePair) -> Result<(), PolicyError> {
        self.validate()?;
        let Some(visible_timestamp) = pair.visible_timestamp_micros else {
            return if self.require_visible {
                Err(PolicyError::VisibleFrameRequired)
            } else {
                Ok(())
            };
        };
        let skew = pair.infrared_timestamp_micros.abs_diff(visible_timestamp);
        if skew > self.max_pair_skew_micros {
            return Err(PolicyError::FramePairTooFarApart {
                actual_micros: skew,
                maximum_micros: self.max_pair_skew_micros,
            });
        }
        Ok(())
    }
}

/// Required presentation-attack detection strength.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum LivenessLevel {
    /// Passive model checks only. Intended for diagnostics, not production authentication.
    Passive,
    /// Passive checks plus a randomized user action.
    PassiveAndActive,
}

/// Authentication policy evaluated after capture and inference.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AuthPolicy {
    /// Capture requirements.
    pub capture: CapturePolicy,
    /// Required liveness mode.
    pub liveness: LivenessLevel,
    /// Minimum model similarity score in the inclusive range 0..=1.
    pub minimum_similarity: f32,
    /// Minimum aggregate image quality score in the inclusive range 0..=1.
    pub minimum_quality: f32,
    /// Preserve password fallback when face authentication fails.
    pub password_fallback: bool,
}

impl Default for AuthPolicy {
    fn default() -> Self {
        Self {
            capture: CapturePolicy::default(),
            liveness: LivenessLevel::PassiveAndActive,
            minimum_similarity: 0.55,
            minimum_quality: 0.65,
            password_fallback: true,
        }
    }
}

impl AuthPolicy {
    /// Validate policy invariants.
    ///
    /// # Errors
    ///
    /// Returns [`PolicyError`] when the capture policy or score thresholds are invalid, or when
    /// password fallback has been disabled.
    pub fn validate(self) -> Result<(), PolicyError> {
        self.capture.validate()?;
        validate_unit_score("minimum_similarity", self.minimum_similarity)?;
        validate_unit_score("minimum_quality", self.minimum_quality)?;
        if !self.password_fallback {
            return Err(PolicyError::PasswordFallbackRequired);
        }
        Ok(())
    }

    /// Evaluate already-computed evidence without performing camera or model work.
    ///
    /// # Errors
    ///
    /// Returns [`PolicyError`] when the policy or normalized evidence scores are invalid.
    pub fn evaluate(
        self,
        evidence: AuthenticationEvidence,
    ) -> Result<AuthenticationDecision, PolicyError> {
        self.validate()?;
        validate_unit_score("similarity", evidence.similarity)?;
        validate_unit_score("quality", evidence.quality)?;

        let reason = if evidence.infrared != ObservationStatus::Observed {
            DecisionReason::InfraredMissing
        } else if self.capture.require_visible && evidence.visible != ObservationStatus::Observed {
            DecisionReason::VisibleMissing
        } else if evidence.quality < self.minimum_quality {
            DecisionReason::InsufficientQuality
        } else if evidence.passive_liveness != CheckStatus::Passed {
            DecisionReason::PassiveLivenessFailed
        } else if self.liveness == LivenessLevel::PassiveAndActive
            && evidence.active_challenge != CheckStatus::Passed
        {
            DecisionReason::ActiveChallengeFailed
        } else if evidence.similarity < self.minimum_similarity {
            DecisionReason::FaceMismatch
        } else {
            DecisionReason::Accepted
        };

        Ok(AuthenticationDecision { accepted: reason == DecisionReason::Accepted, reason })
    }
}

/// Whether a required camera observation is part of the evidence set.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ObservationStatus {
    /// No valid observation was supplied.
    Missing,
    /// A valid observation was supplied by the capture pipeline.
    Observed,
}

/// Result of a liveness or challenge check.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CheckStatus {
    /// The check was not performed.
    NotPerformed,
    /// The check ran and failed.
    Failed,
    /// The check ran and passed.
    Passed,
}

/// Evidence produced by the capture, quality, liveness, and recognition pipeline.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct AuthenticationEvidence {
    /// Face embedding similarity normalized to 0..=1 by the model adapter.
    pub similarity: f32,
    /// Aggregate capture quality normalized to 0..=1.
    pub quality: f32,
    /// Status of the IR observation.
    pub infrared: ObservationStatus,
    /// Status of the paired visible-light observation.
    pub visible: ObservationStatus,
    /// Result of passive presentation-attack checks.
    pub passive_liveness: CheckStatus,
    /// Result of the randomized active challenge.
    pub active_challenge: CheckStatus,
}

/// Result of policy evaluation.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AuthenticationDecision {
    /// Whether authentication can succeed.
    pub accepted: bool,
    /// Stable machine-readable reason.
    pub reason: DecisionReason,
}

/// Stable policy decision reason.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum DecisionReason {
    /// All required evidence passed.
    Accepted,
    /// IR evidence was missing.
    InfraredMissing,
    /// Visible-light evidence was missing.
    VisibleMissing,
    /// Capture quality was below policy.
    InsufficientQuality,
    /// Passive presentation-attack detection failed.
    PassiveLivenessFailed,
    /// The randomized active challenge failed.
    ActiveChallengeFailed,
    /// Face similarity was below policy.
    FaceMismatch,
}

/// Invalid policy or evidence.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum PolicyError {
    /// Production policy must require IR evidence.
    #[error("infrared capture must remain required")]
    InfraredRequired,
    /// Timestamp tolerance cannot be zero.
    #[error("frame-pair skew tolerance must be positive")]
    PairSkewMustBePositive,
    /// A required visible frame was not supplied.
    #[error("a visible-light frame is required")]
    VisibleFrameRequired,
    /// The IR/RGB timestamps cannot be treated as one observation.
    #[error("frame-pair skew {actual_micros}us exceeds {maximum_micros}us")]
    FramePairTooFarApart {
        /// Measured timestamp difference.
        actual_micros: u64,
        /// Configured timestamp difference limit.
        maximum_micros: u64,
    },
    /// A normalized score was not finite or outside 0..=1.
    #[error("{field} must be a finite score in 0..=1")]
    InvalidUnitScore {
        /// Name of the invalid score.
        field: &'static str,
    },
    /// Password fallback is mandatory until the biometric path is independently certified.
    #[error("password fallback must remain enabled")]
    PasswordFallbackRequired,
}

fn validate_unit_score(field: &'static str, value: f32) -> Result<(), PolicyError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(PolicyError::InvalidUnitScore { field })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn passing_evidence() -> AuthenticationEvidence {
        AuthenticationEvidence {
            similarity: 0.8,
            quality: 0.8,
            infrared: ObservationStatus::Observed,
            visible: ObservationStatus::Observed,
            passive_liveness: CheckStatus::Passed,
            active_challenge: CheckStatus::Passed,
        }
    }

    #[test]
    fn default_policy_requires_dual_camera_and_password_fallback() {
        let policy = AuthPolicy::default();
        assert!(policy.capture.require_infrared);
        assert!(policy.capture.require_visible);
        assert!(policy.password_fallback);
        assert_eq!(policy.validate(), Ok(()));
    }

    #[test]
    fn paired_frames_must_be_close_in_monotonic_time() {
        let policy = CapturePolicy::default();
        let pair = CapturePair {
            infrared_timestamp_micros: 1_000_000,
            visible_timestamp_micros: Some(1_100_001),
        };
        assert_eq!(
            policy.accepts(pair),
            Err(PolicyError::FramePairTooFarApart {
                actual_micros: 100_001,
                maximum_micros: 100_000,
            })
        );
    }

    #[test]
    fn active_liveness_cannot_be_skipped_by_a_high_match_score() {
        let policy = AuthPolicy::default();
        let mut evidence = passing_evidence();
        evidence.active_challenge = CheckStatus::Failed;
        let decision = policy.evaluate(evidence);
        assert_eq!(
            decision,
            Ok(AuthenticationDecision {
                accepted: false,
                reason: DecisionReason::ActiveChallengeFailed,
            })
        );
    }

    #[test]
    fn hpd_presence_is_not_part_of_authentication_evidence() {
        let policy = AuthPolicy::default();
        assert_eq!(
            policy.evaluate(passing_evidence()),
            Ok(AuthenticationDecision { accepted: true, reason: DecisionReason::Accepted })
        );
    }
}
