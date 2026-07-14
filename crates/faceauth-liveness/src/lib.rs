//! Bounded randomized active-liveness challenge state machine.
//!
//! Landmark and presentation-attack models provide measurements; this crate owns challenge
//! randomness, freshness, ordering, deadlines, and the final active-check result.

use faceauth_core::{CapturePair, CapturePolicy};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Randomized action requested from the user.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ChallengeAction {
    /// Close both eyes and open them again.
    Blink,
    /// Turn the face toward the user's left and return to center.
    TurnLeft,
    /// Turn the face toward the user's right and return to center.
    TurnRight,
}

impl ChallengeAction {
    fn random() -> Result<Self, ChallengeError> {
        loop {
            let mut byte = [0_u8; 1];
            getrandom::fill(&mut byte)?;
            if byte[0] < 255 {
                return Ok(match byte[0] % 3 {
                    0 => Self::Blink,
                    1 => Self::TurnLeft,
                    _ => Self::TurnRight,
                });
            }
        }
    }
}

/// Calibrated bounds for one active challenge.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChallengeConfig {
    /// Maximum IR/RGB timestamp skew accepted for each measurement.
    pub max_pair_skew_micros: u64,
    /// Total challenge duration measured using monotonic timestamps.
    pub duration_micros: u64,
    /// Maximum number of fresh observations processed before failure.
    pub max_observations: u16,
    /// Minimum normalized eye openness considered neutral/open.
    pub open_eye_threshold: f32,
    /// Maximum normalized eye openness considered closed during a blink.
    pub closed_eye_threshold: f32,
    /// Maximum absolute yaw in degrees considered centered.
    pub neutral_yaw_degrees: f32,
    /// Minimum absolute yaw in degrees required for a turn.
    pub turn_yaw_degrees: f32,
}

impl ChallengeConfig {
    /// Validate structural bounds. Model-specific values still require hardware calibration.
    ///
    /// # Errors
    ///
    /// Returns [`ChallengeError::InvalidConfig`] for non-finite, contradictory, or unbounded
    /// settings.
    pub fn validate(self) -> Result<(), ChallengeError> {
        let scores_valid = self.open_eye_threshold.is_finite()
            && self.closed_eye_threshold.is_finite()
            && (0.0..=1.0).contains(&self.open_eye_threshold)
            && (0.0..=1.0).contains(&self.closed_eye_threshold)
            && self.closed_eye_threshold < self.open_eye_threshold;
        let yaw_valid = self.neutral_yaw_degrees.is_finite()
            && self.turn_yaw_degrees.is_finite()
            && (0.0..=45.0).contains(&self.neutral_yaw_degrees)
            && (5.0..=75.0).contains(&self.turn_yaw_degrees)
            && self.neutral_yaw_degrees < self.turn_yaw_degrees;
        if self.max_pair_skew_micros == 0
            || !(1_000_000..=30_000_000).contains(&self.duration_micros)
            || !(3..=300).contains(&self.max_observations)
            || !scores_valid
            || !yaw_valid
        {
            return Err(ChallengeError::InvalidConfig);
        }
        Ok(())
    }
}

/// Model-derived measurements for one fresh paired observation.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChallengeObservation {
    /// Paired monotonic capture timestamps.
    pub timing: CapturePair,
    /// IR driver sequence number used to reject duplicated frames.
    pub infrared_sequence: u32,
    /// Visible-light driver sequence number used to reject duplicated frames.
    pub visible_sequence: u32,
    /// Number of faces detected in the paired observation.
    pub face_count: u8,
    /// Normalized aggregate openness of both eyes in the range 0..=1.
    pub eye_openness: f32,
    /// Head yaw in degrees: negative is user's left and positive is user's right.
    pub yaw_degrees: f32,
}

impl ChallengeObservation {
    fn timestamp(self) -> Result<u64, ChallengeError> {
        self.timing
            .visible_timestamp_micros
            .ok_or(ChallengeError::VisibleFrameRequired)
            .map(|visible| self.timing.infrared_timestamp_micros.max(visible))
    }

    fn validate(self, max_pair_skew_micros: u64) -> Result<(), ChallengeError> {
        CapturePolicy { require_infrared: true, require_visible: true, max_pair_skew_micros }
            .accepts(self.timing)
            .map_err(|_| ChallengeError::InvalidFramePair)?;
        if self.face_count != 1 {
            return Err(ChallengeError::SingleFaceRequired { actual: self.face_count });
        }
        if !self.eye_openness.is_finite()
            || !(0.0..=1.0).contains(&self.eye_openness)
            || !self.yaw_degrees.is_finite()
            || !(-90.0..=90.0).contains(&self.yaw_degrees)
        {
            return Err(ChallengeError::InvalidMeasurement);
        }
        Ok(())
    }
}

/// Current externally visible progress of an active challenge.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ChallengeProgress {
    /// Waiting for a centered, eyes-open baseline before prompting an action.
    BaselineRequired,
    /// Baseline passed; prompt the randomized action.
    ActionRequired(ChallengeAction),
    /// The action was observed; the user must return to centered, eyes-open neutral pose.
    RecoveryRequired,
    /// The complete temporal challenge passed.
    Passed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Phase {
    Baseline,
    Action,
    Recovery,
    Passed,
}

/// One bounded randomized active-liveness transaction.
pub struct ChallengeSession {
    config: ChallengeConfig,
    action: ChallengeAction,
    issued_at_micros: u64,
    deadline_micros: u64,
    observations: u16,
    last_timestamp_micros: Option<u64>,
    last_sequences: Option<(u32, u32)>,
    phase: Phase,
}

impl ChallengeSession {
    /// Begin a challenge using operating-system randomness.
    ///
    /// `issued_at_micros` must come from the same monotonic clock as capture timestamps.
    ///
    /// # Errors
    ///
    /// Returns [`ChallengeError`] for invalid configuration, deadline overflow, or random-source
    /// failure.
    pub fn begin(config: ChallengeConfig, issued_at_micros: u64) -> Result<Self, ChallengeError> {
        Self::with_action(config, issued_at_micros, ChallengeAction::random()?)
    }

    fn with_action(
        config: ChallengeConfig,
        issued_at_micros: u64,
        action: ChallengeAction,
    ) -> Result<Self, ChallengeError> {
        config.validate()?;
        let deadline_micros = issued_at_micros
            .checked_add(config.duration_micros)
            .ok_or(ChallengeError::DeadlineOverflow)?;
        Ok(Self {
            config,
            action,
            issued_at_micros,
            deadline_micros,
            observations: 0,
            last_timestamp_micros: None,
            last_sequences: None,
            phase: Phase::Baseline,
        })
    }

    /// Return current progress without exposing any biometric measurement.
    #[must_use]
    pub const fn progress(&self) -> ChallengeProgress {
        match self.phase {
            Phase::Baseline => ChallengeProgress::BaselineRequired,
            Phase::Action => ChallengeProgress::ActionRequired(self.action),
            Phase::Recovery => ChallengeProgress::RecoveryRequired,
            Phase::Passed => ChallengeProgress::Passed,
        }
    }

    /// Process one fresh paired observation.
    ///
    /// # Errors
    ///
    /// Fails closed on malformed measurements, invalid pairing, repeated or non-increasing frames,
    /// deadline expiry, or observation-budget exhaustion.
    pub fn observe(
        &mut self,
        observation: ChallengeObservation,
    ) -> Result<ChallengeProgress, ChallengeError> {
        if self.phase == Phase::Passed {
            return Err(ChallengeError::AlreadyCompleted);
        }
        observation.validate(self.config.max_pair_skew_micros)?;
        let timestamp = observation.timestamp()?;
        if timestamp < self.issued_at_micros {
            return Err(ChallengeError::PredatesChallenge);
        }
        if timestamp > self.deadline_micros {
            return Err(ChallengeError::Expired);
        }
        if self.last_timestamp_micros.is_some_and(|last| timestamp <= last) {
            return Err(ChallengeError::NonIncreasingTimestamp);
        }
        let sequences = (observation.infrared_sequence, observation.visible_sequence);
        if self.last_sequences.is_some_and(|last| last.0 == sequences.0 || last.1 == sequences.1) {
            return Err(ChallengeError::RepeatedCameraFrame);
        }
        self.observations =
            self.observations.checked_add(1).ok_or(ChallengeError::ObservationBudgetExhausted)?;
        if self.observations > self.config.max_observations {
            return Err(ChallengeError::ObservationBudgetExhausted);
        }
        self.last_timestamp_micros = Some(timestamp);
        self.last_sequences = Some(sequences);

        let neutral = observation.eye_openness >= self.config.open_eye_threshold
            && observation.yaw_degrees.abs() <= self.config.neutral_yaw_degrees;
        match self.phase {
            Phase::Baseline if neutral => self.phase = Phase::Action,
            Phase::Action if self.action_observed(observation) => self.phase = Phase::Recovery,
            Phase::Recovery if neutral => self.phase = Phase::Passed,
            Phase::Baseline | Phase::Action | Phase::Recovery | Phase::Passed => {}
        }
        Ok(self.progress())
    }

    fn action_observed(&self, observation: ChallengeObservation) -> bool {
        match self.action {
            ChallengeAction::Blink => observation.eye_openness <= self.config.closed_eye_threshold,
            ChallengeAction::TurnLeft => observation.yaw_degrees <= -self.config.turn_yaw_degrees,
            ChallengeAction::TurnRight => observation.yaw_degrees >= self.config.turn_yaw_degrees,
        }
    }
}

/// Active challenge failure.
#[derive(Debug, Error)]
pub enum ChallengeError {
    /// Configuration is malformed, contradictory, or outside hard bounds.
    #[error("invalid active-challenge configuration")]
    InvalidConfig,
    /// The monotonic deadline overflowed.
    #[error("active-challenge deadline overflowed")]
    DeadlineOverflow,
    /// Operating-system randomness was unavailable.
    #[error("unable to generate a randomized active challenge: {0}")]
    Random(#[from] getrandom::Error),
    /// A paired visible-light timestamp is mandatory.
    #[error("active challenge requires a visible-light frame")]
    VisibleFrameRequired,
    /// IR/RGB timestamps did not satisfy the configured pairing policy.
    #[error("active-challenge observation has an invalid IR/RGB frame pair")]
    InvalidFramePair,
    /// Exactly one face must be measured throughout the challenge.
    #[error("active challenge requires exactly one face; observed {actual}")]
    SingleFaceRequired {
        /// Number of faces reported by the detector.
        actual: u8,
    },
    /// A normalized or angular measurement was malformed.
    #[error("active-challenge measurement is invalid")]
    InvalidMeasurement,
    /// Observation time did not strictly increase.
    #[error("active-challenge timestamps must strictly increase")]
    NonIncreasingTimestamp,
    /// Observation was captured before this challenge was issued.
    #[error("active-challenge observation predates challenge issuance")]
    PredatesChallenge,
    /// At least one required camera repeated its prior frame.
    #[error("active challenge rejected a repeated camera frame")]
    RepeatedCameraFrame,
    /// The monotonic challenge deadline elapsed.
    #[error("active challenge expired")]
    Expired,
    /// Too many observations were processed without completing the challenge.
    #[error("active-challenge observation budget exhausted")]
    ObservationBudgetExhausted,
    /// Additional input was supplied after terminal success.
    #[error("active challenge is already complete")]
    AlreadyCompleted,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ChallengeConfig {
        ChallengeConfig {
            max_pair_skew_micros: 100_000,
            duration_micros: 10_000_000,
            max_observations: 90,
            open_eye_threshold: 0.65,
            closed_eye_threshold: 0.25,
            neutral_yaw_degrees: 10.0,
            turn_yaw_degrees: 20.0,
        }
    }

    fn observation(timestamp: u64, sequence: u32, eyes: f32, yaw: f32) -> ChallengeObservation {
        ChallengeObservation {
            timing: CapturePair {
                infrared_timestamp_micros: timestamp,
                visible_timestamp_micros: Some(timestamp + 10_000),
            },
            infrared_sequence: sequence,
            visible_sequence: sequence + 100,
            face_count: 1,
            eye_openness: eyes,
            yaw_degrees: yaw,
        }
    }

    #[test]
    fn blink_requires_baseline_action_and_recovery() -> Result<(), ChallengeError> {
        let mut session =
            ChallengeSession::with_action(config(), 1_000_000, ChallengeAction::Blink)?;

        assert_eq!(
            session.observe(observation(1_100_000, 1, 0.8, 0.0))?,
            ChallengeProgress::ActionRequired(ChallengeAction::Blink)
        );
        assert_eq!(
            session.observe(observation(1_200_000, 2, 0.1, 0.0))?,
            ChallengeProgress::RecoveryRequired
        );
        assert_eq!(
            session.observe(observation(1_300_000, 3, 0.8, 0.0))?,
            ChallengeProgress::Passed
        );
        Ok(())
    }

    #[test]
    fn wrong_turn_does_not_advance() -> Result<(), ChallengeError> {
        let mut session =
            ChallengeSession::with_action(config(), 1_000_000, ChallengeAction::TurnLeft)?;
        let _ = session.observe(observation(1_100_000, 1, 0.8, 0.0))?;

        assert_eq!(
            session.observe(observation(1_200_000, 2, 0.8, 30.0))?,
            ChallengeProgress::ActionRequired(ChallengeAction::TurnLeft)
        );
        assert_eq!(
            session.observe(observation(1_300_000, 3, 0.8, -30.0))?,
            ChallengeProgress::RecoveryRequired
        );
        Ok(())
    }

    #[test]
    fn repeated_or_non_increasing_frames_fail_closed() -> Result<(), ChallengeError> {
        let mut session =
            ChallengeSession::with_action(config(), 1_000_000, ChallengeAction::Blink)?;
        let first = observation(1_100_000, 1, 0.8, 0.0);
        let _ = session.observe(first)?;

        assert!(matches!(session.observe(first), Err(ChallengeError::NonIncreasingTimestamp)));

        let mut session =
            ChallengeSession::with_action(config(), 1_000_000, ChallengeAction::Blink)?;
        let _ = session.observe(first)?;
        let mut repeated_ir = observation(1_200_000, 2, 0.1, 0.0);
        repeated_ir.infrared_sequence = first.infrared_sequence;
        assert!(matches!(session.observe(repeated_ir), Err(ChallengeError::RepeatedCameraFrame)));
        Ok(())
    }

    #[test]
    fn invalid_pair_and_multiple_faces_fail_closed() -> Result<(), ChallengeError> {
        let mut session =
            ChallengeSession::with_action(config(), 1_000_000, ChallengeAction::Blink)?;
        let mut invalid_pair = observation(1_100_000, 1, 0.8, 0.0);
        invalid_pair.timing.visible_timestamp_micros = Some(1_300_001);
        assert!(matches!(session.observe(invalid_pair), Err(ChallengeError::InvalidFramePair)));

        let mut multiple_faces = observation(1_100_000, 1, 0.8, 0.0);
        multiple_faces.face_count = 2;
        assert!(matches!(
            session.observe(multiple_faces),
            Err(ChallengeError::SingleFaceRequired { actual: 2 })
        ));
        Ok(())
    }

    #[test]
    fn deadline_and_observation_budget_are_bounded() -> Result<(), ChallengeError> {
        let config = ChallengeConfig { max_observations: 3, ..config() };
        let mut expired = ChallengeSession::with_action(config, 1_000_000, ChallengeAction::Blink)?;
        assert!(matches!(
            expired.observe(observation(11_000_001, 1, 0.8, 0.0)),
            Err(ChallengeError::Expired)
        ));

        let mut budgeted =
            ChallengeSession::with_action(config, 1_000_000, ChallengeAction::Blink)?;
        let _ = budgeted.observe(observation(1_100_000, 1, 0.4, 0.0))?;
        let _ = budgeted.observe(observation(1_200_000, 2, 0.4, 0.0))?;
        let _ = budgeted.observe(observation(1_300_000, 3, 0.4, 0.0))?;
        assert!(matches!(
            budgeted.observe(observation(1_400_000, 4, 0.4, 0.0)),
            Err(ChallengeError::ObservationBudgetExhausted)
        ));
        Ok(())
    }

    #[test]
    fn observations_cannot_predate_challenge_issuance() -> Result<(), ChallengeError> {
        let mut session =
            ChallengeSession::with_action(config(), 1_000_000, ChallengeAction::Blink)?;
        assert!(matches!(
            session.observe(observation(900_000, 1, 0.8, 0.0)),
            Err(ChallengeError::PredatesChallenge)
        ));
        Ok(())
    }

    #[test]
    fn configuration_requires_explicit_calibrated_bounds() {
        assert!(ChallengeConfig { open_eye_threshold: f32::NAN, ..config() }.validate().is_err());
        assert!(
            ChallengeConfig { closed_eye_threshold: 0.8, open_eye_threshold: 0.7, ..config() }
                .validate()
                .is_err()
        );
        assert!(ChallengeConfig { max_observations: 301, ..config() }.validate().is_err());
    }
}
