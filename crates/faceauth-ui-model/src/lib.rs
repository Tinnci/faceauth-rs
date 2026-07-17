//! Desktop-independent, privacy-preserving experience state for KCM, OSD, and lock-screen clients.

use faceauth_management::{ManagementProgress, ManagementResult, ManagementUpdate, OperationId};
use faceauth_protocol::{DecisionCode, ProgressCode, TransactionId};
use serde::Serialize;
use thiserror::Error;

/// Exact operation identity owned by one visible experience.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum ExperienceFlow {
    /// PAM, lock-screen, login, or Polkit authentication transaction.
    Authentication {
        /// Exact daemon transaction.
        transaction_id: TransactionId,
    },
    /// Manager1 enrollment or template replacement operation.
    Enrollment {
        /// Exact management operation.
        operation_id: OperationId,
    },
}

/// Closed, localization-independent user guidance.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ExperienceCue {
    /// Cameras and model sessions are being prepared.
    Preparing,
    /// Move into the calibrated capture region.
    PositionFace,
    /// Hold a centered, eyes-open pose.
    HoldStill,
    /// Compatibility fallback when an older enrollment source omits the exact action.
    FollowChallenge,
    /// Close and reopen both eyes.
    Blink,
    /// Turn toward the user's left.
    TurnLeft,
    /// Turn toward the user's right.
    TurnRight,
    /// Return to a centered, eyes-open pose.
    ReturnToCenter,
    /// Derived evidence is being checked; the user can remain still.
    Processing,
}

impl ExperienceCue {
    /// Stable token shared with QML and accessibility tests.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Preparing => "preparing",
            Self::PositionFace => "position-face",
            Self::HoldStill => "hold-still",
            Self::FollowChallenge => "active-challenge",
            Self::Blink => "blink",
            Self::TurnLeft => "turn-left",
            Self::TurnRight => "turn-right",
            Self::ReturnToCenter => "return-to-center",
            Self::Processing => "processing",
        }
    }

    /// Stable icon name from the platform theme.
    #[must_use]
    pub const fn icon_name(self) -> &'static str {
        match self {
            Self::Preparing | Self::Processing => "view-refresh",
            Self::PositionFace => "edit-image-face-recognize",
            Self::HoldStill => "media-playback-pause",
            Self::FollowChallenge => "system-run",
            Self::Blink => "face-smile",
            Self::TurnLeft => "go-previous",
            Self::TurnRight => "go-next",
            Self::ReturnToCenter => "go-home",
        }
    }

    /// Stable localization key for the primary instruction.
    #[must_use]
    pub const fn title_key(self) -> &'static str {
        match self {
            Self::Preparing => "faceauth.cue.preparing.title",
            Self::PositionFace => "faceauth.cue.position-face.title",
            Self::HoldStill => "faceauth.cue.hold-still.title",
            Self::FollowChallenge => "faceauth.cue.active-challenge.title",
            Self::Blink => "faceauth.cue.blink.title",
            Self::TurnLeft => "faceauth.cue.turn-left.title",
            Self::TurnRight => "faceauth.cue.turn-right.title",
            Self::ReturnToCenter => "faceauth.cue.return-to-center.title",
            Self::Processing => "faceauth.cue.processing.title",
        }
    }
}

/// Privacy-preserving terminal experience category.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ExperienceOutcome {
    /// Authentication succeeded or an encrypted template was committed.
    Succeeded,
    /// Biometric evidence did not pass; no sensitive reason is exposed to the UI.
    TryAgain,
    /// The owning caller cancelled.
    Cancelled,
    /// The fixed operation deadline elapsed.
    TimedOut,
    /// A camera, daemon, storage, or other internal boundary was unavailable.
    Unavailable,
}

impl ExperienceOutcome {
    /// Stable localization key for the terminal headline.
    #[must_use]
    pub const fn title_key(self) -> &'static str {
        match self {
            Self::Succeeded => "faceauth.outcome.succeeded.title",
            Self::TryAgain => "faceauth.outcome.try-again.title",
            Self::Cancelled => "faceauth.outcome.cancelled.title",
            Self::TimedOut => "faceauth.outcome.timed-out.title",
            Self::Unavailable => "faceauth.outcome.unavailable.title",
        }
    }

    /// Stable semantic icon name.
    #[must_use]
    pub const fn icon_name(self) -> &'static str {
        match self {
            Self::Succeeded => "dialog-ok-apply",
            Self::TryAgain => "data-warning",
            Self::Cancelled => "dialog-cancel",
            Self::TimedOut => "chronometer",
            Self::Unavailable => "dialog-error",
        }
    }
}

/// Current complete UI state.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "kebab-case")]
pub enum ExperienceState {
    /// No visible biometric operation.
    Idle,
    /// One exact operation is active.
    Active {
        /// Bound operation identity.
        flow: ExperienceFlow,
        /// Current safe user guidance.
        cue: ExperienceCue,
    },
    /// One exact operation reached a one-shot terminal state.
    Terminal {
        /// Bound operation identity.
        flow: ExperienceFlow,
        /// Sanitized terminal category.
        outcome: ExperienceOutcome,
    },
}

/// Small stable presentation summary consumed by QML or other desktop clients.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ExperiencePresentation {
    /// Stable state/cue token.
    pub token: &'static str,
    /// Localization key selected by the client.
    pub title_key: &'static str,
    /// Theme icon name.
    pub icon_name: &'static str,
    /// Whether an operation is currently running.
    pub busy: bool,
    /// Whether the UI must keep the password path visible.
    pub show_password_fallback: bool,
}

/// One-operation experience state machine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExperienceModel {
    state: ExperienceState,
}

impl Default for ExperienceModel {
    fn default() -> Self {
        Self { state: ExperienceState::Idle }
    }
}

impl ExperienceModel {
    /// Return the current complete state.
    #[must_use]
    pub const fn state(self) -> ExperienceState {
        self.state
    }

    /// Begin one accepted authentication transaction.
    ///
    /// # Errors
    ///
    /// Returns [`ExperienceError::Busy`] until the previous flow is reset.
    pub fn begin_authentication(
        &mut self,
        transaction_id: TransactionId,
    ) -> Result<(), ExperienceError> {
        self.begin(ExperienceFlow::Authentication { transaction_id })
    }

    /// Begin one accepted Manager1 enrollment operation.
    ///
    /// # Errors
    ///
    /// Returns [`ExperienceError::Busy`] until the previous flow is reset.
    pub fn begin_enrollment(&mut self, operation_id: OperationId) -> Result<(), ExperienceError> {
        self.begin(ExperienceFlow::Enrollment { operation_id })
    }

    fn begin(&mut self, flow: ExperienceFlow) -> Result<(), ExperienceError> {
        if self.state != ExperienceState::Idle {
            return Err(ExperienceError::Busy);
        }
        self.state = ExperienceState::Active { flow, cue: ExperienceCue::Preparing };
        Ok(())
    }

    /// Apply one transaction-bound authentication progress code.
    ///
    /// # Errors
    ///
    /// Returns [`ExperienceError`] unless the exact authentication flow is active.
    pub fn authentication_progress(
        &mut self,
        transaction_id: TransactionId,
        progress: ProgressCode,
    ) -> Result<(), ExperienceError> {
        let flow = ExperienceFlow::Authentication { transaction_id };
        self.require_active(flow)?;
        self.state = ExperienceState::Active { flow, cue: authentication_cue(progress) };
        Ok(())
    }

    /// Apply one terminal authentication decision without exposing biometric measurements.
    ///
    /// # Errors
    ///
    /// Returns [`ExperienceError`] unless the exact authentication flow is active.
    pub fn authentication_completed(
        &mut self,
        transaction_id: TransactionId,
        decision: DecisionCode,
    ) -> Result<(), ExperienceError> {
        let flow = ExperienceFlow::Authentication { transaction_id };
        self.require_active(flow)?;
        self.state = ExperienceState::Terminal { flow, outcome: authentication_outcome(decision) };
        Ok(())
    }

    /// Apply one Manager1 progress or terminal update.
    ///
    /// # Errors
    ///
    /// Returns [`ExperienceError`] unless the exact enrollment operation is active.
    pub fn management_update(&mut self, update: ManagementUpdate) -> Result<(), ExperienceError> {
        match update {
            ManagementUpdate::Progress { operation_id, progress } => {
                let flow = ExperienceFlow::Enrollment { operation_id };
                self.require_active(flow)?;
                self.state = ExperienceState::Active { flow, cue: management_cue(progress) };
            }
            ManagementUpdate::Completed { operation_id, result } => {
                let flow = ExperienceFlow::Enrollment { operation_id };
                self.require_active(flow)?;
                self.state =
                    ExperienceState::Terminal { flow, outcome: management_outcome(result) };
            }
        }
        Ok(())
    }

    /// Reset an idle or terminal experience before starting another flow.
    ///
    /// # Errors
    ///
    /// Returns [`ExperienceError::Busy`] while an operation is active.
    pub const fn reset(&mut self) -> Result<(), ExperienceError> {
        if matches!(self.state, ExperienceState::Active { .. }) {
            return Err(ExperienceError::Busy);
        }
        self.state = ExperienceState::Idle;
        Ok(())
    }

    /// Return stable presentation tokens for the current state.
    #[must_use]
    pub const fn presentation(self) -> ExperiencePresentation {
        match self.state {
            ExperienceState::Idle => ExperiencePresentation {
                token: "idle",
                title_key: "faceauth.idle.title",
                icon_name: "edit-image-face-recognize",
                busy: false,
                show_password_fallback: false,
            },
            ExperienceState::Active { flow, cue } => ExperiencePresentation {
                token: cue.token(),
                title_key: cue.title_key(),
                icon_name: cue.icon_name(),
                busy: true,
                show_password_fallback: matches!(flow, ExperienceFlow::Authentication { .. }),
            },
            ExperienceState::Terminal { flow, outcome } => ExperiencePresentation {
                token: match outcome {
                    ExperienceOutcome::Succeeded => "succeeded",
                    ExperienceOutcome::TryAgain => "try-again",
                    ExperienceOutcome::Cancelled => "cancelled",
                    ExperienceOutcome::TimedOut => "timed-out",
                    ExperienceOutcome::Unavailable => "unavailable",
                },
                title_key: outcome.title_key(),
                icon_name: outcome.icon_name(),
                busy: false,
                show_password_fallback: matches!(flow, ExperienceFlow::Authentication { .. })
                    && !matches!(outcome, ExperienceOutcome::Succeeded),
            },
        }
    }

    fn require_active(&self, expected: ExperienceFlow) -> Result<(), ExperienceError> {
        match self.state {
            ExperienceState::Idle => Err(ExperienceError::Idle),
            ExperienceState::Terminal { .. } => Err(ExperienceError::Terminal),
            ExperienceState::Active { flow, .. } if flow == expected => Ok(()),
            ExperienceState::Active { .. } => Err(ExperienceError::WrongFlow),
        }
    }
}

const fn authentication_cue(progress: ProgressCode) -> ExperienceCue {
    match progress {
        ProgressCode::PositionFace => ExperienceCue::PositionFace,
        ProgressCode::HoldStill => ExperienceCue::HoldStill,
        ProgressCode::Blink => ExperienceCue::Blink,
        ProgressCode::TurnLeft => ExperienceCue::TurnLeft,
        ProgressCode::TurnRight => ExperienceCue::TurnRight,
        ProgressCode::ReturnToCenter => ExperienceCue::ReturnToCenter,
        ProgressCode::Processing => ExperienceCue::Processing,
    }
}

const fn management_cue(progress: ManagementProgress) -> ExperienceCue {
    match progress {
        ManagementProgress::Preparing => ExperienceCue::Preparing,
        ManagementProgress::PositionFace => ExperienceCue::PositionFace,
        ManagementProgress::HoldStill => ExperienceCue::HoldStill,
        ManagementProgress::ActiveChallenge => ExperienceCue::FollowChallenge,
        ManagementProgress::Blink => ExperienceCue::Blink,
        ManagementProgress::TurnLeft => ExperienceCue::TurnLeft,
        ManagementProgress::TurnRight => ExperienceCue::TurnRight,
        ManagementProgress::ReturnToCenter => ExperienceCue::ReturnToCenter,
        ManagementProgress::Processing => ExperienceCue::Processing,
    }
}

const fn authentication_outcome(decision: DecisionCode) -> ExperienceOutcome {
    match decision {
        DecisionCode::Accepted => ExperienceOutcome::Succeeded,
        DecisionCode::Cancelled => ExperienceOutcome::Cancelled,
        DecisionCode::TimedOut => ExperienceOutcome::TimedOut,
        DecisionCode::InternalError => ExperienceOutcome::Unavailable,
        DecisionCode::InfraredMissing
        | DecisionCode::VisibleMissing
        | DecisionCode::InsufficientQuality
        | DecisionCode::PassiveLivenessFailed
        | DecisionCode::ActiveChallengeFailed
        | DecisionCode::FaceMismatch => ExperienceOutcome::TryAgain,
    }
}

const fn management_outcome(result: ManagementResult) -> ExperienceOutcome {
    match result {
        ManagementResult::Completed => ExperienceOutcome::Succeeded,
        ManagementResult::Cancelled => ExperienceOutcome::Cancelled,
        ManagementResult::TimedOut => ExperienceOutcome::TimedOut,
        ManagementResult::Failed => ExperienceOutcome::Unavailable,
    }
}

/// Invalid or stale UI event transition.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum ExperienceError {
    /// Another active or unreset terminal flow owns the UI.
    #[error("experience flow is busy")]
    Busy,
    /// No flow is active.
    #[error("experience flow is idle")]
    Idle,
    /// A stale or unrelated operation attempted to update the visible flow.
    #[error("experience update belongs to another flow")]
    WrongFlow,
    /// The flow already reached a terminal state.
    #[error("experience flow is already terminal")]
    Terminal,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn operation(value: &str) -> Result<OperationId, faceauth_management::ManagementError> {
        OperationId::parse(value)
    }

    #[test]
    fn authentication_cues_are_exact_and_password_fallback_stays_visible()
    -> Result<(), Box<dyn std::error::Error>> {
        let transaction_id = TransactionId::generate();
        let mut model = ExperienceModel::default();
        model.begin_authentication(transaction_id)?;
        for (progress, cue) in [
            (ProgressCode::PositionFace, ExperienceCue::PositionFace),
            (ProgressCode::HoldStill, ExperienceCue::HoldStill),
            (ProgressCode::Blink, ExperienceCue::Blink),
            (ProgressCode::TurnLeft, ExperienceCue::TurnLeft),
            (ProgressCode::TurnRight, ExperienceCue::TurnRight),
            (ProgressCode::ReturnToCenter, ExperienceCue::ReturnToCenter),
            (ProgressCode::Processing, ExperienceCue::Processing),
        ] {
            model.authentication_progress(transaction_id, progress)?;
            assert!(
                matches!(model.state(), ExperienceState::Active { cue: actual, .. } if actual == cue)
            );
            assert!(model.presentation().show_password_fallback);
        }
        Ok(())
    }

    #[test]
    fn enrollment_schema_two_exposes_exact_active_actions() -> Result<(), Box<dyn std::error::Error>>
    {
        let operation_id = operation("123e4567-e89b-12d3-a456-426614174000")?;
        let mut model = ExperienceModel::default();
        model.begin_enrollment(operation_id)?;
        for (progress, cue) in [
            (ManagementProgress::ActiveChallenge, ExperienceCue::FollowChallenge),
            (ManagementProgress::Blink, ExperienceCue::Blink),
            (ManagementProgress::TurnLeft, ExperienceCue::TurnLeft),
            (ManagementProgress::TurnRight, ExperienceCue::TurnRight),
            (ManagementProgress::ReturnToCenter, ExperienceCue::ReturnToCenter),
        ] {
            model.management_update(ManagementUpdate::Progress { operation_id, progress })?;
            assert!(
                matches!(model.state(), ExperienceState::Active { cue: actual, .. } if actual == cue)
            );
            assert_eq!(
                model.presentation().token,
                match progress {
                    ManagementProgress::ActiveChallenge => "active-challenge",
                    ManagementProgress::Blink => "blink",
                    ManagementProgress::TurnLeft => "turn-left",
                    ManagementProgress::TurnRight => "turn-right",
                    ManagementProgress::ReturnToCenter => "return-to-center",
                    _ => unreachable!(),
                }
            );
        }
        Ok(())
    }

    #[test]
    fn stale_ids_and_updates_after_terminal_are_rejected() -> Result<(), Box<dyn std::error::Error>>
    {
        let active = TransactionId::generate();
        let stale = TransactionId::generate();
        let mut model = ExperienceModel::default();
        model.begin_authentication(active)?;
        assert_eq!(
            model.authentication_progress(stale, ProgressCode::HoldStill),
            Err(ExperienceError::WrongFlow)
        );
        model.authentication_completed(active, DecisionCode::Accepted)?;
        assert_eq!(
            model.authentication_progress(active, ProgressCode::Processing),
            Err(ExperienceError::Terminal)
        );
        assert_eq!(
            model.authentication_completed(active, DecisionCode::Accepted),
            Err(ExperienceError::Terminal)
        );
        Ok(())
    }

    #[test]
    fn biometric_failure_details_collapse_to_one_safe_experience()
    -> Result<(), Box<dyn std::error::Error>> {
        for decision in [
            DecisionCode::InfraredMissing,
            DecisionCode::VisibleMissing,
            DecisionCode::InsufficientQuality,
            DecisionCode::PassiveLivenessFailed,
            DecisionCode::ActiveChallengeFailed,
            DecisionCode::FaceMismatch,
        ] {
            let transaction_id = TransactionId::generate();
            let mut model = ExperienceModel::default();
            model.begin_authentication(transaction_id)?;
            model.authentication_completed(transaction_id, decision)?;
            assert!(matches!(
                model.state(),
                ExperienceState::Terminal { outcome: ExperienceOutcome::TryAgain, .. }
            ));
            assert_eq!(model.presentation().token, "try-again");
            assert!(model.presentation().show_password_fallback);
        }
        Ok(())
    }

    #[test]
    fn terminal_flow_requires_explicit_reset() -> Result<(), Box<dyn std::error::Error>> {
        let transaction_id = TransactionId::generate();
        let mut model = ExperienceModel::default();
        model.begin_authentication(transaction_id)?;
        model.authentication_completed(transaction_id, DecisionCode::Cancelled)?;
        assert_eq!(
            model.begin_authentication(TransactionId::generate()),
            Err(ExperienceError::Busy)
        );
        model.reset()?;
        model.begin_authentication(TransactionId::generate())?;
        Ok(())
    }
}
