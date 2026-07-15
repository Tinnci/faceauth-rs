//! Desktop-independent, versioned management contract for enrollment clients.

use faceauth_authz::AuthorizationGrant;
use faceauth_protocol::AuthenticationPurpose;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

/// Stable system D-Bus service name reserved for the future adapter.
pub const MANAGER_BUS_NAME: &str = "org.faceauth.Manager1";
/// Stable system D-Bus object path reserved for the future adapter.
pub const MANAGER_OBJECT_PATH: &str = "/org/faceauth/Manager1";
/// Stable system D-Bus interface name reserved for the future adapter.
pub const MANAGER_INTERFACE: &str = "org.faceauth.Manager1";
/// Polkit action required before constructing an enrollment authorization.
pub const ENROLLMENT_POLKIT_ACTION: &str = "org.faceauth.enroll";
/// Current management contract schema.
pub const MANAGEMENT_SCHEMA_VERSION: u16 = 1;
/// Dedicated authorization service name admitted for enrollment.
pub const ENROLLMENT_SERVICE: &str = "faceauth-enroll";

/// Hard maximum lifetime for one externally visible enrollment operation.
pub const MAX_OPERATION_DURATION_MICROS: u64 = 120_000_000;

/// Opaque proof that the daemon accepted the exact root Polkit broker grant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AuthorizedEnrollment {
    target_uid: u32,
}

impl AuthorizedEnrollment {
    /// Validate and consume the authorization fields needed by management.
    ///
    /// # Errors
    ///
    /// Returns [`ManagementError::Unauthorized`] unless the grant is from root and is bound to the
    /// exact enrollment service and Polkit purpose.
    pub fn from_grant(grant: &AuthorizationGrant) -> Result<Self, ManagementError> {
        if grant.peer.uid != 0
            || grant.service.as_str() != ENROLLMENT_SERVICE
            || grant.purpose != AuthenticationPurpose::Polkit
        {
            return Err(ManagementError::Unauthorized);
        }
        Ok(Self { target_uid: grant.target_uid })
    }

    /// Numeric account authorized for enrollment.
    #[must_use]
    pub const fn target_uid(self) -> u32 {
        self.target_uid
    }
}

/// Unpredictable identifier for one management operation.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub struct OperationId(Uuid);

impl OperationId {
    fn generate() -> Self {
        Self(Uuid::new_v4())
    }
}

/// Public progress stages suitable for a KCM or OSD; contains no biometric measurements.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ManagementProgress {
    /// Preparing cameras and model sessions.
    Preparing,
    /// Move one face into the calibrated region.
    PositionFace,
    /// Hold a centered pose for bounded sample collection.
    HoldStill,
    /// Perform the randomized action shown separately by the client mapping.
    ActiveChallenge,
    /// Derived samples are being checked and encrypted.
    Processing,
}

/// Terminal management result.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ManagementResult {
    /// A new encrypted template was committed.
    Completed,
    /// The authorized client cancelled the operation.
    Cancelled,
    /// The fixed operation deadline elapsed.
    TimedOut,
    /// A non-biometric internal failure prevented completion.
    Failed,
}

/// Public operation update emitted by the future D-Bus adapter.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ManagementUpdate {
    /// The operation remains active.
    Progress {
        /// Bound operation identifier.
        operation_id: OperationId,
        /// Safe UI progress code.
        progress: ManagementProgress,
    },
    /// The operation was consumed terminally.
    Completed {
        /// Bound operation identifier.
        operation_id: OperationId,
        /// Stable terminal result.
        result: ManagementResult,
    },
}

/// Management operation timing bounds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ManagementConfig {
    /// Fixed enrollment operation lifetime in monotonic microseconds.
    pub operation_duration_micros: u64,
}

impl ManagementConfig {
    /// Validate management resource bounds.
    ///
    /// # Errors
    ///
    /// Returns [`ManagementError::InvalidConfig`] for sub-five-second or excessive duration.
    pub const fn validate(self) -> Result<(), ManagementError> {
        if self.operation_duration_micros < 5_000_000
            || self.operation_duration_micros > MAX_OPERATION_DURATION_MICROS
        {
            Err(ManagementError::InvalidConfig)
        } else {
            Ok(())
        }
    }
}

impl Default for ManagementConfig {
    fn default() -> Self {
        Self { operation_duration_micros: 120_000_000 }
    }
}

struct ActiveOperation {
    id: OperationId,
    uid: u32,
    deadline_micros: u64,
}

/// Single-capacity management operation coordinator matching one calibrated camera pair.
pub struct ManagementCoordinator {
    config: ManagementConfig,
    active: Option<ActiveOperation>,
}

impl ManagementCoordinator {
    /// Construct an empty coordinator.
    ///
    /// # Errors
    ///
    /// Returns [`ManagementError::InvalidConfig`] for invalid bounds.
    pub fn new(config: ManagementConfig) -> Result<Self, ManagementError> {
        config.validate()?;
        Ok(Self { config, active: None })
    }

    /// Start one already-authorized enrollment operation.
    ///
    /// # Errors
    ///
    /// Returns [`ManagementError::Busy`] when another operation owns the camera capacity, or
    /// [`ManagementError::DeadlineOverflow`] if the monotonic deadline cannot be represented.
    pub fn start(
        &mut self,
        authorization: AuthorizedEnrollment,
        now_micros: u64,
    ) -> Result<ManagementUpdate, ManagementError> {
        if self.active.is_some() {
            return Err(ManagementError::Busy);
        }
        let deadline_micros = now_micros
            .checked_add(self.config.operation_duration_micros)
            .ok_or(ManagementError::DeadlineOverflow)?;
        let operation_id = OperationId::generate();
        self.active = Some(ActiveOperation {
            id: operation_id,
            uid: authorization.target_uid,
            deadline_micros,
        });
        Ok(ManagementUpdate::Progress { operation_id, progress: ManagementProgress::Preparing })
    }

    /// Emit safe progress for the exact UID and operation.
    ///
    /// # Errors
    ///
    /// Returns [`ManagementError`] for an absent or mismatched operation.
    pub fn progress(
        &mut self,
        uid: u32,
        operation_id: OperationId,
        progress: ManagementProgress,
        now_micros: u64,
    ) -> Result<ManagementUpdate, ManagementError> {
        self.verify(uid, operation_id)?;
        if self.deadline_reached(now_micros)? {
            return self.take_terminal(ManagementResult::TimedOut);
        }
        Ok(ManagementUpdate::Progress { operation_id, progress })
    }

    /// Cancel the exact authorized operation.
    ///
    /// # Errors
    ///
    /// Returns [`ManagementError`] for an absent or mismatched operation.
    pub fn cancel(
        &mut self,
        uid: u32,
        operation_id: OperationId,
        now_micros: u64,
    ) -> Result<ManagementUpdate, ManagementError> {
        self.verify(uid, operation_id)?;
        if self.deadline_reached(now_micros)? {
            return self.take_terminal(ManagementResult::TimedOut);
        }
        self.take_terminal(ManagementResult::Cancelled)
    }

    /// Complete the exact operation with a daemon-owned result.
    ///
    /// # Errors
    ///
    /// Returns [`ManagementError`] for an absent or mismatched operation, or if a client-reserved
    /// cancellation/timeout result is supplied.
    pub fn complete(
        &mut self,
        uid: u32,
        operation_id: OperationId,
        result: ManagementResult,
        now_micros: u64,
    ) -> Result<ManagementUpdate, ManagementError> {
        if matches!(result, ManagementResult::Cancelled | ManagementResult::TimedOut) {
            return Err(ManagementError::ReservedResult);
        }
        self.verify(uid, operation_id)?;
        if self.deadline_reached(now_micros)? {
            return self.take_terminal(ManagementResult::TimedOut);
        }
        self.take_terminal(result)
    }

    /// Consume an expired operation, when present.
    #[must_use]
    pub fn expire(&mut self, now_micros: u64) -> Option<ManagementUpdate> {
        let expired =
            self.active.as_ref().is_some_and(|operation| now_micros >= operation.deadline_micros);
        if expired { self.take_terminal(ManagementResult::TimedOut).ok() } else { None }
    }

    /// Return whether management capacity is occupied.
    #[must_use]
    pub const fn is_busy(&self) -> bool {
        self.active.is_some()
    }

    fn verify(&self, uid: u32, operation_id: OperationId) -> Result<(), ManagementError> {
        let active = self.active.as_ref().ok_or(ManagementError::NoActiveOperation)?;
        if active.uid != uid {
            return Err(ManagementError::WrongUid);
        }
        if active.id != operation_id {
            return Err(ManagementError::WrongOperation);
        }
        Ok(())
    }

    fn deadline_reached(&self, now_micros: u64) -> Result<bool, ManagementError> {
        self.active
            .as_ref()
            .map(|active| now_micros >= active.deadline_micros)
            .ok_or(ManagementError::NoActiveOperation)
    }

    fn take_terminal(
        &mut self,
        result: ManagementResult,
    ) -> Result<ManagementUpdate, ManagementError> {
        let operation = self.active.take().ok_or(ManagementError::NoActiveOperation)?;
        Ok(ManagementUpdate::Completed { operation_id: operation.id, result })
    }
}

/// Management authorization or operation lifecycle failure.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum ManagementError {
    /// Grant was not the exact root Polkit enrollment grant.
    #[error("management enrollment authorization is invalid")]
    Unauthorized,
    /// Timing configuration is outside hard bounds.
    #[error("management operation configuration is invalid")]
    InvalidConfig,
    /// Fixed deadline overflowed.
    #[error("management operation deadline overflowed")]
    DeadlineOverflow,
    /// Another operation owns the single camera capacity.
    #[error("management operation capacity is busy")]
    Busy,
    /// No operation is active.
    #[error("no management operation is active")]
    NoActiveOperation,
    /// Numeric account did not match the active authorization.
    #[error("management operation belongs to another UID")]
    WrongUid,
    /// Operation identifier did not match.
    #[error("management operation identifier does not match")]
    WrongOperation,
    /// Caller tried to synthesize coordinator-owned cancellation or timeout.
    #[error("cancelled and timed-out management results are reserved")]
    ReservedResult,
}

#[cfg(test)]
mod tests {
    use faceauth_authz::{AuthorizationGrant, ExecutableFingerprint};
    use faceauth_protocol::{ServiceName, TransactionId};
    use faceauth_transport::PeerIdentity;

    use super::*;

    fn grant() -> Result<AuthorizationGrant, faceauth_protocol::ServiceNameError> {
        Ok(AuthorizationGrant {
            transaction_id: TransactionId::generate(),
            peer: PeerIdentity { pid: 100, uid: 0, gid: 0 },
            target_uid: 1000,
            service: ServiceName::parse(ENROLLMENT_SERVICE)?,
            purpose: AuthenticationPurpose::Polkit,
            executable: ExecutableFingerprint { device: 1, inode: 2 },
        })
    }

    fn id_from_update(update: ManagementUpdate) -> OperationId {
        match update {
            ManagementUpdate::Progress { operation_id, .. }
            | ManagementUpdate::Completed { operation_id, .. } => operation_id,
        }
    }

    #[test]
    fn only_exact_root_polkit_enrollment_grant_is_accepted()
    -> Result<(), Box<dyn std::error::Error>> {
        let valid = grant()?;
        assert_eq!(AuthorizedEnrollment::from_grant(&valid)?.target_uid(), 1000);

        let mut wrong_peer = valid.clone();
        wrong_peer.peer.uid = 1000;
        assert_eq!(
            AuthorizedEnrollment::from_grant(&wrong_peer),
            Err(ManagementError::Unauthorized)
        );

        let mut wrong_purpose = valid;
        wrong_purpose.purpose = AuthenticationPurpose::Test;
        assert_eq!(
            AuthorizedEnrollment::from_grant(&wrong_purpose),
            Err(ManagementError::Unauthorized)
        );
        Ok(())
    }

    #[test]
    fn operation_is_uid_and_id_bound_and_single_capacity() -> Result<(), Box<dyn std::error::Error>>
    {
        let authorization = AuthorizedEnrollment::from_grant(&grant()?)?;
        let mut coordinator = ManagementCoordinator::new(ManagementConfig::default())?;
        let started = coordinator.start(authorization, 1_000_000)?;
        let operation_id = id_from_update(started);
        assert!(coordinator.is_busy());
        assert_eq!(coordinator.start(authorization, 1_000_001), Err(ManagementError::Busy));
        assert_eq!(
            coordinator.progress(1001, operation_id, ManagementProgress::HoldStill, 2_000_000,),
            Err(ManagementError::WrongUid)
        );
        assert_eq!(
            coordinator.progress(
                1000,
                OperationId::generate(),
                ManagementProgress::HoldStill,
                2_000_000,
            ),
            Err(ManagementError::WrongOperation)
        );
        assert_eq!(
            coordinator.progress(1000, operation_id, ManagementProgress::HoldStill, 2_000_000,)?,
            ManagementUpdate::Progress { operation_id, progress: ManagementProgress::HoldStill }
        );
        Ok(())
    }

    #[test]
    fn cancellation_and_completion_are_one_shot() -> Result<(), Box<dyn std::error::Error>> {
        let authorization = AuthorizedEnrollment::from_grant(&grant()?)?;
        let mut coordinator = ManagementCoordinator::new(ManagementConfig::default())?;
        let operation_id = id_from_update(coordinator.start(authorization, 1_000_000)?);
        assert_eq!(
            coordinator.cancel(1000, operation_id, 2_000_000)?,
            ManagementUpdate::Completed { operation_id, result: ManagementResult::Cancelled }
        );
        assert!(!coordinator.is_busy());
        assert_eq!(
            coordinator.complete(1000, operation_id, ManagementResult::Completed, 2_000_001,),
            Err(ManagementError::NoActiveOperation)
        );
        Ok(())
    }

    #[test]
    fn fixed_deadline_overrides_progress_and_can_be_reaped()
    -> Result<(), Box<dyn std::error::Error>> {
        let authorization = AuthorizedEnrollment::from_grant(&grant()?)?;
        let config = ManagementConfig { operation_duration_micros: 5_000_000 };
        let mut coordinator = ManagementCoordinator::new(config)?;
        let operation_id = id_from_update(coordinator.start(authorization, 1_000_000)?);
        assert_eq!(
            coordinator.progress(1000, operation_id, ManagementProgress::Processing, 6_000_000,)?,
            ManagementUpdate::Completed { operation_id, result: ManagementResult::TimedOut }
        );

        let operation_id = id_from_update(coordinator.start(authorization, 10_000_000)?);
        assert_eq!(
            coordinator.expire(15_000_000),
            Some(ManagementUpdate::Completed { operation_id, result: ManagementResult::TimedOut })
        );
        Ok(())
    }
}
