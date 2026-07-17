//! Global non-queueing ownership for cameras and mutable biometric model sessions.

use std::sync::{Arc, Mutex};

use faceauth_management::OperationId;
use faceauth_protocol::TransactionId;
use thiserror::Error;

/// Exact operation allowed to own the shared biometric hardware pipeline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BiometricResourceOwner {
    /// One accepted authentication transaction.
    Authentication(TransactionId),
    /// One authorized enrollment operation.
    Enrollment(OperationId),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ActiveLease {
    generation: u64,
    owner: BiometricResourceOwner,
}

#[derive(Debug, Default)]
struct ArbiterState {
    next_generation: u64,
    active: Option<ActiveLease>,
}

/// Daemon-wide non-queueing arbiter shared by authentication and enrollment.
///
/// Acquisition is immediate: a second caller receives [`BiometricResourceError::Busy`] rather
/// than waiting behind an authorization that may expire. The returned lease releases capacity on
/// normal return, cancellation, early error, or unwind.
#[derive(Clone, Debug, Default)]
pub struct BiometricResourceArbiter {
    state: Arc<Mutex<ArbiterState>>,
}

impl BiometricResourceArbiter {
    /// Acquire exclusive ownership for one exact transaction or operation.
    ///
    /// # Errors
    ///
    /// Returns [`BiometricResourceError::Busy`] if another owner is active, or
    /// [`BiometricResourceError::Unavailable`] if internal synchronization or the generation
    /// counter can no longer be trusted.
    pub fn try_acquire(
        &self,
        owner: BiometricResourceOwner,
    ) -> Result<BiometricResourceLease, BiometricResourceError> {
        let mut state = self.state.lock().map_err(|_| BiometricResourceError::Unavailable)?;
        if let Some(active) = state.active {
            return Err(BiometricResourceError::Busy { owner: active.owner });
        }
        state.next_generation =
            state.next_generation.checked_add(1).ok_or(BiometricResourceError::Unavailable)?;
        let generation = state.next_generation;
        state.active = Some(ActiveLease { generation, owner });
        drop(state);
        Ok(BiometricResourceLease { state: Arc::clone(&self.state), generation, owner })
    }

    /// Return the current owner for readiness and safe UI busy reporting.
    ///
    /// # Errors
    ///
    /// Returns [`BiometricResourceError::Unavailable`] if synchronization was poisoned.
    pub fn owner(&self) -> Result<Option<BiometricResourceOwner>, BiometricResourceError> {
        self.state
            .lock()
            .map(|state| state.active.map(|active| active.owner))
            .map_err(|_| BiometricResourceError::Unavailable)
    }
}

/// Scoped proof of exclusive camera and mutable-model ownership.
pub struct BiometricResourceLease {
    state: Arc<Mutex<ArbiterState>>,
    generation: u64,
    owner: BiometricResourceOwner,
}

impl BiometricResourceLease {
    /// Exact transaction or operation bound to this lease.
    #[must_use]
    pub const fn owner(&self) -> BiometricResourceOwner {
        self.owner
    }
}

impl Drop for BiometricResourceLease {
    fn drop(&mut self) {
        if let Ok(mut state) = self.state.lock()
            && state.active.is_some_and(|active| active.generation == self.generation)
        {
            state.active = None;
        }
    }
}

/// Shared biometric pipeline could not be acquired.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum BiometricResourceError {
    /// Another exact operation currently owns the pipeline.
    #[error("biometric resources are owned by {owner:?}")]
    Busy {
        /// Current owner; safe for daemon-internal routing and closed UI busy mapping.
        owner: BiometricResourceOwner,
    },
    /// Internal lease state can no longer be trusted.
    #[error("biometric resource arbiter is unavailable")]
    Unavailable,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enrollment() -> Result<OperationId, faceauth_management::ManagementError> {
        OperationId::parse("123e4567-e89b-12d3-a456-426614174000")
    }

    #[test]
    fn authentication_and_enrollment_share_one_non_queueing_capacity()
    -> Result<(), Box<dyn std::error::Error>> {
        let arbiter = BiometricResourceArbiter::default();
        let transaction = TransactionId::generate();
        let authentication =
            arbiter.try_acquire(BiometricResourceOwner::Authentication(transaction))?;
        assert!(matches!(
            arbiter.try_acquire(BiometricResourceOwner::Enrollment(enrollment()?)),
            Err(BiometricResourceError::Busy {
                owner: BiometricResourceOwner::Authentication(owner)
            }) if owner == transaction
        ));
        drop(authentication);
        let operation = enrollment()?;
        let enrollment_lease =
            arbiter.try_acquire(BiometricResourceOwner::Enrollment(operation))?;
        assert_eq!(enrollment_lease.owner(), BiometricResourceOwner::Enrollment(operation));
        Ok(())
    }

    #[test]
    fn dropping_any_exit_path_releases_capacity_without_stale_release()
    -> Result<(), Box<dyn std::error::Error>> {
        let arbiter = BiometricResourceArbiter::default();
        let first = arbiter.try_acquire(BiometricResourceOwner::Enrollment(enrollment()?))?;
        assert_eq!(arbiter.owner()?, Some(first.owner()));
        drop(first);
        assert_eq!(arbiter.owner()?, None);

        let transaction = TransactionId::generate();
        let second = arbiter.try_acquire(BiometricResourceOwner::Authentication(transaction))?;
        assert_eq!(arbiter.owner()?, Some(second.owner()));
        drop(second);
        assert_eq!(arbiter.owner()?, None);
        Ok(())
    }
}
