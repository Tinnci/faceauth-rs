//! Optional human-presence hints and generic capture-activity inhibition.
//!
//! Presence never enters biometric evidence or an authentication decision. D-Bus access is
//! intended for a background refresh worker; authentication reads only the bounded cache.

use std::sync::{Arc, Mutex};

use serde::Serialize;
use thiserror::Error;
use zbus::blocking::{Connection, Proxy};

/// Existing independent thinkpad-hpd system service.
pub const HPD_SERVICE: &str = "org.thinkpad.HumanPresence1";
/// Existing thinkpad-hpd object path.
pub const HPD_PATH: &str = "/org/thinkpad/HumanPresence1";
/// Existing thinkpad-hpd interface.
pub const HPD_INTERFACE: &str = "org.thinkpad.HumanPresence1";

/// Stable generic faceauth capture-activity service reserved for desktop/HPD consumers.
pub const CAPTURE_SERVICE: &str = "org.faceauth.Capture1";
/// Stable generic capture-activity object path.
pub const CAPTURE_PATH: &str = "/org/faceauth/Capture1";
/// Stable generic capture-activity interface.
pub const CAPTURE_INTERFACE: &str = "org.faceauth.Capture1";

/// One validated presence observation from an optional external service.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct PresenceSnapshot {
    /// Whether the provider considers its sensor usable.
    pub available: bool,
    /// Whether a human is currently considered present.
    pub present: bool,
    /// Provider-specific diagnostic value; never interpreted as identity evidence.
    pub raw_value: i32,
    /// Daemon monotonic timestamp assigned after receiving the reply.
    pub observed_at_micros: u64,
}

impl PresenceSnapshot {
    fn validate(self) -> Result<(), PresenceError> {
        if !(-1_000_000..=1_000_000).contains(&self.raw_value) {
            return Err(PresenceError::InvalidState);
        }
        Ok(())
    }
}

/// Blocking system-D-Bus adapter used only by a background refresh worker or diagnostics.
pub struct HpdClient {
    connection: Connection,
}

impl HpdClient {
    /// Connect to the system bus without taking ownership of an HPD name.
    ///
    /// # Errors
    ///
    /// Returns [`PresenceError`] when the system bus connection fails.
    pub fn system() -> Result<Self, PresenceError> {
        Ok(Self { connection: Connection::system()? })
    }

    /// Read the existing thinkpad-hpd `GetState() -> (available, present, raw)` method.
    ///
    /// This method must not run on the PAM/authentication hot path. Service absence is an optional
    /// hint failure, not an authentication failure.
    ///
    /// # Errors
    ///
    /// Returns [`PresenceError`] for D-Bus or state-validation failure.
    pub fn get_state(&self, observed_at_micros: u64) -> Result<PresenceSnapshot, PresenceError> {
        let proxy = Proxy::new(&self.connection, HPD_SERVICE, HPD_PATH, HPD_INTERFACE)?;
        let (available, present, raw_value): (bool, bool, i32) = proxy.call("GetState", &())?;
        let snapshot = PresenceSnapshot { available, present, raw_value, observed_at_micros };
        snapshot.validate()?;
        Ok(snapshot)
    }
}

/// Thread-safe bounded cache read by daemon scheduling logic.
#[derive(Default)]
pub struct PresenceCache {
    state: Mutex<Option<PresenceSnapshot>>,
}

impl PresenceCache {
    /// Replace the cache only with a monotonic, validated observation.
    ///
    /// # Errors
    ///
    /// Returns [`PresenceError`] for invalid raw state, timestamp rollback, or poisoned state.
    pub fn update(&self, snapshot: PresenceSnapshot) -> Result<(), PresenceError> {
        snapshot.validate()?;
        let mut state = self.state.lock().map_err(|_| PresenceError::StatePoisoned)?;
        if state.is_some_and(|previous| snapshot.observed_at_micros <= previous.observed_at_micros)
        {
            return Err(PresenceError::StaleObservation);
        }
        *state = Some(snapshot);
        drop(state);
        Ok(())
    }

    /// Drop cached state after provider loss or refresh failure.
    ///
    /// Authentication remains independent and does not fail because this cache is empty.
    ///
    /// # Errors
    ///
    /// Returns [`PresenceError::StatePoisoned`] when the cache mutex was poisoned.
    pub fn clear(&self) -> Result<(), PresenceError> {
        *self.state.lock().map_err(|_| PresenceError::StatePoisoned)? = None;
        Ok(())
    }

    /// Return true only for a fresh available+present hint suitable for optional prewarming.
    ///
    /// # Errors
    ///
    /// Returns [`PresenceError`] when freshness configuration is invalid, time moves backwards,
    /// or the cache mutex was poisoned.
    pub fn should_prewarm(
        &self,
        now_micros: u64,
        maximum_age_micros: u64,
    ) -> Result<bool, PresenceError> {
        if maximum_age_micros == 0 || maximum_age_micros > 60_000_000 {
            return Err(PresenceError::InvalidFreshness);
        }
        let state = self.state.lock().map_err(|_| PresenceError::StatePoisoned)?;
        let Some(snapshot) = *state else {
            return Ok(false);
        };
        let age = now_micros
            .checked_sub(snapshot.observed_at_micros)
            .ok_or(PresenceError::ClockRollback)?;
        let should_prewarm = snapshot.available && snapshot.present && age <= maximum_age_micros;
        drop(state);
        Ok(should_prewarm)
    }
}

/// Public capture activity exposed without any thinkpad-hpd-specific type.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CaptureActivity {
    /// No camera transaction is active.
    Idle,
    /// Authentication owns the calibrated camera pair.
    Authentication,
    /// Enrollment owns the camera pair and desktop lock/OSD conflict should be inhibited.
    Enrollment,
}

#[derive(Default)]
struct ActivityCounts {
    authentication: usize,
    enrollment: usize,
}

/// Generic reference-counted camera activity registry.
#[derive(Clone, Default)]
pub struct CaptureActivityRegistry {
    counts: Arc<Mutex<ActivityCounts>>,
}

impl CaptureActivityRegistry {
    /// Acquire a scoped activity lease. `Idle` cannot be acquired.
    ///
    /// # Errors
    ///
    /// Returns [`PresenceError`] for `Idle`, counter overflow, or poisoned state.
    pub fn acquire(
        &self,
        activity: CaptureActivity,
    ) -> Result<CaptureActivityLease, PresenceError> {
        if activity == CaptureActivity::Idle {
            return Err(PresenceError::InvalidActivity);
        }
        let mut counts = self.counts.lock().map_err(|_| PresenceError::StatePoisoned)?;
        let counter = match activity {
            CaptureActivity::Authentication => &mut counts.authentication,
            CaptureActivity::Enrollment => &mut counts.enrollment,
            CaptureActivity::Idle => return Err(PresenceError::InvalidActivity),
        };
        *counter = counter.checked_add(1).ok_or(PresenceError::ActivityOverflow)?;
        drop(counts);
        Ok(CaptureActivityLease { registry: self.clone(), activity })
    }

    /// Current generic state. Enrollment takes precedence over authentication for inhibition.
    ///
    /// # Errors
    ///
    /// Returns [`PresenceError::StatePoisoned`] when the registry mutex was poisoned.
    pub fn current(&self) -> Result<CaptureActivity, PresenceError> {
        let counts = self.counts.lock().map_err(|_| PresenceError::StatePoisoned)?;
        Ok(if counts.enrollment > 0 {
            CaptureActivity::Enrollment
        } else if counts.authentication > 0 {
            CaptureActivity::Authentication
        } else {
            CaptureActivity::Idle
        })
    }
}

/// RAII capture-activity lease; dropping it clears one registry reference.
pub struct CaptureActivityLease {
    registry: CaptureActivityRegistry,
    activity: CaptureActivity,
}

impl Drop for CaptureActivityLease {
    fn drop(&mut self) {
        let Ok(mut counts) = self.registry.counts.lock() else {
            return;
        };
        let counter = match self.activity {
            CaptureActivity::Authentication => &mut counts.authentication,
            CaptureActivity::Enrollment => &mut counts.enrollment,
            CaptureActivity::Idle => return,
        };
        *counter = counter.saturating_sub(1);
    }
}

/// Optional presence integration or generic activity state failure.
#[derive(Debug, Error)]
pub enum PresenceError {
    /// System D-Bus access failed.
    #[error("presence D-Bus operation failed: {0}")]
    Dbus(#[from] zbus::Error),
    /// Provider returned an out-of-contract diagnostic value.
    #[error("presence provider returned invalid state")]
    InvalidState,
    /// Observation did not advance monotonically.
    #[error("presence observation is stale")]
    StaleObservation,
    /// Freshness bound is zero or excessive.
    #[error("presence freshness bound is invalid")]
    InvalidFreshness,
    /// Caller monotonic time predates the cached observation.
    #[error("presence cache clock moved backwards")]
    ClockRollback,
    /// Shared optional state is unavailable after a panic.
    #[error("presence integration state is poisoned")]
    StatePoisoned,
    /// Idle is a derived state and cannot be leased.
    #[error("idle capture activity cannot be acquired")]
    InvalidActivity,
    /// Activity reference count overflowed.
    #[error("capture activity reference count overflowed")]
    ActivityOverflow,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presence_is_only_a_fresh_optional_prewarm_hint() -> Result<(), PresenceError> {
        let cache = PresenceCache::default();
        assert!(!cache.should_prewarm(10_000, 5_000)?);
        cache.update(PresenceSnapshot {
            available: true,
            present: true,
            raw_value: 2,
            observed_at_micros: 10_000,
        })?;
        assert!(cache.should_prewarm(14_000, 5_000)?);
        assert!(!cache.should_prewarm(16_000, 5_000)?);
        cache.clear()?;
        assert!(!cache.should_prewarm(17_000, 5_000)?);
        Ok(())
    }

    #[test]
    fn stale_presence_updates_and_clock_rollback_fail_closed() -> Result<(), PresenceError> {
        let cache = PresenceCache::default();
        let snapshot = PresenceSnapshot {
            available: true,
            present: false,
            raw_value: 0,
            observed_at_micros: 10_000,
        };
        cache.update(snapshot)?;
        assert!(matches!(cache.update(snapshot), Err(PresenceError::StaleObservation)));
        assert!(matches!(cache.should_prewarm(9_999, 5_000), Err(PresenceError::ClockRollback)));
        Ok(())
    }

    #[test]
    fn capture_activity_is_generic_scoped_and_enrollment_precedes_authentication()
    -> Result<(), PresenceError> {
        let registry = CaptureActivityRegistry::default();
        assert_eq!(registry.current()?, CaptureActivity::Idle);
        let authentication = registry.acquire(CaptureActivity::Authentication)?;
        assert_eq!(registry.current()?, CaptureActivity::Authentication);
        {
            let enrollment = registry.acquire(CaptureActivity::Enrollment)?;
            assert_eq!(registry.current()?, CaptureActivity::Enrollment);
            drop(enrollment);
        }
        assert_eq!(registry.current()?, CaptureActivity::Authentication);
        drop(authentication);
        assert_eq!(registry.current()?, CaptureActivity::Idle);
        Ok(())
    }
}
