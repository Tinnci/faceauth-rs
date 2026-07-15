//! Bounded one-shot authentication transaction lifecycle for the privileged daemon.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use faceauth_authz::AuthorizationGrant;
use faceauth_protocol::{DecisionCode, ProgressCode, Response, TransactionId};
use thiserror::Error;

/// Default authentication transaction duration.
pub const DEFAULT_TRANSACTION_DURATION_MICROS: u64 = 15_000_000;

/// Hard upper bound for one authentication transaction.
pub const MAX_TRANSACTION_DURATION_MICROS: u64 = 60_000_000;

/// Unserialized daemon-internal identity for one accepted socket connection.
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct ConnectionToken([u8; 16]);

impl ConnectionToken {
    /// Generate an unpredictable connection token from the operating-system random source.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError::Random`] when the operating-system random source is unavailable.
    pub fn generate() -> Result<Self, SessionError> {
        let mut token = [0_u8; 16];
        getrandom::fill(&mut token)?;
        Ok(Self(token))
    }
}

/// Daemon-internal cancellation signal for one admitted transaction.
///
/// The token is never serialized and can only be obtained after validating the exact connection
/// and transaction binding. It is safe to clone into a bounded capture/inference worker.
#[derive(Clone, Debug)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    /// Return whether the owning transaction was cancelled or reached a terminal state.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }

    fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
}

/// Resource and time bounds for authentication transactions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SessionConfig {
    /// Fixed lifetime from authorization to terminal result, in monotonic microseconds.
    pub transaction_duration_micros: u64,
}

impl SessionConfig {
    /// Validate transaction time bounds.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError::InvalidConfig`] for zero, sub-second, or excessive durations.
    pub const fn validate(self) -> Result<(), SessionError> {
        if self.transaction_duration_micros < 1_000_000
            || self.transaction_duration_micros > MAX_TRANSACTION_DURATION_MICROS
        {
            Err(SessionError::InvalidConfig)
        } else {
            Ok(())
        }
    }
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self { transaction_duration_micros: DEFAULT_TRANSACTION_DURATION_MICROS }
    }
}

struct ActiveSession {
    grant: AuthorizationGrant,
    connection: ConnectionToken,
    deadline_micros: u64,
    cancellation: CancellationToken,
}

/// Single-capacity authentication transaction manager.
///
/// Capacity remains occupied until a terminal result is emitted. Expired sessions must be reaped
/// with [`Self::expire`] rather than silently replaced by a new request.
pub struct SessionManager {
    config: SessionConfig,
    active: Option<ActiveSession>,
}

impl SessionManager {
    /// Construct an empty transaction manager.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError::InvalidConfig`] when transaction duration is outside hard bounds.
    pub fn new(config: SessionConfig) -> Result<Self, SessionError> {
        config.validate()?;
        Ok(Self { config, active: None })
    }

    /// Start one already-authorized transaction.
    ///
    /// `now_micros` must come from the daemon monotonic clock.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError::Busy`] while any transaction remains active, including an expired
    /// transaction whose timeout response has not yet been emitted.
    pub fn start(
        &mut self,
        grant: AuthorizationGrant,
        connection: ConnectionToken,
        now_micros: u64,
    ) -> Result<Response, SessionError> {
        if self.active.is_some() {
            return Err(SessionError::Busy);
        }
        let deadline_micros = now_micros
            .checked_add(self.config.transaction_duration_micros)
            .ok_or(SessionError::DeadlineOverflow)?;
        let transaction_id = grant.transaction_id;
        self.active = Some(ActiveSession {
            grant,
            connection,
            deadline_micros,
            cancellation: CancellationToken::new(),
        });
        Ok(Response::Started { transaction_id })
    }

    /// Emit non-terminal progress for the exact active connection and transaction.
    ///
    /// If the deadline has elapsed, this consumes the transaction and emits `TimedOut` instead.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError`] when there is no active transaction or binding does not match.
    pub fn progress(
        &mut self,
        connection: ConnectionToken,
        transaction_id: TransactionId,
        progress: ProgressCode,
        now_micros: u64,
    ) -> Result<Response, SessionError> {
        self.verify_binding(connection, transaction_id)?;
        if self.deadline_reached(now_micros)? {
            return self.take_terminal(DecisionCode::TimedOut);
        }
        Ok(Response::Progress { transaction_id, progress })
    }

    /// Complete the exact active transaction with an internal biometric or system decision.
    ///
    /// `Cancelled` and `TimedOut` are reserved for [`Self::cancel`] and [`Self::expire`]. If the
    /// deadline has elapsed, `TimedOut` replaces the supplied result.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError`] for missing/mismatched binding or a reserved decision.
    pub fn complete(
        &mut self,
        connection: ConnectionToken,
        transaction_id: TransactionId,
        decision: DecisionCode,
        now_micros: u64,
    ) -> Result<Response, SessionError> {
        if matches!(decision, DecisionCode::Cancelled | DecisionCode::TimedOut) {
            return Err(SessionError::ReservedDecision);
        }
        self.verify_binding(connection, transaction_id)?;
        if self.deadline_reached(now_micros)? {
            return self.take_terminal(DecisionCode::TimedOut);
        }
        self.take_terminal(decision)
    }

    /// Cancel the exact transaction from the connection that started it.
    ///
    /// If the deadline has elapsed, this emits `TimedOut` rather than `Cancelled`.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError`] for missing or mismatched binding.
    pub fn cancel(
        &mut self,
        connection: ConnectionToken,
        transaction_id: TransactionId,
        now_micros: u64,
    ) -> Result<Response, SessionError> {
        self.verify_binding(connection, transaction_id)?;
        if let Some(active) = self.active.as_ref() {
            active.cancellation.cancel();
        }
        if self.deadline_reached(now_micros)? {
            return self.take_terminal(DecisionCode::TimedOut);
        }
        self.take_terminal(DecisionCode::Cancelled)
    }

    /// Emit a timeout terminal response when the active deadline has elapsed.
    #[must_use]
    pub fn expire(&mut self, now_micros: u64) -> Option<Response> {
        let expired =
            self.active.as_ref().is_some_and(|active| now_micros >= active.deadline_micros);
        if expired { self.take_terminal(DecisionCode::TimedOut).ok() } else { None }
    }

    /// Return whether transaction capacity is currently occupied.
    #[must_use]
    pub const fn is_busy(&self) -> bool {
        self.active.is_some()
    }

    /// Return the authorized target UID for the exact active connection and transaction.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError`] when no transaction is active or either binding differs.
    pub fn target_uid(
        &self,
        connection: ConnectionToken,
        transaction_id: TransactionId,
    ) -> Result<u32, SessionError> {
        self.verify_binding(connection, transaction_id)?;
        self.active
            .as_ref()
            .map(|active| active.grant.target_uid)
            .ok_or(SessionError::NoActiveTransaction)
    }

    /// Return a cancellation signal for the exact active connection and transaction.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError`] when no transaction is active or either binding differs.
    pub fn cancellation_token(
        &self,
        connection: ConnectionToken,
        transaction_id: TransactionId,
    ) -> Result<CancellationToken, SessionError> {
        self.verify_binding(connection, transaction_id)?;
        self.active
            .as_ref()
            .map(|active| active.cancellation.clone())
            .ok_or(SessionError::NoActiveTransaction)
    }

    fn verify_binding(
        &self,
        connection: ConnectionToken,
        transaction_id: TransactionId,
    ) -> Result<(), SessionError> {
        let active = self.active.as_ref().ok_or(SessionError::NoActiveTransaction)?;
        if active.connection != connection {
            return Err(SessionError::WrongConnection);
        }
        if active.grant.transaction_id != transaction_id {
            return Err(SessionError::WrongTransaction);
        }
        Ok(())
    }

    fn deadline_reached(&self, now_micros: u64) -> Result<bool, SessionError> {
        self.active
            .as_ref()
            .map(|active| now_micros >= active.deadline_micros)
            .ok_or(SessionError::NoActiveTransaction)
    }

    fn take_terminal(&mut self, decision: DecisionCode) -> Result<Response, SessionError> {
        let active = self.active.take().ok_or(SessionError::NoActiveTransaction)?;
        active.cancellation.cancel();
        Ok(Response::Completed { transaction_id: active.grant.transaction_id, decision })
    }
}

/// Authentication session lifecycle failure.
#[derive(Debug, Error)]
pub enum SessionError {
    /// Transaction time bounds are invalid.
    #[error("invalid authentication session configuration")]
    InvalidConfig,
    /// Operating-system randomness was unavailable.
    #[error("unable to generate connection token: {0}")]
    Random(#[from] getrandom::Error),
    /// Monotonic deadline overflowed.
    #[error("authentication transaction deadline overflowed")]
    DeadlineOverflow,
    /// The single transaction slot is occupied.
    #[error("authentication service is busy")]
    Busy,
    /// No transaction is active.
    #[error("no authentication transaction is active")]
    NoActiveTransaction,
    /// A different socket connection attempted to operate on the transaction.
    #[error("authentication transaction belongs to another connection")]
    WrongConnection,
    /// Transaction identifier did not match the active request.
    #[error("authentication transaction identifier does not match")]
    WrongTransaction,
    /// Caller attempted to synthesize a manager-owned terminal decision.
    #[error("cancelled and timed-out decisions are reserved for the session manager")]
    ReservedDecision,
}

#[cfg(test)]
mod tests {
    use faceauth_authz::ExecutableFingerprint;
    use faceauth_protocol::{AuthenticationPurpose, ServiceName};
    use faceauth_transport::PeerIdentity;

    use super::*;

    fn grant() -> Result<AuthorizationGrant, faceauth_protocol::ServiceNameError> {
        Ok(AuthorizationGrant {
            transaction_id: TransactionId::generate(),
            peer: PeerIdentity { pid: 123, uid: 1000, gid: 1000 },
            target_uid: 1000,
            service: ServiceName::parse("faceauth-test")?,
            purpose: AuthenticationPurpose::Test,
            executable: ExecutableFingerprint { device: 8, inode: 42 },
        })
    }

    #[test]
    fn start_progress_and_completion_are_one_shot() -> Result<(), Box<dyn std::error::Error>> {
        let mut manager = SessionManager::new(SessionConfig::default())?;
        let connection = ConnectionToken::generate()?;
        let grant = grant()?;
        let transaction_id = grant.transaction_id;

        assert_eq!(
            manager.start(grant, connection, 1_000_000)?,
            Response::Started { transaction_id }
        );
        let cancellation = manager.cancellation_token(connection, transaction_id)?;
        assert_eq!(
            manager.progress(connection, transaction_id, ProgressCode::HoldStill, 2_000_000)?,
            Response::Progress { transaction_id, progress: ProgressCode::HoldStill }
        );
        assert_eq!(
            manager.complete(connection, transaction_id, DecisionCode::Accepted, 3_000_000)?,
            Response::Completed { transaction_id, decision: DecisionCode::Accepted }
        );
        assert!(cancellation.is_cancelled());
        assert!(matches!(
            manager.complete(connection, transaction_id, DecisionCode::Accepted, 3_000_001),
            Err(SessionError::NoActiveTransaction)
        ));
        Ok(())
    }

    #[test]
    fn target_uid_requires_exact_connection_and_transaction()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut manager = SessionManager::new(SessionConfig::default())?;
        let connection = ConnectionToken::generate()?;
        let other_connection = ConnectionToken::generate()?;
        let grant = grant()?;
        let transaction_id = grant.transaction_id;
        let _ = manager.start(grant, connection, 1_000_000)?;
        assert_eq!(manager.target_uid(connection, transaction_id)?, 1000);
        assert!(matches!(
            manager.target_uid(other_connection, transaction_id),
            Err(SessionError::WrongConnection)
        ));
        assert!(matches!(
            manager.target_uid(connection, TransactionId::generate()),
            Err(SessionError::WrongTransaction)
        ));
        Ok(())
    }

    #[test]
    fn cancellation_token_is_bound_and_signaled_before_terminal_cancel()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut manager = SessionManager::new(SessionConfig::default())?;
        let connection = ConnectionToken::generate()?;
        let other_connection = ConnectionToken::generate()?;
        let grant = grant()?;
        let transaction_id = grant.transaction_id;
        let _ = manager.start(grant, connection, 1_000_000)?;
        let token = manager.cancellation_token(connection, transaction_id)?;
        assert!(!token.is_cancelled());
        assert!(matches!(
            manager.cancellation_token(other_connection, transaction_id),
            Err(SessionError::WrongConnection)
        ));
        assert_eq!(
            manager.cancel(connection, transaction_id, 2_000_000)?,
            Response::Completed { transaction_id, decision: DecisionCode::Cancelled }
        );
        assert!(token.is_cancelled());
        assert!(matches!(
            manager.cancellation_token(connection, transaction_id),
            Err(SessionError::NoActiveTransaction)
        ));
        Ok(())
    }

    #[test]
    fn busy_slot_cannot_be_replaced_even_after_deadline() -> Result<(), Box<dyn std::error::Error>>
    {
        let config = SessionConfig { transaction_duration_micros: 1_000_000 };
        let mut manager = SessionManager::new(config)?;
        let connection = ConnectionToken::generate()?;
        let first = grant()?;
        let first_id = first.transaction_id;
        let _ = manager.start(first, connection, 1_000_000)?;

        assert!(matches!(
            manager.start(grant()?, ConnectionToken::generate()?, 2_000_000),
            Err(SessionError::Busy)
        ));
        assert_eq!(
            manager.expire(2_000_000),
            Some(Response::Completed {
                transaction_id: first_id,
                decision: DecisionCode::TimedOut
            })
        );
        assert!(!manager.is_busy());
        Ok(())
    }

    #[test]
    fn wrong_connection_and_transaction_cannot_cancel() -> Result<(), Box<dyn std::error::Error>> {
        let mut manager = SessionManager::new(SessionConfig::default())?;
        let owner = ConnectionToken::generate()?;
        let attacker = ConnectionToken::generate()?;
        let grant = grant()?;
        let transaction_id = grant.transaction_id;
        let _ = manager.start(grant, owner, 1_000_000)?;

        assert!(matches!(
            manager.cancel(attacker, transaction_id, 2_000_000),
            Err(SessionError::WrongConnection)
        ));
        assert!(matches!(
            manager.cancel(owner, TransactionId::generate(), 2_000_000),
            Err(SessionError::WrongTransaction)
        ));
        assert!(manager.is_busy());
        Ok(())
    }

    #[test]
    fn deadline_overrides_progress_completion_and_cancellation()
    -> Result<(), Box<dyn std::error::Error>> {
        for operation in 0..3 {
            let config = SessionConfig { transaction_duration_micros: 1_000_000 };
            let mut manager = SessionManager::new(config)?;
            let connection = ConnectionToken::generate()?;
            let grant = grant()?;
            let transaction_id = grant.transaction_id;
            let _ = manager.start(grant, connection, 1_000_000)?;

            let response = match operation {
                0 => manager.progress(
                    connection,
                    transaction_id,
                    ProgressCode::Processing,
                    2_000_000,
                )?,
                1 => manager.complete(
                    connection,
                    transaction_id,
                    DecisionCode::Accepted,
                    2_000_000,
                )?,
                _ => manager.cancel(connection, transaction_id, 2_000_000)?,
            };
            assert_eq!(
                response,
                Response::Completed { transaction_id, decision: DecisionCode::TimedOut }
            );
        }
        Ok(())
    }

    #[test]
    fn reserved_terminal_decisions_cannot_be_injected() -> Result<(), Box<dyn std::error::Error>> {
        let mut manager = SessionManager::new(SessionConfig::default())?;
        let connection = ConnectionToken::generate()?;
        let grant = grant()?;
        let transaction_id = grant.transaction_id;
        let _ = manager.start(grant, connection, 1_000_000)?;

        assert!(matches!(
            manager.complete(connection, transaction_id, DecisionCode::Cancelled, 2_000_000),
            Err(SessionError::ReservedDecision)
        ));
        assert!(manager.is_busy());
        Ok(())
    }
}
