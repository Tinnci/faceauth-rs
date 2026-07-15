//! Privileged daemon authentication boundary orchestration.

use faceauth_authz::{
    AuthorizationError, AuthorizationPolicy, ExecutableError, VerifiedExecutable,
};
use faceauth_protocol::{RejectionCode, Request, Response};
use faceauth_session::{ConnectionToken, SessionError, SessionManager};
use faceauth_transport::{PeerStream, TransportError};
use thiserror::Error;

/// Decodes and admits requests through executable verification, authorization, and session state.
pub struct BoundaryService {
    authorization: AuthorizationPolicy,
    sessions: SessionManager,
}

impl BoundaryService {
    /// Construct a boundary service from validated authorization and session policies.
    #[must_use]
    pub const fn new(authorization: AuthorizationPolicy, sessions: SessionManager) -> Self {
        Self { authorization, sessions }
    }

    /// Read and process one request from an already peer-credentialed connection.
    ///
    /// Successful authentication admission emits `Started`; no biometric success is produced here.
    /// Cancellation is accepted only for the exact connection-bound active transaction.
    /// Rejections are written as closed protocol responses.
    ///
    /// # Errors
    ///
    /// Returns [`BoundaryError`] only when framing or response I/O fails. Authorization and session
    /// failures are converted to fail-closed protocol rejections.
    pub fn handle_one(
        &mut self,
        stream: &mut PeerStream,
        connection: ConnectionToken,
        now_micros: u64,
    ) -> Result<Response, BoundaryError> {
        let request = stream.read_message::<Request>()?.message;
        let response = match request {
            Request::Authenticate { context } => {
                let transaction_id = context.transaction_id;
                let executable = VerifiedExecutable::from_peer(stream.peer(), stream.peer_pidfd());
                match executable.map_err(ExecutableOrAuthorizationError::from).and_then(
                    |executable| {
                        self.authorization
                            .authorize(stream.peer(), Some(executable), &context)
                            .map_err(ExecutableOrAuthorizationError::Authorization)
                    },
                ) {
                    Ok(grant) => match self.sessions.start(grant, connection, now_micros) {
                        Ok(response) => response,
                        Err(SessionError::Busy) => Response::Rejected {
                            transaction_id: Some(transaction_id),
                            reason: RejectionCode::Busy,
                        },
                        Err(_) => Response::Rejected {
                            transaction_id: Some(transaction_id),
                            reason: RejectionCode::ServiceUnavailable,
                        },
                    },
                    Err(ExecutableOrAuthorizationError::Authorization(
                        AuthorizationError::ServiceNotAllowed,
                    )) => Response::Rejected {
                        transaction_id: Some(transaction_id),
                        reason: RejectionCode::ServiceNotAllowed,
                    },
                    Err(_) => Response::Rejected {
                        transaction_id: Some(transaction_id),
                        reason: RejectionCode::UnauthorizedPeer,
                    },
                }
            }
            Request::Cancel { transaction_id } => {
                match self.sessions.cancel(connection, transaction_id, now_micros) {
                    Ok(response) => response,
                    Err(SessionError::WrongConnection | SessionError::WrongTransaction) => {
                        Response::Rejected {
                            transaction_id: Some(transaction_id),
                            reason: RejectionCode::UnauthorizedPeer,
                        }
                    }
                    Err(_) => Response::Rejected {
                        transaction_id: Some(transaction_id),
                        reason: RejectionCode::ServiceUnavailable,
                    },
                }
            }
        };
        stream.write_message(response.clone())?;
        Ok(response)
    }

    /// Borrow the transaction manager for the internal capture/inference pipeline.
    #[must_use]
    pub const fn sessions(&mut self) -> &mut SessionManager {
        &mut self.sessions
    }
}

enum ExecutableOrAuthorizationError {
    Executable,
    Authorization(AuthorizationError),
}

impl From<ExecutableError> for ExecutableOrAuthorizationError {
    fn from(_error: ExecutableError) -> Self {
        Self::Executable
    }
}

/// Boundary framing or response delivery failure.
#[derive(Debug, Error)]
pub enum BoundaryError {
    /// Local protocol transport failed.
    #[error("daemon authentication boundary transport failed: {0}")]
    Transport(#[from] TransportError),
}

#[cfg(test)]
mod tests {
    use std::os::unix::net::UnixStream;

    use faceauth_authz::{
        AuthorizationGrant, AuthorizationRule, CallerRelation, ExecutableFingerprint,
    };
    use faceauth_protocol::{
        AuthenticationPurpose, Envelope, RequestContext, ServiceName, TransactionId,
    };
    use faceauth_session::SessionConfig;
    use faceauth_transport::TransportConfig;

    use super::*;

    #[test]
    fn untrusted_development_executable_fails_before_session_start()
    -> Result<(), Box<dyn std::error::Error>> {
        let rule = AuthorizationRule {
            service: ServiceName::parse("faceauth-test")?,
            purpose: AuthenticationPurpose::Test,
            caller: CallerRelation::RootOrTargetUser,
            executables: vec![ExecutableFingerprint { device: 1, inode: 1 }],
        };
        let mut service = BoundaryService::new(
            AuthorizationPolicy::new(vec![rule])?,
            SessionManager::new(SessionConfig::default())?,
        );
        let connection = ConnectionToken::generate()?;
        let (client, server) = UnixStream::pair()?;
        let mut client = PeerStream::connect(client, TransportConfig::default())?;
        let mut server = PeerStream::connect(server, TransportConfig::default())?;
        let transaction_id = TransactionId::generate();
        client.write_message(Request::Authenticate {
            context: RequestContext {
                transaction_id,
                target_uid: server.peer().uid,
                service: ServiceName::parse("faceauth-test")?,
                purpose: AuthenticationPurpose::Test,
            },
        })?;

        assert_eq!(
            service.handle_one(&mut server, connection, 1_000_000)?,
            Response::Rejected {
                transaction_id: Some(transaction_id),
                reason: RejectionCode::UnauthorizedPeer,
            }
        );
        let wire: Envelope<Response> = client.read_message()?;
        assert_eq!(
            wire.message,
            Response::Rejected {
                transaction_id: Some(transaction_id),
                reason: RejectionCode::UnauthorizedPeer,
            }
        );
        assert!(!service.sessions().is_busy());
        Ok(())
    }

    #[test]
    fn cancellation_without_an_active_session_never_succeeds()
    -> Result<(), Box<dyn std::error::Error>> {
        let rule = AuthorizationRule {
            service: ServiceName::parse("faceauth-test")?,
            purpose: AuthenticationPurpose::Test,
            caller: CallerRelation::RootOnly,
            executables: vec![ExecutableFingerprint { device: 1, inode: 1 }],
        };
        let mut service = BoundaryService::new(
            AuthorizationPolicy::new(vec![rule])?,
            SessionManager::new(SessionConfig::default())?,
        );
        let connection = ConnectionToken::generate()?;
        let (client, server) = UnixStream::pair()?;
        let mut client = PeerStream::connect(client, TransportConfig::default())?;
        let mut server = PeerStream::connect(server, TransportConfig::default())?;
        let transaction_id = TransactionId::generate();
        client.write_message(Request::Cancel { transaction_id })?;

        assert_eq!(
            service.handle_one(&mut server, connection, 1_000_000)?,
            Response::Rejected {
                transaction_id: Some(transaction_id),
                reason: RejectionCode::ServiceUnavailable,
            }
        );
        Ok(())
    }

    #[test]
    fn exact_connection_can_cancel_an_internally_started_session()
    -> Result<(), Box<dyn std::error::Error>> {
        let service_name = ServiceName::parse("faceauth-test")?;
        let rule = AuthorizationRule {
            service: service_name.clone(),
            purpose: AuthenticationPurpose::Test,
            caller: CallerRelation::RootOnly,
            executables: vec![ExecutableFingerprint { device: 1, inode: 1 }],
        };
        let mut service = BoundaryService::new(
            AuthorizationPolicy::new(vec![rule])?,
            SessionManager::new(SessionConfig::default())?,
        );
        let connection = ConnectionToken::generate()?;
        let transaction_id = TransactionId::generate();
        let grant = AuthorizationGrant {
            transaction_id,
            peer: faceauth_transport::PeerIdentity { pid: 123, uid: 0, gid: 0 },
            target_uid: 1000,
            service: service_name,
            purpose: AuthenticationPurpose::Test,
            executable: ExecutableFingerprint { device: 1, inode: 1 },
        };
        let _ = service.sessions().start(grant, connection, 1_000_000)?;
        let (client, server) = UnixStream::pair()?;
        let mut client = PeerStream::connect(client, TransportConfig::default())?;
        let mut server = PeerStream::connect(server, TransportConfig::default())?;
        client.write_message(Request::Cancel { transaction_id })?;

        assert_eq!(
            service.handle_one(&mut server, connection, 2_000_000)?,
            Response::Completed {
                transaction_id,
                decision: faceauth_protocol::DecisionCode::Cancelled
            }
        );
        let wire: Envelope<Response> = client.read_message()?;
        assert_eq!(
            wire.message,
            Response::Completed {
                transaction_id,
                decision: faceauth_protocol::DecisionCode::Cancelled
            }
        );
        assert!(!service.sessions().is_busy());
        Ok(())
    }
}
