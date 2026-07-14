//! Versioned messages shared by unprivileged clients and the privileged daemon.

use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

/// Current protocol version.
pub const PROTOCOL_VERSION: u16 = 2;

/// Maximum encoded PAM or desktop service-name length.
pub const MAX_SERVICE_NAME_LENGTH: usize = 64;

/// Versioned transport envelope.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Envelope<T> {
    /// Protocol version used to encode the message.
    pub protocol_version: u16,
    /// Request or response payload.
    pub message: T,
}

impl<T> Envelope<T> {
    /// Wrap a message in the current protocol version.
    pub const fn current(message: T) -> Self {
        Self { protocol_version: PROTOCOL_VERSION, message }
    }

    /// Validate that the peer uses the current protocol version.
    ///
    /// # Errors
    ///
    /// Returns [`ProtocolError::UnsupportedVersion`] when the envelope version differs from
    /// [`PROTOCOL_VERSION`].
    pub const fn validate_version(&self) -> Result<(), ProtocolError> {
        if self.protocol_version == PROTOCOL_VERSION {
            Ok(())
        } else {
            Err(ProtocolError::UnsupportedVersion {
                received: self.protocol_version,
                supported: PROTOCOL_VERSION,
            })
        }
    }
}

/// Unpredictable identifier binding a request to all of its responses.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub struct TransactionId(Uuid);

impl TransactionId {
    /// Generate a transaction identifier using the operating system random source.
    #[must_use]
    pub fn generate() -> Self {
        Self(Uuid::new_v4())
    }
}

impl fmt::Display for TransactionId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

/// Validated PAM or desktop service name used for policy and audit context.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(try_from = "String", into = "String")]
pub struct ServiceName(String);

impl ServiceName {
    /// Parse and validate a service name.
    ///
    /// Allowed names contain 1 to 64 ASCII alphanumeric, `.`, `_`, or `-` characters.
    ///
    /// # Errors
    ///
    /// Returns [`ServiceNameError`] when the input is empty, too long, or contains unsupported
    /// characters.
    pub fn parse(value: impl Into<String>) -> Result<Self, ServiceNameError> {
        let value = value.into();
        if value.is_empty() {
            return Err(ServiceNameError::Empty);
        }
        if value.len() > MAX_SERVICE_NAME_LENGTH {
            return Err(ServiceNameError::TooLong {
                actual: value.len(),
                maximum: MAX_SERVICE_NAME_LENGTH,
            });
        }
        if !value.bytes().all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte)) {
            return Err(ServiceNameError::InvalidCharacter);
        }
        Ok(Self(value))
    }

    /// Borrow the validated service name.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for ServiceName {
    type Error = ServiceNameError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::parse(value)
    }
}

impl From<ServiceName> for String {
    fn from(value: ServiceName) -> Self {
        value.0
    }
}

impl fmt::Display for ServiceName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

/// Invalid service-name input.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ServiceNameError {
    /// The service name was empty.
    #[error("service name cannot be empty")]
    Empty,
    /// The service name exceeded the protocol limit.
    #[error("service name length {actual} exceeds {maximum}")]
    TooLong {
        /// Actual encoded byte length.
        actual: usize,
        /// Maximum encoded byte length.
        maximum: usize,
    },
    /// The service name contained a character outside the allowlist.
    #[error("service name contains an unsupported character")]
    InvalidCharacter,
}

/// User-visible operation requesting biometric authentication.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AuthenticationPurpose {
    /// Dedicated development-only PAM test service.
    Test,
    /// Desktop screen unlock.
    ScreenUnlock,
    /// Graphical or console login.
    Login,
    /// Polkit authorization prompt.
    Polkit,
    /// Privilege elevation through sudo. This remains opt-in.
    Sudo,
}

/// Authentication request context supplied by a trusted local caller.
///
/// The daemon must derive peer UID, PID, and executable identity from Unix peer credentials. It
/// must not treat this serialized context as proof of caller identity.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RequestContext {
    /// Unique transaction identifier used to prevent response reuse.
    pub transaction_id: TransactionId,
    /// Numeric account identifier whose enrolled template is requested.
    pub target_uid: u32,
    /// Validated PAM or desktop service name.
    pub service: ServiceName,
    /// Policy category for this authentication attempt.
    pub purpose: AuthenticationPurpose,
}

/// Request sent over the authenticated local transport.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum Request {
    /// Start a bounded authentication transaction.
    Authenticate {
        /// Identity and caller context bound to the transaction.
        context: RequestContext,
    },
    /// Cancel an in-flight transaction.
    Cancel {
        /// Identifier of the transaction to cancel.
        transaction_id: TransactionId,
    },
}

/// Stable result of biometric policy evaluation.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum DecisionCode {
    /// All required evidence passed.
    Accepted,
    /// Required IR evidence was absent.
    InfraredMissing,
    /// Required visible-light evidence was absent.
    VisibleMissing,
    /// Capture quality was below policy.
    InsufficientQuality,
    /// Passive presentation-attack detection failed.
    PassiveLivenessFailed,
    /// The randomized active challenge failed.
    ActiveChallengeFailed,
    /// The face embedding did not match the enrolled template.
    FaceMismatch,
    /// The caller cancelled the transaction.
    Cancelled,
    /// The bounded transaction deadline elapsed.
    TimedOut,
    /// A non-biometric internal failure prevented a decision.
    InternalError,
}

impl DecisionCode {
    /// Return whether this code authorizes the requested operation.
    #[must_use]
    pub const fn is_accepted(self) -> bool {
        matches!(self, Self::Accepted)
    }
}

/// Stable reason for rejecting a request before biometric processing.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RejectionCode {
    /// The caller is not authorized by peer credentials and service policy.
    UnauthorizedPeer,
    /// The requested service or purpose is disabled by administrator policy.
    ServiceNotAllowed,
    /// The target account has no usable enrolled template.
    NotEnrolled,
    /// A required camera is unavailable or does not match configuration.
    CameraUnavailable,
    /// The daemon cannot accept another bounded transaction.
    Busy,
    /// The service is not production-ready or is shutting down.
    ServiceUnavailable,
}

/// Localized clients map these closed progress codes to user-visible text or icons.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProgressCode {
    /// Move one face into the calibrated capture region.
    PositionFace,
    /// Hold a centered, eyes-open pose while baseline evidence is collected.
    HoldStill,
    /// Close and reopen both eyes.
    Blink,
    /// Turn toward the user's left.
    TurnLeft,
    /// Turn toward the user's right.
    TurnRight,
    /// Return to a centered, eyes-open pose.
    ReturnToCenter,
    /// Required evidence has been captured and bounded inference is running.
    Processing,
}

/// Response returned by the daemon.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum Response {
    /// Request was accepted for processing.
    Started {
        /// Identifier of the accepted transaction.
        transaction_id: TransactionId,
    },
    /// Non-terminal UI progress that carries no frame, landmark, score, or template data.
    Progress {
        /// Identifier of the active transaction.
        transaction_id: TransactionId,
        /// Stable prompt code for PAM conversations or a lock-screen OSD.
        progress: ProgressCode,
    },
    /// Authentication completed.
    Completed {
        /// Identifier of the completed transaction.
        transaction_id: TransactionId,
        /// Stable machine-readable policy decision.
        decision: DecisionCode,
    },
    /// Request was rejected before biometric work began.
    Rejected {
        /// Transaction identifier, when the request could be decoded far enough to recover it.
        transaction_id: Option<TransactionId>,
        /// Stable machine-readable rejection reason.
        reason: RejectionCode,
    },
}

/// Invalid protocol envelope.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum ProtocolError {
    /// The peer sent a protocol version this build does not support.
    #[error("unsupported protocol version {received}; this build supports {supported}")]
    UnsupportedVersion {
        /// Version in the received envelope.
        received: u16,
        /// Version supported by this build.
        supported: u16,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn service_names_are_bounded_and_allowlisted() {
        assert_eq!(ServiceName::parse("kde"), Ok(ServiceName("kde".to_owned())));
        assert_eq!(
            ServiceName::parse("faceauth_test-1"),
            Ok(ServiceName("faceauth_test-1".to_owned()))
        );
        assert_eq!(ServiceName::parse("../kde"), Err(ServiceNameError::InvalidCharacter));
        assert_eq!(ServiceName::parse(""), Err(ServiceNameError::Empty));
    }

    #[test]
    fn request_round_trip_preserves_transaction_binding() -> Result<(), Box<dyn std::error::Error>>
    {
        let transaction_id = TransactionId::generate();
        let envelope = Envelope::current(Request::Authenticate {
            context: RequestContext {
                transaction_id,
                target_uid: 1000,
                service: ServiceName::parse("faceauth-test")?,
                purpose: AuthenticationPurpose::Test,
            },
        });
        let encoded = serde_json::to_vec(&envelope)?;
        let decoded: Envelope<Request> = serde_json::from_slice(&encoded)?;

        assert_eq!(decoded, envelope);
        assert_eq!(decoded.validate_version(), Ok(()));
        Ok(())
    }

    #[test]
    fn unsupported_versions_fail_closed() {
        let envelope = Envelope {
            protocol_version: PROTOCOL_VERSION + 1,
            message: Request::Cancel { transaction_id: TransactionId::generate() },
        };

        assert_eq!(
            envelope.validate_version(),
            Err(ProtocolError::UnsupportedVersion {
                received: PROTOCOL_VERSION + 1,
                supported: PROTOCOL_VERSION,
            })
        );
    }

    #[test]
    fn only_the_accepted_decision_authorizes() {
        assert!(DecisionCode::Accepted.is_accepted());
        assert!(!DecisionCode::InternalError.is_accepted());
        assert!(!DecisionCode::FaceMismatch.is_accepted());
    }

    #[test]
    fn progress_is_transaction_bound_and_contains_no_measurements()
    -> Result<(), Box<dyn std::error::Error>> {
        let transaction_id = TransactionId::generate();
        let envelope =
            Envelope::current(Response::Progress { transaction_id, progress: ProgressCode::Blink });
        let encoded = serde_json::to_string(&envelope)?;
        let decoded: Envelope<Response> = serde_json::from_str(&encoded)?;

        assert_eq!(decoded, envelope);
        assert!(!encoded.contains("eye"));
        assert!(!encoded.contains("yaw"));
        assert!(!encoded.contains("score"));
        Ok(())
    }
}
