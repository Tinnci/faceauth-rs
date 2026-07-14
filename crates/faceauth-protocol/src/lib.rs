//! Versioned messages shared by unprivileged clients and the privileged daemon.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Current protocol version.
pub const PROTOCOL_VERSION: u16 = 1;

/// Authentication request context supplied by a trusted local caller.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RequestContext {
    /// Unique request identifier used to prevent response reuse.
    pub request_id: Uuid,
    /// Local numeric user identifier being authenticated.
    pub uid: u32,
    /// PAM or desktop service name, such as `kde` or a dedicated test service.
    pub service: String,
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
        request_id: Uuid,
    },
}

/// Response returned by the daemon.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum Response {
    /// Request was accepted for processing.
    Started {
        /// Identifier of the accepted transaction.
        request_id: Uuid,
    },
    /// Authentication completed.
    Completed {
        /// Identifier of the completed transaction.
        request_id: Uuid,
        /// Whether policy accepted the biometric evidence.
        accepted: bool,
        /// Stable machine-readable completion reason.
        reason: String,
    },
    /// Request was rejected before biometric work began.
    Rejected {
        /// Transaction identifier, when the request could be decoded far enough to recover it.
        request_id: Option<Uuid>,
        /// Stable machine-readable rejection reason.
        reason: String,
    },
}
