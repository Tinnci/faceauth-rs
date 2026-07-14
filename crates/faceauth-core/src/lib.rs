//! Desktop-independent policy and decision types for face authentication.

mod policy;

pub use policy::{
    AuthPolicy, AuthenticationDecision, AuthenticationEvidence, CaptureModality, CapturePair,
    CapturePolicy, CheckStatus, DecisionReason, LivenessLevel, ObservationStatus, PolicyError,
};
