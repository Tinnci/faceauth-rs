//! Fail-closed daemon authorization for local biometric authentication requests.

use std::{
    fs::File,
    io,
    os::{
        fd::{AsRawFd, BorrowedFd},
        unix::fs::MetadataExt,
    },
    path::PathBuf,
};

use faceauth_protocol::{AuthenticationPurpose, RequestContext, ServiceName, TransactionId};
use faceauth_transport::PeerIdentity;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Maximum number of exact service/purpose rules accepted from administrator configuration.
pub const MAX_AUTHORIZATION_RULES: usize = 64;

/// Maximum executable fingerprints accepted for one exact rule.
pub const MAX_EXECUTABLES_PER_RULE: usize = 32;

/// Stable filesystem identity of an installed caller executable.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutableFingerprint {
    /// Filesystem device number.
    pub device: u64,
    /// Filesystem inode number.
    pub inode: u64,
}

/// Executable evidence derived from an already-open file descriptor.
///
/// Construction requires a root-owned regular file that is not writable by group or others.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VerifiedExecutable {
    fingerprint: ExecutableFingerprint,
}

impl VerifiedExecutable {
    /// Resolve and verify the executable of a pidfd-pinned connected peer.
    ///
    /// # Errors
    ///
    /// Returns [`ExecutableError`] when the pidfd does not match the `SO_PEERCRED` PID, the process
    /// exits during lookup, metadata cannot be read, or the executable is not a root-owned,
    /// non-group-writable, non-world-writable regular executable file.
    pub fn from_peer(
        peer: PeerIdentity,
        peer_pidfd: BorrowedFd<'_>,
    ) -> Result<Self, ExecutableError> {
        verify_pidfd(peer.pid, peer_pidfd)?;
        let file = File::open(PathBuf::from(format!("/proc/{}/exe", peer.pid)))?;
        verify_pidfd(peer.pid, peer_pidfd)?;
        Self::from_verified_file(&file)
    }

    fn from_verified_file(file: &File) -> Result<Self, ExecutableError> {
        let metadata = file.metadata()?;
        if !metadata.file_type().is_file() {
            return Err(ExecutableError::NotRegularFile);
        }
        if metadata.uid() != 0 {
            return Err(ExecutableError::NotRootOwned { actual_uid: metadata.uid() });
        }
        let mode = metadata.mode() & 0o777;
        if mode & 0o022 != 0 {
            return Err(ExecutableError::WritableByUntrustedUser { mode });
        }
        if mode & 0o111 == 0 {
            return Err(ExecutableError::NotExecutable { mode });
        }
        Ok(Self {
            fingerprint: ExecutableFingerprint { device: metadata.dev(), inode: metadata.ino() },
        })
    }

    /// Return the verified stable filesystem identity.
    #[must_use]
    pub const fn fingerprint(self) -> ExecutableFingerprint {
        self.fingerprint
    }
}

fn verify_pidfd(expected_pid: u32, pidfd: BorrowedFd<'_>) -> Result<(), ExecutableError> {
    let fdinfo =
        std::fs::read_to_string(PathBuf::from(format!("/proc/self/fdinfo/{}", pidfd.as_raw_fd())))?;
    let pid = fdinfo
        .lines()
        .find_map(|line| line.strip_prefix("Pid:").map(str::trim))
        .ok_or(ExecutableError::InvalidPidfd)?
        .parse::<i64>()
        .map_err(|_| ExecutableError::InvalidPidfd)?;
    if pid < 0 {
        return Err(ExecutableError::PeerExited);
    }
    let actual_pid = u32::try_from(pid).map_err(|_| ExecutableError::InvalidPidfd)?;
    if actual_pid != expected_pid {
        return Err(ExecutableError::PeerPidMismatch {
            expected: expected_pid,
            actual: actual_pid,
        });
    }
    Ok(())
}

/// Required relationship between the kernel peer UID and requested target UID.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CallerRelation {
    /// Only a UID 0 helper may use the rule.
    RootOnly,
    /// The peer UID must equal the requested target UID.
    TargetUser,
    /// Either UID 0 or the requested target UID may use the rule.
    RootOrTargetUser,
}

impl CallerRelation {
    const fn accepts(self, peer_uid: u32, target_uid: u32) -> bool {
        match self {
            Self::RootOnly => peer_uid == 0,
            Self::TargetUser => peer_uid == target_uid,
            Self::RootOrTargetUser => peer_uid == 0 || peer_uid == target_uid,
        }
    }
}

/// One exact administrator-controlled authentication entry point.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AuthorizationRule {
    /// Exact PAM or desktop service name.
    pub service: ServiceName,
    /// Exact operation category.
    pub purpose: AuthenticationPurpose,
    /// Allowed relationship between caller and target account.
    pub caller: CallerRelation,
    /// Root-owned executable fingerprints permitted to invoke this entry point.
    pub executables: Vec<ExecutableFingerprint>,
}

/// Validated bounded authorization policy.
pub struct AuthorizationPolicy {
    rules: Vec<AuthorizationRule>,
}

impl AuthorizationPolicy {
    /// Validate and construct an exact-match authorization policy.
    ///
    /// # Errors
    ///
    /// Returns [`PolicyError`] for an empty or excessive rule set, duplicate service/purpose
    /// entries, empty/excessive executable allowlists, duplicate fingerprints, or a non-root sudo
    /// caller relation.
    pub fn new(rules: Vec<AuthorizationRule>) -> Result<Self, PolicyError> {
        if rules.is_empty() || rules.len() > MAX_AUTHORIZATION_RULES {
            return Err(PolicyError::InvalidRuleCount { actual: rules.len() });
        }
        for (index, rule) in rules.iter().enumerate() {
            if rule.executables.is_empty() || rule.executables.len() > MAX_EXECUTABLES_PER_RULE {
                return Err(PolicyError::InvalidExecutableCount {
                    service: rule.service.clone(),
                    actual: rule.executables.len(),
                });
            }
            if rule.purpose == AuthenticationPurpose::Sudo
                && rule.caller != CallerRelation::RootOnly
            {
                return Err(PolicyError::SudoMustBeRootOnly);
            }
            if rules[..index].iter().any(|existing| {
                existing.service == rule.service && existing.purpose == rule.purpose
            }) {
                return Err(PolicyError::DuplicateRule {
                    service: rule.service.clone(),
                    purpose: rule.purpose,
                });
            }
            for (fingerprint_index, fingerprint) in rule.executables.iter().enumerate() {
                if rule.executables[..fingerprint_index].contains(fingerprint) {
                    return Err(PolicyError::DuplicateExecutable {
                        service: rule.service.clone(),
                        fingerprint: *fingerprint,
                    });
                }
            }
        }
        Ok(Self { rules })
    }

    /// Authorize one request before camera, template, or model work begins.
    ///
    /// # Errors
    ///
    /// Returns [`AuthorizationError`] unless service and purpose match an exact rule, peer/target
    /// UID relationship is allowed, and verified executable evidence matches the rule allowlist.
    pub fn authorize(
        &self,
        peer: PeerIdentity,
        executable: Option<VerifiedExecutable>,
        request: &RequestContext,
    ) -> Result<AuthorizationGrant, AuthorizationError> {
        let rule = self
            .rules
            .iter()
            .find(|rule| rule.service == request.service && rule.purpose == request.purpose)
            .ok_or(AuthorizationError::ServiceNotAllowed)?;
        if !rule.caller.accepts(peer.uid, request.target_uid) {
            return Err(AuthorizationError::PeerTargetMismatch {
                peer_uid: peer.uid,
                target_uid: request.target_uid,
            });
        }
        let executable = executable.ok_or(AuthorizationError::ExecutableEvidenceRequired)?;
        if !rule.executables.contains(&executable.fingerprint()) {
            return Err(AuthorizationError::ExecutableNotAllowed);
        }
        Ok(AuthorizationGrant {
            transaction_id: request.transaction_id,
            peer,
            target_uid: request.target_uid,
            service: request.service.clone(),
            purpose: request.purpose,
            executable: executable.fingerprint(),
        })
    }
}

/// Reusable issuer bound to one validated policy, kernel peer, executable, service, and purpose.
///
/// Each issued grant still passes through [`AuthorizationPolicy::authorize`] with a fresh
/// transaction identifier and exact target UID.
pub struct BoundAuthorizationIssuer {
    policy: AuthorizationPolicy,
    peer: PeerIdentity,
    executable: VerifiedExecutable,
    service: ServiceName,
    purpose: AuthenticationPurpose,
}

impl BoundAuthorizationIssuer {
    /// Bind immutable caller evidence and an exact service/purpose to a validated policy.
    #[must_use]
    pub const fn new(
        policy: AuthorizationPolicy,
        peer: PeerIdentity,
        executable: VerifiedExecutable,
        service: ServiceName,
        purpose: AuthenticationPurpose,
    ) -> Self {
        Self { policy, peer, executable, service, purpose }
    }

    /// Issue a new policy-authorized grant for one exact target UID.
    ///
    /// # Errors
    ///
    /// Returns [`AuthorizationError`] unless the bound peer, executable, service, and purpose are
    /// admitted for the requested target UID.
    pub fn issue(&self, target_uid: u32) -> Result<AuthorizationGrant, AuthorizationError> {
        let request = RequestContext {
            transaction_id: TransactionId::generate(),
            target_uid,
            service: self.service.clone(),
            purpose: self.purpose,
        };
        self.policy.authorize(self.peer, Some(self.executable), &request)
    }
}

/// Immutable authorization result bound to the request and kernel peer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthorizationGrant {
    /// Random request identifier bound during authorization.
    pub transaction_id: TransactionId,
    /// Kernel connection identity.
    pub peer: PeerIdentity,
    /// Account whose enrolled template may be accessed.
    pub target_uid: u32,
    /// Exact authorized service.
    pub service: ServiceName,
    /// Exact authorized operation category.
    pub purpose: AuthenticationPurpose,
    /// Verified caller executable identity.
    pub executable: ExecutableFingerprint,
}

/// Failure to derive trustworthy executable evidence.
#[derive(Debug, Error)]
pub enum ExecutableError {
    /// Metadata lookup on the open file failed.
    #[error("unable to inspect caller executable: {0}")]
    Io(#[from] io::Error),
    /// Caller evidence was not a regular file.
    #[error("caller executable is not a regular file")]
    NotRegularFile,
    /// Caller executable was not owned by root.
    #[error("caller executable is not root-owned; owner UID is {actual_uid}")]
    NotRootOwned {
        /// Actual file owner UID.
        actual_uid: u32,
    },
    /// Caller executable could be modified by a non-root group or user.
    #[error("caller executable mode {mode:o} permits untrusted writes")]
    WritableByUntrustedUser {
        /// Permission bits observed on the open file.
        mode: u32,
    },
    /// File permissions do not contain an executable bit.
    #[error("caller executable mode {mode:o} is not executable")]
    NotExecutable {
        /// Permission bits observed on the open file.
        mode: u32,
    },
    /// The supplied descriptor does not expose Linux pidfd metadata.
    #[error("caller process descriptor is not a valid pidfd")]
    InvalidPidfd,
    /// The pidfd-pinned peer exited during executable verification.
    #[error("caller process exited during executable verification")]
    PeerExited,
    /// The pidfd does not identify the process captured by `SO_PEERCRED`.
    #[error("pidfd identifies PID {actual}, but SO_PEERCRED identified PID {expected}")]
    PeerPidMismatch {
        /// PID captured from `SO_PEERCRED`.
        expected: u32,
        /// PID reported by pidfd metadata.
        actual: u32,
    },
}

/// Invalid administrator authorization policy.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum PolicyError {
    /// Rule count must be bounded and non-zero.
    #[error("authorization policy has invalid rule count {actual}")]
    InvalidRuleCount {
        /// Actual rule count.
        actual: usize,
    },
    /// Each exact entry point requires a bounded non-empty executable allowlist.
    #[error("authorization rule {service} has invalid executable count {actual}")]
    InvalidExecutableCount {
        /// Affected service.
        service: ServiceName,
        /// Actual executable count.
        actual: usize,
    },
    /// A service/purpose pair may appear only once.
    #[error("duplicate authorization rule for {service} and {purpose:?}")]
    DuplicateRule {
        /// Duplicate service.
        service: ServiceName,
        /// Duplicate purpose.
        purpose: AuthenticationPurpose,
    },
    /// Fingerprints must not repeat within one rule.
    #[error("duplicate executable fingerprint in authorization rule {service}: {fingerprint:?}")]
    DuplicateExecutable {
        /// Affected service.
        service: ServiceName,
        /// Duplicate fingerprint.
        fingerprint: ExecutableFingerprint,
    },
    /// Sudo authentication must only accept a UID 0 helper.
    #[error("sudo authorization rules must be root-only")]
    SudoMustBeRootOnly,
}

/// Request authorization failure.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum AuthorizationError {
    /// No exact service/purpose rule exists.
    #[error("authentication service or purpose is not allowed")]
    ServiceNotAllowed,
    /// Peer UID cannot act for the requested target UID under this rule.
    #[error("peer UID {peer_uid} cannot authenticate target UID {target_uid}")]
    PeerTargetMismatch {
        /// Kernel peer UID.
        peer_uid: u32,
        /// Requested target UID.
        target_uid: u32,
    },
    /// No pidfd-bound verified executable was supplied.
    #[error("verified caller executable evidence is required")]
    ExecutableEvidenceRequired,
    /// Verified executable fingerprint is absent from the exact rule.
    #[error("caller executable is not allowed for this service and purpose")]
    ExecutableNotAllowed,
}

#[cfg(test)]
mod tests {
    use std::os::unix::net::UnixStream;

    use faceauth_transport::{PeerStream, TransportConfig};

    use super::*;

    const ALLOWED: ExecutableFingerprint = ExecutableFingerprint { device: 8, inode: 42 };
    const OTHER: ExecutableFingerprint = ExecutableFingerprint { device: 8, inode: 43 };

    fn service() -> Result<ServiceName, faceauth_protocol::ServiceNameError> {
        ServiceName::parse("faceauth-test")
    }

    fn rule() -> Result<AuthorizationRule, faceauth_protocol::ServiceNameError> {
        Ok(AuthorizationRule {
            service: service()?,
            purpose: AuthenticationPurpose::Test,
            caller: CallerRelation::RootOrTargetUser,
            executables: vec![ALLOWED],
        })
    }

    fn request() -> Result<RequestContext, faceauth_protocol::ServiceNameError> {
        Ok(RequestContext {
            transaction_id: faceauth_protocol::TransactionId::generate(),
            target_uid: 1000,
            service: service()?,
            purpose: AuthenticationPurpose::Test,
        })
    }

    fn peer(uid: u32) -> PeerIdentity {
        PeerIdentity { pid: 123, uid, gid: uid }
    }

    fn executable(fingerprint: ExecutableFingerprint) -> VerifiedExecutable {
        VerifiedExecutable { fingerprint }
    }

    #[test]
    fn exact_rule_authorizes_root_or_target_user() -> Result<(), Box<dyn std::error::Error>> {
        let policy = AuthorizationPolicy::new(vec![rule()?])?;
        let request = request()?;

        let root_grant = policy.authorize(peer(0), Some(executable(ALLOWED)), &request)?;
        assert_eq!(root_grant.transaction_id, request.transaction_id);
        assert!(policy.authorize(peer(1000), Some(executable(ALLOWED)), &request).is_ok());
        Ok(())
    }

    #[test]
    fn cross_uid_and_missing_executable_fail_closed() -> Result<(), Box<dyn std::error::Error>> {
        let policy = AuthorizationPolicy::new(vec![rule()?])?;
        let request = request()?;

        assert_eq!(
            policy.authorize(peer(1001), Some(executable(ALLOWED)), &request),
            Err(AuthorizationError::PeerTargetMismatch { peer_uid: 1001, target_uid: 1000 })
        );
        assert_eq!(
            policy.authorize(peer(1000), None, &request),
            Err(AuthorizationError::ExecutableEvidenceRequired)
        );
        Ok(())
    }

    #[test]
    fn executable_and_service_purpose_must_match_exactly() -> Result<(), Box<dyn std::error::Error>>
    {
        let policy = AuthorizationPolicy::new(vec![rule()?])?;
        let mut request = request()?;
        assert_eq!(
            policy.authorize(peer(1000), Some(executable(OTHER)), &request),
            Err(AuthorizationError::ExecutableNotAllowed)
        );

        request.purpose = AuthenticationPurpose::ScreenUnlock;
        assert_eq!(
            policy.authorize(peer(1000), Some(executable(ALLOWED)), &request),
            Err(AuthorizationError::ServiceNotAllowed)
        );
        Ok(())
    }

    #[test]
    fn bound_issuer_reauthorizes_each_target_with_fresh_transaction()
    -> Result<(), Box<dyn std::error::Error>> {
        let issuer = BoundAuthorizationIssuer::new(
            AuthorizationPolicy::new(vec![rule()?])?,
            peer(0),
            executable(ALLOWED),
            service()?,
            AuthenticationPurpose::Test,
        );
        let first = issuer.issue(1000)?;
        let second = issuer.issue(1001)?;
        assert_eq!(first.target_uid, 1000);
        assert_eq!(second.target_uid, 1001);
        assert_ne!(first.transaction_id, second.transaction_id);

        let denied = BoundAuthorizationIssuer::new(
            AuthorizationPolicy::new(vec![rule()?])?,
            peer(1001),
            executable(ALLOWED),
            service()?,
            AuthenticationPurpose::Test,
        );
        assert_eq!(
            denied.issue(1000),
            Err(AuthorizationError::PeerTargetMismatch { peer_uid: 1001, target_uid: 1000 })
        );
        Ok(())
    }

    #[test]
    fn ambiguous_and_unbounded_policies_are_rejected() -> Result<(), Box<dyn std::error::Error>> {
        let exact = rule()?;
        assert!(matches!(
            AuthorizationPolicy::new(vec![exact.clone(), exact]),
            Err(PolicyError::DuplicateRule { .. })
        ));
        assert!(matches!(
            AuthorizationPolicy::new(Vec::new()),
            Err(PolicyError::InvalidRuleCount { actual: 0 })
        ));

        let mut duplicate_executable = rule()?;
        duplicate_executable.executables.push(ALLOWED);
        assert!(matches!(
            AuthorizationPolicy::new(vec![duplicate_executable]),
            Err(PolicyError::DuplicateExecutable { .. })
        ));
        Ok(())
    }

    #[test]
    fn sudo_can_only_be_root_only() -> Result<(), Box<dyn std::error::Error>> {
        let mut sudo = rule()?;
        sudo.purpose = AuthenticationPurpose::Sudo;
        sudo.caller = CallerRelation::TargetUser;
        assert_eq!(
            AuthorizationPolicy::new(vec![sudo]).err(),
            Some(PolicyError::SudoMustBeRootOnly)
        );
        Ok(())
    }

    #[test]
    fn pidfd_must_match_the_socket_peer() -> Result<(), Box<dyn std::error::Error>> {
        let (socket, _peer) = UnixStream::pair()?;
        let stream = PeerStream::connect(socket, TransportConfig::default())?;
        let mut wrong_peer = stream.peer();
        wrong_peer.pid = wrong_peer.pid.checked_add(1).ok_or("PID overflow")?;

        assert!(matches!(
            VerifiedExecutable::from_peer(wrong_peer, stream.peer_pidfd()),
            Err(ExecutableError::PeerPidMismatch { .. })
        ));
        Ok(())
    }

    #[test]
    fn root_owned_executable_permissions_are_enforced() -> Result<(), Box<dyn std::error::Error>> {
        let shell = File::open("/bin/sh")?;
        assert!(VerifiedExecutable::from_verified_file(&shell).is_ok());

        let passwd = File::open("/etc/passwd")?;
        assert!(matches!(
            VerifiedExecutable::from_verified_file(&passwd),
            Err(ExecutableError::NotExecutable { .. })
        ));
        Ok(())
    }
}
