# Local IPC security boundary

The future PAM bridge and desktop clients will communicate with the privileged
daemon over a root-owned Unix socket. System D-Bus remains appropriate for
management and status, but the bounded authentication hot path needs a narrow
transport with explicit peer-credential checks.

## Trust rules

- The transport captures caller UID, GID, and PID from Linux `SO_PEERCRED` and
  pins the process with `SO_PEERPIDFD`
  before reading caller-controlled bytes. Serialized fields never prove caller
  identity. Executable identity requires a separate pidfd-backed authorization
  step and is not implied by the numeric PID alone.
- Each request carries a random transaction identifier and every response echoes
  it. A client accepts exactly one terminal response for its active transaction.
- The protocol is versioned and rejects unknown versions before camera or model
  work begins.
- Service names are length-limited and character-allowlisted. Authentication
  purpose is a closed enum rather than a caller-controlled policy string.
- Every transaction has a server-side deadline and cancellation path. The wire
  protocol does not let a caller extend that deadline.
- Decision and rejection reasons are closed enums. Human-readable localized text
  belongs in clients and is never parsed as policy.
- Every frame has a four-byte big-endian length prefix. Empty frames and payloads
  over 16 KiB are rejected before payload allocation; configuration cannot raise
  the ceiling beyond 64 KiB. Connected reads and writes have fixed timeouts.
- Protocol version 2 adds transaction-bound progress prompts for positioning,
  active challenges, recovery, and processing. These messages contain no raw
  frames, landmarks, similarity scores, liveness scores, or templates.

The daemon listener binds only inside an existing root-owned directory that is
not writable by group or others. It accepts only socket modes `0600` or `0660`,
never unlinks or replaces an existing path, and captures credentials and pidfd
immediately after `accept`. Stale socket cleanup and optional group ownership are
explicit service-manager operations.

`SO_PEERCRED` establishes connection identity, not authorization. The daemon must
still verify the allowed service/purpose pair, target UID relationship, executable
policy, transaction capacity, and enrollment state before starting camera work.

## Authorization policy

`faceauth-authz` accepts at most 64 exact service/purpose rules and 32 executable
fingerprints per rule. There are no wildcard services or caller-provided policy
strings. Each rule chooses one caller relationship: UID 0 only, target UID only,
or UID 0/target UID. Sudo rules are always UID 0 only.

Executable evidence is resolved through the socket peer pidfd, checked before and
after opening `/proc/<pid>/exe`, and reduced to filesystem device and inode. The
opened file must be a regular executable owned by root and not writable by group
or others. Missing evidence, PID mismatch, process exit, permission weakness, or
an absent fingerprint fails before cameras, templates, or models are accessed.

Package upgrades normally replace an executable inode and therefore invalidate
the old fingerprint. Updating the root-controlled authorization policy is an
explicit deployment action, not an automatic path-based trust decision.

## Transaction lifecycle

`faceauth-session` initially exposes one global transaction slot because the
daemon owns one calibrated IR/RGB pair. Starting requires a completed
authorization grant whose transaction ID is copied from the decoded request.
The session is additionally bound to an unpredictable daemon-internal connection
token that is never serialized.

Progress, completion, and cancellation require both the exact transaction ID and
the originating connection token. A wrong connection or ID cannot alter or reap
the active transaction. The monotonic deadline is fixed at start and cannot be
extended by wire input.

Completion consumes the slot and produces exactly one terminal response.
`Cancelled` and `TimedOut` are manager-owned results and cannot be injected by
the inference pipeline. At or after the deadline, progress, completion, or
cancellation produces `TimedOut`. An expired transaction is not silently replaced
by a new request: the daemon event loop must call `expire()`, deliver or audit the
timeout terminal result, and only then admit another transaction.

## PAM staging

The first PAM integration must use a dedicated `faceauth-test` service. It must
exercise invalid peer credentials, malformed frames, response replay, timeout,
daemon restart, camera loss, and password fallback before any SDDM, KDE locker,
Polkit, login, or sudo configuration is modified.

The current `BoundaryService` implements decode → peer executable verification →
exact authorization → session start/cancel → framed response. The installed
production `serve` command still refuses to run until inference, passive PAD,
active-liveness orchestration, and enrollment are complete; the boundary never
substitutes a scaffold response for biometric success.
