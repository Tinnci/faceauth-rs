# Local IPC security boundary

The future PAM bridge and desktop clients will communicate with the privileged
daemon over a root-owned Unix socket. System D-Bus remains appropriate for
management and status, but the bounded authentication hot path needs a narrow
transport with explicit peer-credential checks.

## Trust rules

- The transport captures caller UID, GID, and PID from Linux `SO_PEERCRED`
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

`SO_PEERCRED` establishes connection identity, not authorization. The daemon must
still verify the allowed service/purpose pair, target UID relationship, executable
policy, transaction capacity, and enrollment state before starting camera work.

## PAM staging

The first PAM integration must use a dedicated `faceauth-test` service. It must
exercise invalid peer credentials, malformed frames, response replay, timeout,
daemon restart, camera loss, and password fallback before any SDDM, KDE locker,
Polkit, login, or sudo configuration is modified.
