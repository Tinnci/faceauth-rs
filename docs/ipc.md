# Local IPC security boundary

The future PAM bridge and desktop clients will communicate with the privileged
daemon over a root-owned Unix socket. System D-Bus remains appropriate for
management and status, but the bounded authentication hot path needs a narrow
transport with explicit peer-credential checks.

## Trust rules

- The daemon derives caller UID, PID, and executable identity from Unix peer
  credentials. Serialized fields never prove caller identity.
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

## PAM staging

The first PAM integration must use a dedicated `faceauth-test` service. It must
exercise invalid peer credentials, malformed frames, response replay, timeout,
daemon restart, camera loss, and password fallback before any SDDM, KDE locker,
Polkit, login, or sudo configuration is modified.
