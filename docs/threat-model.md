# Threat model

## Protected assets

- biometric templates and model-derived identity data;
- the one-shot authentication result;
- camera frames while an authentication transaction is active;
- enrollment authorization and template replacement;
- password fallback and account recovery.

## In-scope attackers

- a person presenting a photograph, screen replay, video, mask, or look-alike;
- an unprivileged local process attempting to replay or forge daemon messages;
- a malicious local process attempting to read templates or camera frames;
- stale device nodes, virtual cameras, prerecorded frames, and timestamp replay;
- malformed model, configuration, IPC, or PAM inputs;
- accidental lockout caused by service, model, camera, or desktop failure.

## Required controls

- IR is mandatory for production policy; visible light is paired by default.
- Passive presentation-attack detection is combined with a randomized active challenge.
- Authentication is bounded by monotonic deadlines and request-specific nonces.
- Peer credentials and requested UID must agree with trusted caller policy.
- Authentication entry points require exact service/purpose rules and a
  pidfd-bound, root-owned executable allowlist match.
- The PAM bridge fails closed and never reports success on daemon errors.
- Password authentication remains available and independently testable.
- Enrollment requires an existing credential and cannot overwrite another UID silently.
- Raw frames are not logged, included in crash dumps, or retained by default.
- Templates are encrypted, root-owned, integrity-protected, and atomically updated.
- Model provenance, hashes, licenses, and thresholds are versioned with each template.

## Out of scope for the initial release

- resistance to a fully compromised root account or kernel;
- certification equivalent to Windows Hello Enhanced Sign-in Security;
- remote biometric authentication;
- replacing passwords, FIDO2 keys, or full-disk encryption credentials;
- claiming presentation-attack certification without an independent test corpus.
