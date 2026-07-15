# Optional thinkpad-hpd presence integration

`faceauth-presence` is an independent adapter for the existing system D-Bus
service:

- service: `org.thinkpad.HumanPresence1`;
- object: `/org/thinkpad/HumanPresence1`;
- interface: `org.thinkpad.HumanPresence1`;
- method: `GetState() -> (b available, b present, i raw_value)`.

The adapter is deliberately used by a background refresh worker or diagnostics,
never by the PAM/lock-screen authentication hot path. D-Bus absence, timeout,
malformed state, and stale state clear an optional cache and do not reject or
accept authentication. A fresh `available && present` cache entry can only hint
that camera capture may be prewarmed.

The generic faceauth activity registry is independent of HPD-specific types. It
exposes the stable conceptual states `idle`, `authentication`, and `enrollment`
through the reserved `org.faceauth.Capture1` contract. Scoped leases make
enrollment take precedence over authentication for desktop/HPD inhibition. HPD
must treat this as a UI/locking coordination hint, not identity evidence.

Read-only validation on a running system:

```bash
cargo run -p faceauth-daemon -- presence-doctor
```

This command only calls the existing D-Bus method. It does not change HPD state,
claim a D-Bus name, alter power settings, or restart any service.
