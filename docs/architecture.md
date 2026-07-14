# Architecture

## Security boundary

The PAM module must not open cameras, load ML models, parse configuration, or
read biometric templates. Those operations belong to a long-running privileged
daemon with a narrow authenticated local protocol.

Authentication messages are bound to a random request ID, caller UID, PAM
service name, peer credentials, and a short deadline. A successful response is
valid for one request only. The daemon fails closed on transport loss, stale
frames, camera ambiguity, model failure, or missing liveness evidence.

## Components

1. `faceauth-core`: policy, evidence, and decision types without desktop code.
2. `faceauth-protocol`: versioned messages that contain no raw images or embeddings.
3. `faceauth-daemon`: camera ownership, inference, liveness, encrypted templates,
   audit events, and authentication transactions.
4. `pam_faceauth`: future minimal PAM bridge. It will be tested against a
   dedicated PAM service before any system login stack is touched.
5. `faceauth-cli`: enrollment, removal, diagnostics, dry-run, and recovery.
6. KDE KCM and lock-screen status UI: optional clients over stable APIs.

The authentication transport contract is described in [ipc.md](ipc.md). Caller
identity always comes from Unix peer credentials, not serialized request fields.

## Camera pipeline

The preferred observation pairs one 640x360 IR frame with one 640x360 visible
frame using monotonic timestamps. Frames outside the configured skew window are
not one observation. Camera paths are resolved from stable USB and V4L2 metadata,
not hard-coded `/dev/videoN` numbers.

IR emitter control should reuse `linux-enable-ir-emitter` behavior or an audited
equivalent adapter. Vendor extension-unit probing is never performed during PAM
authentication.

## Inference

Use ONNX Runtime through Rust bindings for face detection, landmarks, embeddings,
and presentation-attack detection. Model files require explicit license,
provenance, hash, input normalization, and benchmark records before inclusion.
The project will not implement a face-recognition network from scratch.
The model admission and evaluation requirements are defined in
[model-policy.md](model-policy.md).

## Template storage

Only derived templates and metadata are persisted. Templates are encrypted with
an authenticated cipher using a machine key protected by TPM 2.0 when available.
A root-only file-key fallback is permitted only with a visible diagnostic warning.
Template files are owned by root, keyed by numeric UID, versioned, and replaced
atomically. Decrypted embeddings are zeroized after use.

## KDE and PAM

KScreenLocker already uses PAM and ships separate fingerprint and smart-card
service definitions. Face integration should follow the same modality-aware UI
pattern, while preserving the regular `kde` password stack. SDDM, sudo, Polkit,
and screen unlock are separate policy targets; sudo is opt-in because silent
camera approval is unsafe.

## HPD integration

`thinkpad-hpd` remains an independent service. Face authentication may subscribe
to presence changes to prewarm the daemon after stable return. It must ignore HPD
for the final decision. A future generic inhibitor interface can expose
`capture-active` so HPD avoids locking or showing conflicting OSD during enrollment.
