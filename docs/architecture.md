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
3. `faceauth-transport`: peer-credentialed, length-bounded local socket framing.
4. `faceauth-authz`: exact service, purpose, UID, and executable authorization.
5. `faceauth-session`: one-shot connection-bound transaction lifecycle and deadlines.
6. `faceauth-capture`: bounded V4L2 streaming and monotonic IR/RGB frame pairing.
7. `faceauth-liveness`: randomized, deadline-bound active-challenge state machine.
8. `faceauth-enrollment`: authorized, bounded multi-observation registration that
   retains only zeroizing embeddings and emits one encrypted-storage record.
9. `faceauth-presence`: optional versioned thinkpad-hpd D-Bus hints and a generic
   capture-activity lease registry; never part of identity evidence.
10. `faceauth-model`: bounded schema-v4 provenance, digest, preprocessing, semantic
   output roles, and exact tensor-contract admission.
11. `faceauth-inference`: dynamically loaded, resource-bounded ONNX Runtime sessions
   whose graph I/O must exactly match an admitted manifest.
12. `faceauth-daemon`: camera ownership, inference, liveness, encrypted templates,
   audit events, and authentication transactions.
13. `faceauth-management`: desktop-independent enrollment authorization and operation lifecycle.
14. `faceauth-management-dbus`: thin asynchronous zbus Manager1 adapter with real system-bus
   sender-UID and PolicyKit clients plus injected template-state and root-grant boundaries; it has
   no permissive default backend.
15. `pam_faceauth`: future minimal PAM bridge. It will be tested against a
   dedicated PAM service before any system login stack is touched.
16. `faceauth-cli`: enrollment, removal, diagnostics, dry-run, and recovery.
17. KDE KCM and lock-screen status UI: optional clients over stable APIs.

The authentication transport contract is described in [ipc.md](ipc.md). Caller
numeric peer identity comes from Unix peer credentials, not serialized request
fields. Executable authorization additionally requires a pidfd-backed check.

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

The runtime is selected by an explicit trusted local path and is never downloaded
by the service. Production artifacts and their containing directories must be
root-owned and immutable to non-root users. Sessions are CPU-bound and sequential
with fixed thread, model-size, tensor-rank, and tensor-element ceilings. The graph
must expose exactly the input and outputs declared by its schema-v4 manifest.
The manifest also fixes bilinear half-pixel resizing, channel order, and per-channel
affine normalization. Tightly packed Gray8/RGB8/BGR8/YUYV sources are converted
into zeroizing float buffers; copied outputs are shape-checked, finite-only, and
zeroized on drop. A per-run watchdog requests ONNX Runtime cancellation at a hard
configured ceiling, while future worker-process isolation will provide a stronger
kill boundary for runtimes that do not promptly honor termination.

Role adapters are fail-closed. Face embeddings are bounded, finite, non-degenerate,
L2-normalized, and compared only when their complete manifest compatibility digest
and dimension match. Passive-PAD adapters accept only a semantic scalar live
probability; model-specific calibrated thresholds remain outside the generic
runtime and have no production defaults.

The daemon is the bridge between inference and encrypted storage: enrollment can
construct a template only from a normalized `FaceEmbedding`, and authentication
reconstructs an enrolled comparison vector only after the authenticated record's
schema, unit norm, compatibility digest, and dimension match the active session.
Its authentication worker binds one private session cancellation token across
dual-camera pairing, deterministic preprocessing, and active-liveness observation
callbacks. Each underlying crate remains desktop/session independent, while the
daemon facade ensures cancellation cannot stop at one stage and leave later camera
or landmark work running.
The authentication evidence orchestrator then requires a valid paired IR/visible
timestamp, explicitly calibrated IR and visible PAD scores bound to exact model
compatibility digests, a completed randomized challenge, a finite quality value,
and the compatible template comparison before the core policy can accept. An
optional fusion PAD model is also exact-contract-bound; missing, duplicated,
unexpected, or incompatible PAD evidence fails closed. This boundary contains no
raw image type and therefore cannot persist capture frames.
Multi-sample registration and its dedicated root Polkit-broker grant are described
in [enrollment.md](enrollment.md).
The optional HPD contract and its fail-open semantics are described in
[presence.md](presence.md).

Landmark models provide bounded eye-openness, yaw, and face-count measurements to
the active-challenge state machine. They do not decide challenge success. The
state machine requires a neutral baseline, randomized action, neutral recovery,
fresh paired frames, and monotonic deadline compliance. See
[liveness.md](liveness.md).

## Template storage

Only derived templates and metadata are persisted. Templates are encrypted with
an authenticated cipher using a machine key protected by TPM 2.0 when available.
A root-only file-key fallback is permitted only with a visible diagnostic warning.
Template files are owned by root, keyed by numeric UID, versioned, and replaced
atomically. Decrypted embeddings are zeroized after use.
Implementation details and the current machine-bound TPM policy are documented
in [storage.md](storage.md).

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
