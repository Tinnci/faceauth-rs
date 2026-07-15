# Desktop-independent management contract

`faceauth-management` defines the version-1 contract that a future system D-Bus adapter, KDE
System Settings module, and enrollment OSD consume. The Rust crate owns the security-relevant
operation lifecycle; a desktop adapter must remain a thin translation layer.

Stable identifiers:

- bus/interface: `org.faceauth.Manager1`;
- object path: `/org/faceauth/Manager1`;
- Polkit action: `org.faceauth.enroll`;
- schema version: `1`.

The matching introspection input is
[`org.faceauth.Manager1.xml`](../contrib/dbus/org.faceauth.Manager1.xml). It is a packaging input and
is not installed or claimed by the current daemon.

## Authorization and identity

`BeginEnrollment` cannot directly create an operation from caller-supplied UID data. The D-Bus
adapter must derive the unique bus sender, resolve its kernel credentials, perform the
`org.faceauth.enroll` Polkit check, and have the daemon validate the resulting exact root broker
grant. Only then can `AuthorizedEnrollment::from_grant` produce the opaque value accepted by
`ManagementCoordinator::start`.

Enrollment state queries must likewise enforce caller-to-target UID policy. Serialized UIDs,
operation IDs, object paths, and bus names never prove identity. Disconnect handling must cancel
the sender's active operation rather than leaving camera work running.

## Operation lifecycle

The coordinator exposes one operation slot, matching the single calibrated IR/RGB pair. Every
operation has an unpredictable UUID, an authorized numeric UID, and a fixed 5–120 second monotonic
deadline. Progress, cancellation, and completion require the exact UID and operation ID. Wrong
bindings do not consume or alter the active operation.

Public progress is limited to `preparing`, `position-face`, `hold-still`, `active-challenge`, and
`processing`. Terminal results are `completed`, `cancelled`, `timed-out`, or `failed`. Neither the
Rust types nor the D-Bus XML contain raw frames, landmarks, image quality, match scores, PAD scores,
embeddings, or templates. Localized UI strings are selected by the client from these stable codes.

Cancellation and timeout are coordinator-owned. The biometric worker may report successful or
failed completion but cannot synthesize cancellation/timeout. Completion is one-shot and releases
the operation slot.

## KDE boundary

The future KCM should display enrollment state, begin/cancel enrollment after Polkit authorization,
and render safe progress. It must not load models, open cameras, access encrypted storage, or make
authentication decisions. The future OSD should subscribe only while its operation ID is active and
must ignore signals for all other operations.

No Plasma, KScreenLocker, SDDM, or PAM configuration is modified by this milestone.
