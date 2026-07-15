# Desktop-independent management contract

`faceauth-management` defines the version-1 contract that the `faceauth-management-dbus` adapter,
future KDE System Settings module, and enrollment OSD consume. The core crate owns the
security-relevant operation lifecycle; the D-Bus crate remains a thin translation layer.

Stable identifiers:

- bus/interface: `org.faceauth.Manager1`;
- object path: `/org/faceauth/Manager1`;
- Polkit action: `org.faceauth.enroll`;
- schema version: `1`.

The matching introspection input is
[`org.faceauth.Manager1.xml`](../contrib/dbus/org.faceauth.Manager1.xml). It is a packaging input and
is not installed or claimed by the current daemon.

## D-Bus adapter

`faceauth-management-dbus` implements the `org.faceauth.Manager1` method and signal shape with zbus
4.x. It reads the unique sender from the D-Bus message header, never from a serialized method
argument. Its object-safe asynchronous `AuthorizationBackend` permits credential and PolicyKit
lookups without blocking the zbus executor.

`SystemBusAuthority` implements the real system-bus operations: it accepts only a D-Bus unique
sender name, resolves its UID with `GetConnectionUnixUser`, and calls PolicyKit
`CheckAuthorization` for the exact `org.faceauth.enroll` action and `system-bus-name` subject. The
call permits user interaction because enrollment requires a fresh administrator decision; a false
authorization result, malformed sender, missing authority, transport error, or credential lookup
failure is rejected.

`SystemBusBackend` composes that authority with an injected authenticated-template state source and
dedicated root grant issuer. It rechecks caller UID before state access and before PolicyKit. After
PolicyKit succeeds it still passes the issued grant through `AuthorizedEnrollment::from_grant`, so
only an exact root `faceauth-enroll`/Polkit grant is admitted. UID mismatch stops before PolicyKit is
called. The crate contains no permissive defaults.

The daemon builds the grant issuer with `management_enrollment_grant_issuer`. Its
`BoundAuthorizationIssuer` retains one validated `AuthorizationPolicy`, kernel peer,
`VerifiedExecutable`, service, and purpose. Construction requires UID 0 and preflights the exact
`faceauth-enroll`/Polkit rule. Every real target is reauthorized by the original policy with a fresh
transaction ID, and the D-Bus adapter revalidates the resulting grant before use.

The daemon supplies authenticated template state through
`management_template_state_source`. This wraps its existing encrypted `TemplateSource` in a
single-worker bounded queue, so TPM/key retrieval and authenticated decryption never run on the
zbus executor. The worker returns only a boolean; the zeroizing template record is dropped inside
the worker. Missing templates return `false`, while corrupt, tampered, or unsafe storage fails the
query. Queue saturation and worker loss also fail closed.

The adapter currently admits only a caller managing its own numeric UID. `GetEnrollmentState`
delegates the enrolled-template lookup to the injected backend after that identity check;
coordinator busy state is not mistaken for enrollment state. `BeginEnrollment` also checks that the
returned authorization proof is bound to that exact UID. Each active operation is then bound to the
exact unique D-Bus sender, target UID, and opaque UUID. Cancellation requires all three; another
connection owned by the same UID cannot cancel the operation. Backend failures, absent senders,
mismatched senders or UIDs, malformed operation IDs, and poisoned coordinator state fail closed.

`ManagementDisconnectHandle::watch` subscribes to the bus daemon's `NameOwnerChanged` signal. It
ignores well-known names, owner acquisition/replacement, and unrelated unique names. When the exact
operation sender loses its owner, the handle consumes the operation as `cancelled`, releases camera
capacity, and emits the stable terminal signal. The service must start this watcher before exposing
Manager1 methods and stop if the watcher exits; continuing without disconnect cleanup is unsafe.
If the initial `EnrollmentProgress` signal cannot be sent after `BeginEnrollment`, the adapter rolls
the just-created operation back immediately.

The daemon obtains a `ManagementWorkerHandle` that uses the adapter's exact coordinator and
monotonic clock for progress, completion, and timeout reaping. It can emit only the resulting
`ManagementUpdate` values through the adapter. Signals are restricted to the stable public codes
listed below. No test claims the system bus name, and no production Polkit broker or D-Bus service
activation is enabled by this milestone. Reviewed production broker evidence, configuration, and
service startup wiring remain required before activation.

## Authorization and identity

`BeginEnrollment` cannot directly create an operation from caller-supplied UID data. The D-Bus
adapter derives the unique bus sender and delegates kernel-credential resolution and the
`org.faceauth.enroll` Polkit check to the injected backend. The daemon must validate the resulting
exact root broker grant. Only then can `AuthorizedEnrollment::from_grant` produce the opaque value
accepted by `ManagementCoordinator::start`.

Enrollment state queries must likewise enforce caller-to-target UID policy. Serialized UIDs,
operation IDs, object paths, and bus names never prove identity. The adapter uses only the sender
attached by the bus daemon and the kernel UID resolved for that unique name.

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
