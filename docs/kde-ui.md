# KDE face-authentication experience

`ui/kde` contains a KF6 System Settings module and reusable authentication OSD view. The UI is a
thin client: it receives closed progress and terminal codes only. Raw frames, landmarks, quality
measurements, match scores, PAD scores, embeddings, templates, and keys have no QML property or
D-Bus representation.

## System Settings module

The KCM is registered in Plasma 6 under
`System Settings → Security & Privacy → Face Authentication` and is searchable by face,
authentication, biometrics, login, lock-screen, and IR keywords. Each KCM instance creates a private
system-bus connection to `org.faceauth.Manager1`, requires management schema 2, reads the
authenticated enrollment state, and starts or cancels enrollment for the current numeric UID. The
private connection is disconnected when the page is destroyed, so the daemon's sender-disconnect
watcher also cancels an enrollment that is still waiting for authorization and has not returned an
operation ID yet.

The page has explicit `loading`, `ready`, `authorizing`, `starting`, `capturing`, `cancelling`, and
`terminal` states. Initial requests complete as one joined refresh, avoiding a transient
"disconnected" message. Cancel is one-shot while the request is in flight, and retry actions are
bound to their context: reconnect, begin enrollment again, or retry cancellation. Signals are
accepted only for the exact active operation ID. A generation counter invalidates stale asynchronous
replies after service restart, cancellation, completion, or a newer refresh. The
completion-before-method-reply race is handled without leaving the page permanently busy. Unknown
operation IDs, progress values, or terminal results fail closed as a protocol mismatch.

Manager1 can be constructed with a daemon-side `EnrollmentOperationController`. After Polkit and
UID authorization succeeds and the coordinator allocates an operation ID, the controller submits
that exact enrollment job. Explicit cancellation and unique-sender disconnect notify the
controller before Manager1 consumes and broadcasts the public terminal state. The daemon bridge
retains the private cancellation token; QML never receives it or a worker handle.

The operation ID remains private to the C++ adapter and is not exposed as a QML property. QML maps
only stable safe codes to localized instructions. Password fallback and the fact that raw images are
not retained remain visible in the primary setup experience.

## Authentication OSD

`FaceAuthOsd.qml` is a reusable view for an eventual KScreenLocker or Plasma-owned secure host. It
supports active guidance, sanitized success/retry/unavailable outcomes, theme icons, accessibility
roles, and an explicit password-fallback line. Biometric failure reasons intentionally collapse to
`try-again`; the OSD must not reveal whether identity, quality, PAD, or modality caused rejection.

The repository does not yet claim a production OSD host. A real host must bind the exact
authentication transaction, filter stale updates, stop observing after the one-shot terminal
result, preserve the existing password path, and never accept UI-originated success.

The shared status ring uses a restrained outward breathing pulse while capture is active, short
theme-color and scale transitions for terminal feedback, and cue-specific icons for positioning,
blink, turn, recovery, processing, cancellation, and timeout. Both shared components expose a
`reducedMotion` property so a production host can disable nonessential motion while preserving the
same state and accessibility text.

`faceauth-ui-model` provides the desktop-independent reference state machine for exact flow binding,
safe cues, sanitized outcomes, and password-fallback visibility. The next integration step is a
generated or typed Qt adapter so the Rust presentation tokens become the single source of truth for
KCM, OSD, and lock-screen clients.

## Local verification

The build and visual test install only below `/tmp`; they do not install or enable host desktop,
D-Bus, PAM, Polkit, or systemd configuration.

```console
cmake -S ui/kde -B /tmp/faceauth-kde-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_INSTALL_PREFIX=/tmp/faceauth-kde-install \
  -DFACEAUTH_BUILD_PREVIEW=ON
cmake --build /tmp/faceauth-kde-build --parallel 2
cmake --install /tmp/faceauth-kde-build

/usr/lib/qt6/bin/qmllint -I ui/kde/ui -I /usr/lib/qt6/qml \
  ui/kde/ui/main.qml ui/kde/ui/components/*.qml \
  ui/kde/ui/osd/*.qml ui/kde/tests/OsdPreview.qml

QT_QPA_PLATFORM=offscreen QT_QUICK_BACKEND=software \
  /tmp/faceauth-kde-build/bin/faceauth-osd-preview \
  "$PWD/ui/kde/tests/OsdPreview.qml" /tmp/faceauth-osd-preview.png
```

The C++ preview host injects the same KI18n context expected from KDE and captures active, success,
and safe retry states. It is a visual-regression harness, not the secure production OSD host.
The repository check also points `kcmshell6` at the temporary install prefix, verifies that
`kcm_faceauth` is discoverable, and smoke-loads the module offscreen. This tests the real KDE plugin
boundary without installing the KCM on the host.
