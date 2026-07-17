# KDE face-authentication experience

`ui/kde` contains a KF6 System Settings module and reusable authentication OSD view. The UI is a
thin client: it receives closed progress and terminal codes only. Raw frames, landmarks, quality
measurements, match scores, PAD scores, embeddings, templates, and keys have no QML property or
D-Bus representation.

## System Settings module

The KCM connects to `org.faceauth.Manager1` on the system bus, requires management schema 2, reads
the authenticated enrollment state, and starts or cancels enrollment for the current numeric UID.
Signals are accepted only for the exact active operation ID. A generation counter invalidates stale
asynchronous replies after service restart, cancellation, completion, or a newer refresh. The
completion-before-method-reply race is handled without leaving the page permanently busy. Unknown
operation IDs, progress values, or terminal results fail closed as a protocol mismatch.

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
