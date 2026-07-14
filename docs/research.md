# Linux ecosystem research

## Current community options

### Howdy

Howdy is the main Windows Hello-style Linux project and integrates through PAM.
It supports IR cameras and has broad distribution packaging. Its own security
documentation warns that it is not as secure as a password and should not be the
sole authentication method. Current issue history also includes PAM crashes,
insecure model permissions, photo/spoof concerns, IR emitter failures, KDE
integration problems, and requests to prevent silent sudo approval.

Repository: <https://github.com/boltgolt/howdy> (MIT). Its existence confirms
that a Linux PAM workflow is practical, but its own security warning and issue
history are why this project keeps camera, inference, liveness, and storage out
of PAM.

Useful lessons: PAM coverage, enrollment UX, hardware configuration, and camera
compatibility. We should not copy the architecture of doing substantial dynamic
runtime work directly in PAM.

### linux-enable-ir-emitter

This active Rust project discovers vendor UVC controls that enable IR emitters
and can pass an already-open camera file descriptor. It is a strong integration
candidate. Configuration probing can be risky and belongs in explicit enrollment
or setup, never in the authentication hot path.

Repository: <https://github.com/EmixamPP/linux-enable-ir-emitter> (MIT). It is a
candidate adapter, not an identity or liveness implementation.

### fprintd and libfprint

They provide a mature D-Bus and PAM path, but their API and device model are
fingerprint-specific. They are useful architectural references, not a generic
face-biometric backend.

### KDE Plasma

KScreenLocker 6.7 ships PAM definitions for password, fingerprint, and smart
card. Plasma has no equivalent generic face enrollment service or KCM. A proper
integration therefore needs both a PAM backend and modality-aware lock-screen
status, with a desktop-independent daemon beneath the KDE frontend.

KDE's existing modality-specific PAM services are useful integration points, but
they do not provide a generic face enrollment database or face-authentication
KCM. The first frontend should therefore be optional and consume the same
desktop-independent D-Bus API as non-KDE clients.

## Proposed reusable stack

- V4L2 and media-controller APIs for capture and stable device identity;
- ONNX Runtime for audited pretrained detector, landmark, embedding, and PAD models;
- OpenCV only where its mature image operations materially reduce risk;
- TPM2-TSS for machine-bound template encryption keys;
- Linux PAM for local authentication and Polkit for enrollment authorization;
- system D-Bus for management/status and a narrow peer-credential local transport
  for the PAM authentication transaction;
- KDE KI18n/KCMUtils for the optional System Settings frontend.

## Windows Hello comparison boundary

Windows Hello combines calibrated IR hardware, anti-spoofing, protected template
storage, secure key release, and vendor certification. Commodity Linux webcams
and community models do not automatically provide those guarantees. This project
can reproduce the workflow and improve Linux integration, but must describe its
assurance level precisely and keep password/FIDO recovery available.
