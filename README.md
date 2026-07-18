# faceauth-rs

Linux-native face enrollment and authentication, designed for IR and visible
light cameras with a Rust-first security boundary.

## Status

This repository is an early security and architecture scaffold. It does not
install a PAM module, change login configuration, or claim Windows Hello
equivalence. Password fallback is mandatory.

The initial target hardware is a ThinkPad Z13 Gen 1 with:

- Chicony `04f2:b769` IR camera: 640x360 GREY at 15 fps;
- Chicony `04f2:b768` visible camera: 640x360 at 30 fps or higher;
- TPM 2.0 resource manager at `/dev/tpmrm0`;
- optional `thinkpad-hpd` presence hints over system D-Bus.

HPD can prewarm capture or improve desktop feedback. It is never accepted as
identity or liveness evidence.

## Intended architecture

- A privileged daemon owns cameras, inference, liveness, and encrypted templates.
- A small PAM module exchanges nonce-bound messages with the daemon.
- Enrollment requires existing authorization through Polkit or PAM.
- KDE System Settings provides a thin `Security & Privacy → Face Authentication` client over
  desktop-independent service APIs.
- Raw enrollment frames are ephemeral and are not retained by default.
- Face authentication never removes the password path.

See [architecture](docs/architecture.md), [threat model](docs/threat-model.md),
[IPC boundary](docs/ipc.md), [model policy](docs/model-policy.md),
[template storage](docs/storage.md), [camera capture](docs/capture.md),
[active liveness](docs/liveness.md), [enrollment](docs/enrollment.md), and
[presence integration](docs/presence.md), [PAM boundary](docs/pam.md), and
[management contract](docs/management.md), [KDE experience](docs/kde-ui.md),
[deployment artifacts](docs/deployment.md), and
[production readiness](docs/production-readiness.md), and [research](docs/research.md).

## Development

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo deny check
cargo audit
```

Read-only local diagnostics:

```bash
cargo run -p faceauth-daemon -- doctor --config /etc/faceauth/faceauth.json
cargo run -p faceauth-daemon -- capture-doctor --config ./cameras.json
cargo run -p faceauth-daemon -- presence-doctor
cargo run -p faceauth-cli -- policy
```

## License

Licensed under either Apache-2.0 or MIT, at your option.
