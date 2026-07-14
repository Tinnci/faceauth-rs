# Contributing

All changes must preserve password fallback, avoid retaining raw biometric
frames, and document new model provenance and licenses.

Before committing:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo deny check
cargo audit
```

Unsafe Rust is forbidden by workspace lint. Any future PAM FFI must live in a
small isolated crate with a documented exception and dedicated tests.

