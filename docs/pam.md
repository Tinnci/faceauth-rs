# PAM authentication boundary

`pam-faceauth` is a deliberately small PAM authentication bridge. At this stage it is restricted
to the dedicated `faceauth-test` PAM service and is not ready to be added to login, SDDM, screen
unlock, `sudo`, Polkit, or any other production authentication stack.

## Security contract

- The module accepts no module arguments and returns `PAM_IGNORE` if any are supplied.
- It runs only when `PAM_SERVICE` is exactly `faceauth-test`.
- It resolves the PAM user to a numeric UID with bounded, reentrant libc lookup.
- It connects only to `/run/faceauth/auth.sock` and requires the socket peer UID to be root.
- It sends the versioned faceauth protocol with `AuthenticationPurpose::Test`.
- It accepts only a matching transaction's terminal `DecisionCode::Accepted` response.
- The exchange is bounded to 64 responses and 20 seconds.
- Panics, malformed data, timeouts, unavailable services, rejections, and mismatches all become
  `PAM_IGNORE`; the surrounding PAM policy must retain a password path.
- The module never opens cameras, loads models, reads biometric templates, or requests passwords.

The module contains a narrow `unsafe` boundary for the libpam and libc ABIs. Unsafe code remains
forbidden across the rest of the workspace. Changes to the FFI declarations, pointer lifetimes,
PAM return mapping, or exported symbols require focused review.

## Build and package

```bash
cargo build -p pam-faceauth --release
nm -D --defined-only target/release/libpam_faceauth.so \
  | grep -E 'pam_sm_(authenticate|setcred)'
```

The Rust artifact is named `libpam_faceauth.so`. A distribution package must install it in that
distribution's PAM module directory under the conventional name `pam_faceauth.so`, owned by root
and not writable by unprivileged users. Packaging must not automatically edit any PAM service.

## Isolated test policy

[`contrib/pam/faceauth-test`](../contrib/pam/faceauth-test) is the only supplied PAM policy:

```text
auth    [success=done ignore=ignore default=bad] pam_faceauth.so
auth    required                               pam_unix.so
account required                               pam_unix.so
```

This policy preserves `pam_unix` fallback when face authentication is unavailable or declined.
Test it only in an isolated PAM configuration location supported by the target distribution or in
a disposable container/VM. Do not copy it over an existing system policy. Before any production
integration, add an end-to-end libpam harness, package ownership checks, daemon mediation tests,
distribution-specific policy review, and recovery testing for a stopped or compromised daemon.

