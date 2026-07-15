# Deployment artifacts and activation gate

The repository contains packaging inputs under `contrib/`, but they are not installed or enabled
by the build, tests, or CI. No command in this repository edits `/etc/pam.d`, enables a systemd
unit, installs a Polkit action, or changes KDE/SDDM/lock-screen policy.

## Polkit enrollment action

[`org.faceauth.policy`](../contrib/polkit/org.faceauth.policy) defines only the
`org.faceauth.enroll` administrative action. A future desktop broker must use this action to
authorize enrollment and then create the exact root-owned `faceauth-enroll` / `Polkit` grant checked
by `begin_authorized_enrollment`. The action is not an authentication decision and cannot authorize
screen unlock, login, sudo, or PAM.
The policy deliberately requires a fresh administrator authorization rather than a cached
`auth_admin_keep` grant.

Packaging review must verify the action's installed ownership and XML validity, bind the broker's
executable fingerprint into the daemon authorization policy, and preserve password fallback. An
administrator must explicitly approve installation and activation.

The `faceauth-management-dbus` crate now provides the asynchronous system-bus credential resolver
and PolicyKit `CheckAuthorization` client for `org.faceauth.enroll`. A production daemon must still
provide the authenticated encrypted-template state source and exact root grant issuer, then wire
sender-disconnect cancellation. Until those implementations, D-Bus policy, and isolated
integration tests exist, packaging must not claim `org.faceauth.Manager1` or activate it on the
system bus.

## systemd service template

[`faceauth.service`](../contrib/systemd/faceauth.service) is a hardened template for a future
package. It runs as root, restricts network families to Unix sockets, uses private state/runtime
directories, and makes no assumption that a TPM or model suite is present. The current `serve`
command intentionally refuses to start because the audited production model set and full capture
pipeline are incomplete; installing this unit does not change that gate.

Before activation, a distribution package must review device access for the selected IR/RGB nodes,
TPM resource manager, root-owned model/runtime paths, service readiness behavior, socket group
ownership, shutdown/restart recovery, and log privacy. Do not enable the unit on a production login
system until isolated PAM, Polkit, camera-loss, daemon-restart, and password-recovery tests pass.

## Explicit activation checklist

1. Install reviewed model manifests, model files, and the trusted ONNX Runtime library with root-only
   ownership and permissions.
2. Install the service and action through the distribution package manager; never copy files into
   `/etc` manually during development.
3. Validate the camera selector configuration and encrypted template storage policy.
4. Run the read-only `doctor`, capture, storage, and presence diagnostics.
5. Test the dedicated `faceauth-test` PAM policy in an isolated environment.
6. Obtain an explicit administrator decision before enabling any login, locker, sudo, or Polkit
   authentication integration.
