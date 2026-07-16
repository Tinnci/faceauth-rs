# Production configuration and readiness

`faceauth-daemon` uses one strict, versioned configuration for production composition. The schema
rejects unknown fields, unsupported versions, relative security-sensitive paths, incomplete model
roles, unbounded resources, passive-only liveness, missing visible capture, and disabled password
fallback. The starting point is
[`faceauth.json.example`](../contrib/config/faceauth.json.example); every placeholder digest,
executable fingerprint, path, selector, threshold, and review artifact must be replaced with
evidence from the exact packaged build and target hardware.

The production configuration and every referenced runtime, manifest, model, and review artifact
must be root-owned, must not be writable by group or others, and must live below equally trusted
directories. Readiness follows symlinks to their canonical targets and validates the complete target
ancestor chain. Model manifests and artifacts are hash-checked without loading ONNX Runtime or
executing a graph. Review reports are bounded and bound by lowercase SHA-256.

Run the read-only report with:

```console
sudo faceauth-daemon doctor --config /etc/faceauth/faceauth.json
```

The report has stable schema version `1` and ten independent gates:

1. configuration schema and invariants;
2. unique exact IR/RGB hardware resolution;
3. trusted ONNX Runtime installation;
4. the complete six-role reviewed and hash-verified model suite;
5. model-bound recognition, passive PAD, active-liveness, and enrollment calibration evidence;
6. TPM-bound storage and restart/recovery evidence;
7. executable-bound screen-unlock, login, and Polkit authorization review;
8. stable Manager1 and PolicyKit lifecycle review;
9. root-owned local socket plus isolated password-fallback/recovery evidence;
10. complete supervised daemon service composition.

`production_ready` is true only if every gate is true. A missing or untrusted configuration makes
all dependent gates explicitly unevaluated. A root-only file key is retained for diagnostics, but
it can never satisfy the production storage gate.

The current build deliberately reports the service-composition gate as false: capture, inference,
authentication, enrollment, Manager1, cancellation, and shutdown supervision are not yet wired into
one audited production runner. Consequently `faceauth-daemon serve` still refuses startup even if
all external files appear complete. This prevents configuration from claiming capabilities the
binary does not yet implement.

Neither `doctor` nor readiness inspection opens cameras, creates TPM keys, binds the authentication
socket, claims a D-Bus name, installs policy, enables a service, or changes PAM configuration.
