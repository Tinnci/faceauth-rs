# Production configuration and readiness

`faceauth-daemon` uses strict production-configuration schema 4 for production composition. The
machine-readable readiness report remains schema 1. The configuration schema
rejects unknown fields, unsupported versions, relative security-sensitive paths, incomplete model
roles, unbounded resources, passive-only liveness, missing visible capture, and disabled password
fallback. It also bounds supervisor polling, service count, and graceful shutdown time. Schema 4
binds each camera selector to an exact V4L2 width, height, pixel format, frame rate, buffer
count, warmup count, per-frame timeout, maximum byte length, and IR/RGB pairing budget. Negotiation
drift or an invalid modality/format combination prevents production construction. The starting
point is
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
5. model-bound recognition, quality, passive PAD, active-liveness, and enrollment calibration
   evidence;
6. TPM-bound storage and restart/recovery evidence;
7. executable-bound screen-unlock, login, and Polkit authorization review;
8. stable Manager1 and PolicyKit lifecycle review;
9. root-owned local socket plus isolated password-fallback/recovery evidence;
10. complete supervised daemon service composition.

`production_ready` is true only if every gate is true. A missing or untrusted configuration makes
all dependent gates explicitly unevaluated. A root-only file key is retained for diagnostics, but
it can never satisfy the production storage gate.

The current build deliberately reports the service-composition gate as false: capture, inference,
authentication, enrollment, and Manager1 are not yet wired into one audited production runner. The
single-capacity authentication engine and shutdown supervision boundaries now exist, but no
complete runner yet combines the now-supervised authentication composition with the enrollment
engine and Manager1 under one supervisor. Consequently
`faceauth-daemon serve` still refuses startup even if all external files appear complete. This
prevents configuration from claiming capabilities the binary does not yet compose and verify.

Neither `doctor` nor readiness inspection opens cameras, creates TPM keys, binds the authentication
socket, claims a D-Bus name, installs policy, enables a service, or changes PAM configuration.
The explicit production authentication builder is separate: it first revalidates trusted model and
calibration evidence, then resolves and negotiates both configured V4L2 streams, loads all six
role-bound ONNX sessions, verifies their complete compatibility digests, and only then returns the
single-worker engine.
Separate explicit builders self-test the TPM-unsealed machine key before returning an encrypted
template store, and bind the authentication socket without unlinking any existing path before
returning its exact authorization, framing, and transaction policies. Diagnostic file keys cannot
enter this production construction path.
The authentication-side composition now runs the V4L2/ONNX engine worker and secure listener under
one fail-fast supervisor; a connection-coordination failure requests global shutdown and is
reported as a service failure. This partial runner is not the main `serve` entry point and does not
weaken its blocked readiness gate.
