# Enrollment transaction

`faceauth-enrollment` is a bounded daemon-side state machine. It never accepts,
stores, serializes, or returns camera frames. Its only biometric input is a
zeroizing, L2-normalized `FaceEmbedding` produced after the admitted capture,
quality, passive-PAD, and active-liveness pipeline.

## Authorization boundary

The daemon starts enrollment only from an existing exact authorization grant for
the dedicated `faceauth-enroll` service, the `polkit` purpose, and a root broker
peer. Caller identity and executable identity have already been derived from
kernel peer credentials and pidfd-backed executable verification. A generic PAM
authentication grant or an unprivileged desktop client cannot be reused to start
enrollment.

This is the internal boundary; a future Polkit action and KDE KCM must remain thin
clients of it and must not receive template or raw-frame data.

## Evidence and bounds

One explicitly calibrated configuration fixes:

- a 5–120 second monotonic transaction duration;
- 3–16 required/maximum derived samples;
- the minimum image-quality score;
- the minimum mapped cosine similarity between every sample pair;
- a minimum monotonic interval between retained samples;
- a minimum model-derived yaw span across the accepted set.

Each accepted observation must be fresh, sufficiently separated from the prior one,
inside the original deadline, from the same complete model compatibility digest
and embedding dimension, and have passed both passive and randomized active
liveness. Low-quality observations may be discarded and retried. A stale or
duplicated observation, liveness failure, model switch, dimension switch, sample
overflow, invalid pose, invalid arithmetic, or inconsistent identity permanently fails that
transaction.

## Template production

Completion requires both the configured minimum sample count and calibrated left-to-right pose
coverage before the fixed deadline. This prevents a burst of nearly identical frontal frames from
masquerading as a robust enrollment. The state machine computes a quality-weighted centroid from
only the accepted normalized embeddings, so marginal but admissible frames exert less influence
than clear observations. It rejects a
non-finite or degenerate aggregate, normalizes the aggregate again, and emits a
template-format-v2 record. The record contains only numeric UID, complete model
compatibility digest, and the derived embedding. All temporary sample and
aggregate buffers are zeroized on drop.

Thresholds in tests demonstrate invariants only. Production values require
evaluation on the exact camera, model suite, preprocessing contract, lighting,
and target population.

## Worker and commit boundary

The daemon enrollment service is single-capacity and shares both the global camera/model arbiter
and the exact `ProductionObservationPipeline` instance with authentication. That pipeline is the
unique owner of the configured IR/RGB devices and all six mutable ONNX sessions: detector,
landmarks, embedding, IR PAD, visible PAD, and named-input fusion PAD. Submission is non-queueing
and bound to the exact Manager1 operation ID, root broker authorization, target UID, private
cancellation signal, and daemon shutdown token.

One operation-bound randomized challenge must pass before any enrollment sample is admitted. The
same paired frames then feed detection, landmarks, quality, aligned embedding, and all three PAD
paths. Later samples require new IR and visible sequence values and a strictly newer paired
timestamp; frames closer than the calibrated sample interval are discarded without poisoning the
enrollment state. Low-quality observations may be retried, while PAD failure, model mismatch,
identity inconsistency, stale evidence, or exhausted bounded attempts fail closed. The engine may
return only a validated derived `TemplateRecord`.

The service rechecks the record UID and structure, then commits it through the authenticated atomic
template sink. Only after storage succeeds does the worker channel emit an empty successful
completion. Cancellation and storage failure can therefore never be presented as enrolled. Worker
updates contain only closed management progress or sanitized failure categories; raw frames,
landmarks, tensors, embeddings, and template contents have no channel representation.

The Manager1 adapter invokes a dependency-inverted operation controller rather than depending on
daemon internals. The daemon bridge submits the authorized job, exposes its update handle exactly
once to the service coordinator, preserves private cancellation ownership, and clears it only after
the exact terminal update has been relayed.

The relay passes every worker update back through `ManagementWorkerHandle`, which revalidates the
target UID, operation ID, and deadline before a signal sink can emit it. Worker failures collapse
to `failed`; cancellation maps to `cancelled`. If the public operation was already consumed by a
client cancellation, a late worker terminal performs private cleanup only and is never broadcast a
second time.
