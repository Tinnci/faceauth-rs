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

The daemon enrollment service is single-capacity and shares the global camera/model arbiter with
authentication. Submission is non-queueing and bound to the exact Manager1 operation ID, root
broker authorization, target UID, private cancellation signal, and daemon shutdown token. The
engine owns capture and inference and may return only a validated derived `TemplateRecord`.

The service rechecks the record UID and structure, then commits it through the authenticated atomic
template sink. Only after storage succeeds does the worker channel emit an empty successful
completion. Cancellation and storage failure can therefore never be presented as enrolled. Worker
updates contain only closed management progress or sanitized failure categories; raw frames,
landmarks, tensors, embeddings, and template contents have no channel representation.
