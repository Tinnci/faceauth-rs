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
- the minimum mapped cosine similarity between every sample pair.

Each accepted observation must be fresh and strictly newer than the prior one,
inside the original deadline, from the same complete model compatibility digest
and embedding dimension, and have passed both passive and randomized active
liveness. Low-quality observations may be discarded and retried. A stale or
duplicated observation, liveness failure, model switch, dimension switch, sample
overflow, invalid arithmetic, or inconsistent identity permanently fails that
transaction.

## Template production

Completion requires the configured minimum sample count before the fixed deadline.
The state machine averages only the accepted normalized embeddings, rejects a
non-finite or degenerate aggregate, normalizes the aggregate again, and emits a
template-format-v2 record. The record contains only numeric UID, complete model
compatibility digest, and the derived embedding. All temporary sample and
aggregate buffers are zeroized on drop.

Thresholds in tests demonstrate invariants only. Production values require
evaluation on the exact camera, model suite, preprocessing contract, lighting,
and target population.
