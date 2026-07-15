# Model acceptance policy

No model weights are bundled or downloaded merely because they are technically
compatible with ONNX Runtime. Every artifact must have a schema-v4 reviewed
manifest with an HTTPS provenance URL, valid SPDX license expression, exact
SHA-256 digest, pipeline role, and exact static input and output contracts.

The initial runtime admits exactly one fixed-shape float32 input and a bounded,
non-empty list of fixed-shape float32 outputs. Tensor names, ranks, dimensions,
and element counts are bounded. Dynamic dimensions, unexpected graph inputs or
outputs, duplicate outputs, and name/type/shape mismatches fail closed.
Schema v4 binds the resize filter, finite per-channel affine normalization
(`pixel * scale + bias`), and security-relevant output semantics to the model and
its calibration record. Embedding roles require exactly one bounded embedding
output; passive-PAD roles require exactly one scalar live-probability output.

## Intended pipeline

1. Detect one face and reject zero-face or multi-face frames.
2. Estimate landmarks and align both IR and visible observations consistently.
3. Run image-quality gates before recognition or liveness scoring.
4. Produce an embedding with a mature ArcFace-family model or equivalent whose
   training provenance and redistribution terms are acceptable.
5. Run passive presentation-attack detection on IR and visible inputs, preferably
   with a model designed for multi-modal evidence.
6. Run a randomized active challenge and verify temporal response over multiple
   fresh paired frames.
7. Apply thresholds calibrated on the exact hardware and model versions. Record
   false-accept and false-reject measurements; do not inherit demonstration
   thresholds from upstream sample code.

## Required review evidence

- immutable upstream release or commit and model-file hash;
- model-weight license, not only the source-code license;
- training dataset provenance and biometric-use constraints;
- exact preprocessing, color order, layout, normalization, and alignment method;
- CPU/GPU latency and memory behavior on the target machine;
- demographic and lighting evaluation notes;
- print, screen replay, video replay, mask, occlusion, and look-alike tests;
- behavior when one modality is missing, stale, duplicated, or desynchronized.

Models remain data, never executable plugins. ONNX Runtime sessions must use a
restricted operator set where practical, fixed tensor bounds, bounded execution
time, and no network access. Updating any model invalidates previous calibration
and requires an explicit template compatibility decision.

`faceauth-inference` loads ONNX Runtime dynamically from an explicitly configured
path; it does not download a runtime or model. The runtime library, model, and
every containing directory must be root-owned and not group/world writable. The
model is canonicalized, permission-checked, size-bounded, and hash-verified before
session creation. CPU sessions use sequential graph execution, one inter-op
thread, a bounded intra-op thread count, memory-pattern planning, and ONNX Runtime
optimization level 2. Every call uses a bounded watchdog that requests ONNX Runtime
termination at its configured deadline; stronger hard isolation remains future
worker-process work. Runtime and model packaging remains a distribution/admin
responsibility; the source tree intentionally contains no model weights or copied
ONNX Runtime binary.

Preprocessing accepts only tightly packed, dimension-bounded Gray8, RGB8, BGR8,
or YUYV bytes. It uses schema-bound bilinear half-pixel resizing, explicit channel
order, and affine normalization. Input and copied output float buffers are
zeroized on drop, and non-finite runtime outputs fail closed. MJPEG must be decoded
by a separately bounded decoder before this boundary and is not accepted directly.

The authentication worker uses cancellable preprocessing. After validating the
source contract, it checks cancellation before tensor allocation and before each
output row of resize/color conversion. Cancellation drops the partially populated
zeroizing tensor. A false cancellation callback produces byte-for-byte identical
float output to the regular deterministic path. ONNX graph execution remains
bounded separately by per-run `RunOptions` termination and the hard watchdog;
worker cancellation must be checked again before starting a graph call.

The embedding adapter rejects degenerate vectors and L2-normalizes accepted model
outputs before enrollment or comparison. Cosine similarity is mapped from
`[-1, 1]` to `[0, 1]` for the core policy. Persisted templates are accepted only
when already unit-normalized and when their complete manifest compatibility digest
and dimension exactly match the active embedding session. Passive-PAD values are
not inferred from arbitrary tensors: only a schema-tagged scalar probability in
`[0, 1]` is admitted, and thresholds must come from explicit calibration.
At daemon orchestration, both the IR and visible PAD roles are mandatory and each
score must match the configured full model/preprocessing compatibility digest.
Exactly one score per required role is accepted. A configured fusion role is
mandatory; an unconfigured fusion score is rejected rather than silently changing
the evidence policy. Passing one modality can never compensate for failure in the
other.

Landmark output used by active liveness is an input to the independently tested
challenge state machine, not a terminal authentication decision. Eye-openness
and yaw thresholds must be explicitly configured from calibration evidence; the
active-liveness crate intentionally provides no production defaults.
