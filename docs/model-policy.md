# Model acceptance policy

No model weights are bundled or downloaded merely because they are technically
compatible with ONNX Runtime. Every artifact must have a schema-v2 reviewed
manifest with an HTTPS provenance URL, valid SPDX license expression, exact
SHA-256 digest, pipeline role, and exact static input and output contracts.

The initial runtime admits exactly one fixed-shape float32 input and a bounded,
non-empty list of fixed-shape float32 outputs. Tensor names, ranks, dimensions,
and element counts are bounded. Dynamic dimensions, unexpected graph inputs or
outputs, duplicate outputs, and name/type/shape mismatches fail closed.

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
optimization level 2. Runtime and model packaging remains a distribution/admin
responsibility; the source tree intentionally contains no model weights or copied
ONNX Runtime binary.

Landmark output used by active liveness is an input to the independently tested
challenge state machine, not a terminal authentication decision. Eye-openness
and yaw thresholds must be explicitly configured from calibration evidence; the
active-liveness crate intentionally provides no production defaults.
