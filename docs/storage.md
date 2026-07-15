# Template and key storage

`faceauth-storage` persists only derived embeddings and model metadata. Raw IR or
visible-light frames are not representable in the stored template schema.

## Template encryption

- XChaCha20-Poly1305 provides confidentiality and integrity.
- A fresh 192-bit nonce is generated for every write.
- Format version and numeric UID are authenticated as associated data.
- Production directories, template files, file keys, TPM blobs, and every checked
  ancestor must be owned by UID 0. Directories must not be group/world writable;
  secret files must have no group/other permission bits.
- Files are opened with `O_NOFOLLOW|O_CLOEXEC`, required to be regular files,
  written with mode `0600`, synced, and atomically renamed.
- Symbolic links, unsafe filesystem types, wrong owners, unsafe replacement
  targets, and group/other-readable template or key files are rejected.
- Secret keys, decrypted JSON, and embedding vectors are zeroized on drop.
- Record dimensions, finite floating-point values, unit embedding norm, complete
  model-manifest compatibility digests, and file sizes are bounded before use.

Template format version 2 stores a compatibility SHA-256 over the validated
embedding manifest, not merely the ONNX file hash. This binds the role, model hash,
input preprocessing, exact tensor contracts, and semantic output tag. Loading for
comparison additionally requires the active digest and embedding dimension to
match exactly; incompatible templates fail closed and must be explicitly
re-enrolled rather than silently migrated or renormalized.

`has_authenticated_template(uid)` reports enrollment state only after the same bounded file checks,
key retrieval, AEAD authentication, JSON decoding, UID binding, and record validation as a normal
load. Missing storage returns `false`; corrupt, tampered, or untrusted storage remains an error. It
does not reduce enrollment state to file existence.

## TPM backend

The native Rust TPM backend uses `tss-esapi` and `/dev/tpmrm0`. It creates a
deterministic owner-hierarchy primary key and stores a TPM-created public/private
blob containing a sealed random 256-bit template-encryption key. Each load
recreates the primary, loads the sealed object, and unseals the key.

The current sealed object is machine-bound but intentionally has no PCR policy.
This avoids destroying enrollment after routine kernel, firmware, rEFInd, or ACPI
updates. It is not equivalent to Windows Hello Enhanced Sign-in Security. A
future optional measured-boot policy needs recovery-key design and explicit
migration tests before it can be enabled.

The file-key provider is an explicit fallback and reports `root-only-file`
strength. Production enrollment must prefer `tpm-bound` and visibly warn or
refuse when policy does not allow the fallback.

The store never treats a restrictive mode alone as proof of trust: a `0600` file
owned by another UID is rejected. Before creating a missing storage directory it
verifies the nearest existing ancestor, creates the final directory as `0700`,
then rechecks ownership, type, and permissions. Atomic replacement is allowed only
inside the verified directory and will not overwrite a symlink, directory,
device, wrong-owner file, or otherwise unsafe existing target.

The TPM path can be tested without enrolling a face:

```bash
sudo faceauth-daemon storage-doctor
```

This uses transient TPM objects. It does not allocate a persistent handle, write
TPM NV storage, modify PCRs, or change firmware state.
