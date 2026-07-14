# Template and key storage

`faceauth-storage` persists only derived embeddings and model metadata. Raw IR or
visible-light frames are not representable in the stored template schema.

## Template encryption

- XChaCha20-Poly1305 provides confidentiality and integrity.
- A fresh 192-bit nonce is generated for every write.
- Format version and numeric UID are authenticated as associated data.
- Files are written with mode `0600`, synced, and atomically renamed.
- Symbolic links and group/other-readable template or key files are rejected.
- Secret keys, decrypted JSON, and embedding vectors are zeroized on drop.
- Record dimensions, finite floating-point values, model hashes, and file sizes
  are bounded before use.

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

The TPM path can be tested without enrolling a face:

```bash
sudo faceauth-daemon storage-doctor
```

This uses transient TPM objects. It does not allocate a persistent handle, write
TPM NV storage, modify PCRs, or change firmware state.
