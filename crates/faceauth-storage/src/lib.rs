//! Encrypted, root-owned face-template storage primitives.
//!
//! This crate deliberately does not decide whether a key is backed by a TPM. The daemon selects
//! a [`KeyProvider`] and must surface its [`KeyStrength`] to administrators before enrollment.

#[cfg(feature = "tpm")]
mod tpm;

#[cfg(feature = "tpm")]
pub use tpm::TpmKeyProvider;

use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
};

use chacha20poly1305::{
    XChaCha20Poly1305, XNonce,
    aead::{Aead, KeyInit, Payload},
};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;
use zeroize::{Zeroize, ZeroizeOnDrop, Zeroizing};

const MAGIC: &[u8; 4] = b"FAT1";
/// Current encrypted template record schema and container version.
pub const TEMPLATE_RECORD_SCHEMA_VERSION: u16 = 2;

const FORMAT_VERSION: u16 = TEMPLATE_RECORD_SCHEMA_VERSION;
const NONCE_LENGTH: usize = 24;
const HEADER_LENGTH: usize = MAGIC.len() + std::mem::size_of::<u16>() + NONCE_LENGTH;
const MAX_CIPHERTEXT_LENGTH: usize = 1024 * 1024;
const MAX_EMBEDDING_DIMENSION: usize = 4096;
const KEY_LENGTH: usize = 32;
const STORED_PAYLOAD_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Copy, Debug)]
pub(crate) struct TrustPolicy {
    owner_uid: u32,
    verify_ancestors: bool,
}

impl TrustPolicy {
    const fn root() -> Self {
        Self { owner_uid: 0, verify_ancestors: true }
    }

    #[cfg(test)]
    const fn test(owner_uid: u32) -> Self {
        Self { owner_uid, verify_ancestors: false }
    }
}

/// Strength of the key protection selected by the daemon.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum KeyStrength {
    /// Key is sealed or otherwise released by a TPM-backed implementation.
    TpmBound,
    /// Key is stored in a root-readable file with owner-only permissions.
    RootOnlyFile,
}

/// Secret key held only for the duration of one storage operation.
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct SecretKey([u8; KEY_LENGTH]);

impl SecretKey {
    /// Construct a key from exactly 32 bytes.
    ///
    /// # Errors
    ///
    /// Returns [`KeyError::InvalidLength`] when the input does not contain exactly 32 bytes.
    pub fn from_slice(bytes: &[u8]) -> Result<Self, KeyError> {
        let key = bytes
            .try_into()
            .map_err(|_| KeyError::InvalidLength { expected: KEY_LENGTH, actual: bytes.len() })?;
        Ok(Self(key))
    }

    const fn as_bytes(&self) -> &[u8; KEY_LENGTH] {
        &self.0
    }

    fn same_secret(&self, other: &Self) -> bool {
        self.0
            .iter()
            .zip(other.0.iter())
            .fold(0_u8, |difference, (left, right)| difference | (left ^ right))
            == 0
    }
}

/// Key source used by encrypted template storage.
pub trait KeyProvider {
    /// Load the machine key for storage operations.
    ///
    /// # Errors
    ///
    /// Returns [`KeyError`] when the key cannot be loaded or its protection is unsafe.
    fn load_key(&self) -> Result<SecretKey, KeyError>;

    /// Report the protection strength to diagnostics and enrollment policy.
    fn strength(&self) -> KeyStrength;
}

/// Explicit root-only file-key fallback for development and systems without a TPM backend.
#[derive(Clone, Debug)]
pub struct FileKeyProvider {
    path: PathBuf,
    trust: TrustPolicy,
}

impl FileKeyProvider {
    /// Use a specific key path. The parent directory must already be administrator-controlled.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into(), trust: TrustPolicy::root() }
    }

    /// Return the configured key path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl KeyProvider for FileKeyProvider {
    fn load_key(&self) -> Result<SecretKey, KeyError> {
        let parent = self.path.parent().ok_or_else(|| KeyError::InvalidPath(self.path.clone()))?;
        verify_directory_key(parent, self.trust)?;
        verify_replace_target_key(&self.path, self.trust)?;
        match open_read_nofollow(&self.path) {
            Ok(file) => read_key_file(file, &self.path, self.trust),
            Err(error) if error.kind() == io::ErrorKind::NotFound => self.create_key(),
            Err(error) => Err(KeyError::Io(error)),
        }
    }

    fn strength(&self) -> KeyStrength {
        KeyStrength::RootOnlyFile
    }
}

impl FileKeyProvider {
    #[cfg(test)]
    fn for_test(path: impl Into<PathBuf>, owner_uid: u32) -> Self {
        Self { path: path.into(), trust: TrustPolicy::test(owner_uid) }
    }

    fn create_key(&self) -> Result<SecretKey, KeyError> {
        let mut bytes = [0_u8; KEY_LENGTH];
        getrandom::fill(&mut bytes).map_err(|error| KeyError::Random(error.to_string()))?;
        let mut file = match OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(&self.path)
        {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                bytes.zeroize();
                return self.load_key();
            }
            Err(error) => {
                bytes.zeroize();
                return Err(KeyError::Io(error));
            }
        };
        let result: Result<SecretKey, KeyError> = (|| {
            verify_open_key_file(&file, &self.path, self.trust)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
            SecretKey::from_slice(&bytes)
        })();
        bytes.zeroize();
        result
    }
}

fn read_key_file(mut file: File, path: &Path, trust: TrustPolicy) -> Result<SecretKey, KeyError> {
    verify_open_key_file(&file, path, trust)?;
    let mut bytes = Vec::with_capacity(KEY_LENGTH);
    file.read_to_end(&mut bytes)?;
    SecretKey::from_slice(&bytes)
}

/// One encrypted template payload. Raw camera frames are intentionally not represented here.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, Zeroize, ZeroizeOnDrop)]
#[serde(deny_unknown_fields)]
pub struct TemplateRecord {
    /// Record schema version.
    pub schema_version: u16,
    /// Account UID this record belongs to.
    pub uid: u32,
    /// Lowercase SHA-256 compatibility digest of the complete embedding manifest.
    pub model_compatibility_sha256: String,
    /// Derived face embedding; never a raw image.
    pub embedding: Vec<f32>,
}

/// Authenticated template snapshot together with its monotonic storage generation.
///
/// Generation zero is reserved for records written by storage versions predating explicit
/// generation tracking. Every mutation performed by this implementation writes a strictly
/// positive, monotonically increasing generation.
#[derive(Clone, Debug, PartialEq)]
pub struct VersionedTemplateRecord {
    /// Generation read from the same authenticated payload as the template.
    pub generation: u64,
    /// Validated biometric template.
    pub record: TemplateRecord,
}

#[derive(Deserialize, Serialize, Zeroize, ZeroizeOnDrop)]
#[serde(tag = "state", rename_all = "kebab-case", deny_unknown_fields)]
enum StoredPayload {
    Live { schema_version: u16, generation: u64, record: TemplateRecord },
    Deleted { schema_version: u16, generation: u64, uid: u32 },
}

impl StoredPayload {
    const fn generation(&self) -> u64 {
        match self {
            Self::Live { generation, .. } | Self::Deleted { generation, .. } => *generation,
        }
    }
}

impl TemplateRecord {
    /// Validate record bounds and finite numeric data before encryption.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::InvalidRecord`] when the schema, model digest, dimension, or
    /// embedding values are invalid.
    pub fn validate(&self) -> Result<(), StorageError> {
        if self.schema_version != FORMAT_VERSION
            || self.model_compatibility_sha256.len() != 64
            || !self.model_compatibility_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
            || self.model_compatibility_sha256.bytes().any(|byte| byte.is_ascii_uppercase())
            || self.embedding.is_empty()
            || self.embedding.len() > MAX_EMBEDDING_DIMENSION
            || self.embedding.iter().any(|value| !value.is_finite())
        {
            return Err(StorageError::InvalidRecord("template bounds or digest are invalid"));
        }
        let squared_norm = self.embedding.iter().try_fold(0.0_f32, |sum, value| {
            let next = value.mul_add(*value, sum);
            next.is_finite().then_some(next)
        });
        if squared_norm.is_none_or(|value| (value.sqrt() - 1.0).abs() > 1.0e-3) {
            return Err(StorageError::InvalidRecord("template embedding is not normalized"));
        }
        Ok(())
    }

    /// Require this template to match an active embedding manifest and dimension.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::IncompatibleTemplate`] when the compatibility digest or embedding
    /// dimension differs.
    pub fn require_compatibility(
        &self,
        compatibility_sha256: &str,
        dimension: usize,
    ) -> Result<(), StorageError> {
        self.validate()?;
        if self.model_compatibility_sha256 != compatibility_sha256
            || self.embedding.len() != dimension
        {
            return Err(StorageError::IncompatibleTemplate);
        }
        Ok(())
    }
}

/// Encrypted template store using one administrator-selected key provider.
#[derive(Debug)]
pub struct EncryptedTemplateStore<K> {
    directory: PathBuf,
    key_provider: K,
    trust: TrustPolicy,
}

impl<K> EncryptedTemplateStore<K>
where
    K: KeyProvider,
{
    /// Create a store rooted at an administrator-controlled directory.
    #[must_use]
    pub fn new(directory: impl Into<PathBuf>, key_provider: K) -> Self {
        Self { directory: directory.into(), key_provider, trust: TrustPolicy::root() }
    }

    /// Report the selected key protection strength.
    #[must_use]
    pub fn key_strength(&self) -> KeyStrength {
        self.key_provider.strength()
    }

    /// Save a record using authenticated encryption and atomic replacement.
    ///
    /// This compatibility entry point performs an unconditional, serialized update. Callers that
    /// read before writing should use [`Self::replace_if_generation`] so a stale observation cannot
    /// overwrite a newer template.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] when validation, key loading, encryption, or the atomic write
    /// fails.
    pub fn save(&self, record: &TemplateRecord) -> Result<(), StorageError> {
        self.save_with_generation(record).map(|_| ())
    }

    /// Save a record unconditionally and return its new authenticated generation.
    ///
    /// Mutations for one UID are serialized with an owner-only, no-follow lock file. Existing
    /// legacy records begin at generation zero; every successful save increments the generation.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] for unsafe paths, invalid or corrupt current state, generation
    /// exhaustion, key/encryption failure, or atomic persistence failure.
    pub fn save_with_generation(&self, record: &TemplateRecord) -> Result<u64, StorageError> {
        record.validate()?;
        self.with_exclusive_uid_lock(record.uid, || {
            let current = self.read_optional_payload(record.uid)?;
            let generation =
                next_generation(current.as_ref().map_or(0, StoredPayload::generation))?;
            let payload = StoredPayload::Live {
                schema_version: STORED_PAYLOAD_SCHEMA_VERSION,
                generation,
                record: record.clone(),
            };
            self.write_payload(record.uid, &payload)?;
            Ok(generation)
        })
    }

    /// Replace an existing live template only if its authenticated generation still matches.
    ///
    /// This is the explicit compare-and-swap entry point for enrollment replacement. A deleted or
    /// absent template is not considered replaceable; callers must deliberately create it through
    /// [`Self::save`] or [`Self::save_with_generation`].
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::StaleGeneration`] when another mutation won the race,
    /// [`StorageError::TemplateNotPresent`] for absent/deleted state, or another storage error for
    /// validation, authentication, key, or atomic-write failure.
    pub fn replace_if_generation(
        &self,
        expected_generation: u64,
        record: &TemplateRecord,
    ) -> Result<u64, StorageError> {
        record.validate()?;
        self.with_exclusive_uid_lock(record.uid, || {
            let current = self.read_optional_payload(record.uid)?;
            let actual_generation = match current {
                Some(StoredPayload::Live { generation, .. }) => generation,
                Some(StoredPayload::Deleted { generation, .. }) => {
                    return Err(StorageError::TemplateNotPresent {
                        last_generation: Some(generation),
                    });
                }
                None => {
                    return Err(StorageError::TemplateNotPresent { last_generation: None });
                }
            };
            if actual_generation != expected_generation {
                return Err(StorageError::StaleGeneration {
                    expected: expected_generation,
                    actual: actual_generation,
                });
            }
            let generation = next_generation(actual_generation)?;
            let payload = StoredPayload::Live {
                schema_version: STORED_PAYLOAD_SCHEMA_VERSION,
                generation,
                record: record.clone(),
            };
            self.write_payload(record.uid, &payload)?;
            Ok(generation)
        })
    }

    /// Load and authenticate a record for one UID.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] when the file is malformed, authentication fails, the record is
    /// invalid, or the record belongs to another UID.
    pub fn load(&self, uid: u32) -> Result<TemplateRecord, StorageError> {
        self.load_with_generation(uid).map(|versioned| versioned.record)
    }

    /// Load and authenticate a template together with the generation used for compare-and-swap.
    ///
    /// Legacy template payloads are returned at generation zero. An authenticated deletion
    /// tombstone is intentionally exposed as `NotFound`, while retaining its generation internally
    /// to prevent ABA replacement after deletion.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] when the file is absent/deleted, malformed, unauthenticated,
    /// untrusted, or belongs to another UID.
    pub fn load_with_generation(&self, uid: u32) -> Result<VersionedTemplateRecord, StorageError> {
        match self.read_payload(uid)? {
            StoredPayload::Live { generation, ref record, .. } => {
                Ok(VersionedTemplateRecord { generation, record: record.clone() })
            }
            StoredPayload::Deleted { .. } => Err(template_not_found(uid)),
        }
    }

    /// Atomically delete a live template while retaining only an authenticated generation
    /// tombstone.
    ///
    /// The tombstone contains no embedding. It prevents a generation observed before deletion from
    /// replacing a later enrollment after the same UID is recreated.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::TemplateNotPresent`] if no live template exists, or another storage
    /// error for unsafe paths, corrupt state, generation exhaustion, key/encryption failure, or
    /// atomic persistence failure.
    pub fn delete(&self, uid: u32) -> Result<u64, StorageError> {
        self.delete_inner(uid, None)
    }

    /// Atomically delete a live template only if its authenticated generation still matches.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::StaleGeneration`] when another mutation won the race,
    /// [`StorageError::TemplateNotPresent`] for absent/deleted state, or another storage error.
    pub fn delete_if_generation(
        &self,
        uid: u32,
        expected_generation: u64,
    ) -> Result<u64, StorageError> {
        self.delete_inner(uid, Some(expected_generation))
    }

    fn delete_inner(
        &self,
        uid: u32,
        expected_generation: Option<u64>,
    ) -> Result<u64, StorageError> {
        self.with_exclusive_uid_lock(uid, || {
            let current = self.read_optional_payload(uid)?;
            let actual_generation = match current {
                Some(StoredPayload::Live { generation, .. }) => generation,
                Some(StoredPayload::Deleted { generation, .. }) => {
                    return Err(StorageError::TemplateNotPresent {
                        last_generation: Some(generation),
                    });
                }
                None => {
                    return Err(StorageError::TemplateNotPresent { last_generation: None });
                }
            };
            if let Some(expected) = expected_generation
                && expected != actual_generation
            {
                return Err(StorageError::StaleGeneration { expected, actual: actual_generation });
            }
            let generation = next_generation(actual_generation)?;
            let payload = StoredPayload::Deleted {
                schema_version: STORED_PAYLOAD_SCHEMA_VERSION,
                generation,
                uid,
            };
            self.write_payload(uid, &payload)?;
            Ok(generation)
        })
    }

    fn read_payload(&self, uid: u32) -> Result<StoredPayload, StorageError> {
        let path = self.path_for(uid);
        verify_directory_storage(&self.directory, self.trust)?;
        verify_replace_target_storage(&path, self.trust)?;
        let mut file = open_read_nofollow(&path)?;
        verify_open_storage_file(&file, &path, self.trust)?;
        let mut encoded = Vec::new();
        file.read_to_end(&mut encoded)?;
        if encoded.len() < HEADER_LENGTH || encoded.len() > HEADER_LENGTH + MAX_CIPHERTEXT_LENGTH {
            return Err(StorageError::InvalidRecord("encrypted template size is invalid"));
        }
        if &encoded[..MAGIC.len()] != MAGIC {
            return Err(StorageError::InvalidRecord("encrypted template magic is invalid"));
        }
        let version_offset = MAGIC.len();
        let version = u16::from_le_bytes([encoded[version_offset], encoded[version_offset + 1]]);
        if version != FORMAT_VERSION {
            return Err(StorageError::InvalidRecord("encrypted template version is unsupported"));
        }
        let nonce_start = version_offset + std::mem::size_of::<u16>();
        let nonce_end = nonce_start + NONCE_LENGTH;
        let nonce: [u8; NONCE_LENGTH] = encoded[nonce_start..nonce_end]
            .try_into()
            .map_err(|_| StorageError::InvalidRecord("encrypted template nonce is invalid"))?;
        let key = self.key_provider.load_key()?;
        let cipher = XChaCha20Poly1305::new_from_slice(key.as_bytes())
            .map_err(|_| StorageError::InvalidRecord("invalid encryption key"))?;
        let nonce = XNonce::try_from(nonce.as_slice())
            .map_err(|_| StorageError::InvalidRecord("invalid encryption nonce"))?;
        let plaintext = Zeroizing::new(
            cipher
                .decrypt(&nonce, Payload { msg: &encoded[nonce_end..], aad: &associated_data(uid) })
                .map_err(|_| StorageError::AuthenticationFailed)?,
        );
        if let Ok(payload) = serde_json::from_slice::<StoredPayload>(&plaintext) {
            validate_stored_payload(&payload, uid)?;
            return Ok(payload);
        }
        let record: TemplateRecord = serde_json::from_slice(&plaintext)?;
        record.validate()?;
        if record.uid != uid {
            return Err(StorageError::InvalidRecord("template UID does not match its path"));
        }
        Ok(StoredPayload::Live { schema_version: 0, generation: 0, record })
    }

    fn read_optional_payload(&self, uid: u32) -> Result<Option<StoredPayload>, StorageError> {
        match self.read_payload(uid) {
            Ok(payload) => Ok(Some(payload)),
            Err(StorageError::Io(error)) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error),
        }
    }

    fn write_payload(&self, uid: u32, payload: &StoredPayload) -> Result<(), StorageError> {
        validate_stored_payload(payload, uid)?;
        let key = self.key_provider.load_key()?;
        let plaintext = Zeroizing::new(serde_json::to_vec(payload)?);
        let nonce = random_nonce()?;
        let aad = associated_data(uid);
        let cipher = XChaCha20Poly1305::new_from_slice(key.as_bytes())
            .map_err(|_| StorageError::InvalidRecord("invalid encryption key"))?;
        let nonce = XNonce::try_from(nonce.as_slice())
            .map_err(|_| StorageError::InvalidRecord("invalid encryption nonce"))?;
        let ciphertext = cipher
            .encrypt(&nonce, Payload { msg: &plaintext, aad: &aad })
            .map_err(|_| StorageError::AuthenticationFailed)?;
        let mut encoded = Vec::with_capacity(HEADER_LENGTH + ciphertext.len());
        encoded.extend_from_slice(MAGIC);
        encoded.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        encoded.extend_from_slice(&nonce);
        encoded.extend_from_slice(&ciphertext);
        atomic_write(&self.path_for(uid), &encoded, &self.directory, self.trust)
    }

    fn with_exclusive_uid_lock<T>(
        &self,
        uid: u32,
        operation: impl FnOnce() -> Result<T, StorageError>,
    ) -> Result<T, StorageError> {
        prepare_directory_storage(&self.directory, self.trust)?;
        let lock_path = self.lock_path_for(uid);
        verify_replace_target_storage(&lock_path, self.trust)?;
        let lock_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(&lock_path)?;
        verify_open_storage_file(&lock_file, &lock_path, self.trust)?;
        let _guard = ExclusiveFileLock::acquire(&lock_file)?;
        verify_directory_storage(&self.directory, self.trust)?;
        verify_replace_target_storage(&lock_path, self.trust)?;
        operation()
    }

    /// Return whether a UID has a fully authenticated, structurally valid encrypted template.
    ///
    /// This performs the same bounded read, key retrieval, authenticated decryption, and record
    /// validation as [`Self::load`]. A missing directory or template returns `false`; corrupt or
    /// untrusted storage remains an error.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] for any failure other than a missing directory or template.
    pub fn has_authenticated_template(&self, uid: u32) -> Result<bool, StorageError> {
        match self.load(uid) {
            Ok(record) => {
                drop(record);
                Ok(true)
            }
            Err(StorageError::Io(error)) if error.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(error),
        }
    }

    fn path_for(&self, uid: u32) -> PathBuf {
        self.directory.join(format!("{uid}.template"))
    }

    fn lock_path_for(&self, uid: u32) -> PathBuf {
        self.directory.join(format!(".{uid}.template.lock"))
    }

    #[cfg(test)]
    fn for_test(directory: impl Into<PathBuf>, key_provider: K, owner_uid: u32) -> Self {
        Self { directory: directory.into(), key_provider, trust: TrustPolicy::test(owner_uid) }
    }
}

fn validate_stored_payload(payload: &StoredPayload, expected_uid: u32) -> Result<(), StorageError> {
    match payload {
        StoredPayload::Live { schema_version, generation, record } => {
            if *schema_version != STORED_PAYLOAD_SCHEMA_VERSION || *generation == 0 {
                return Err(StorageError::InvalidRecord(
                    "stored template generation metadata is invalid",
                ));
            }
            record.validate()?;
            if record.uid != expected_uid {
                return Err(StorageError::InvalidRecord("template UID does not match its path"));
            }
        }
        StoredPayload::Deleted { schema_version, generation, uid } => {
            if *schema_version != STORED_PAYLOAD_SCHEMA_VERSION
                || *generation == 0
                || *uid != expected_uid
            {
                return Err(StorageError::InvalidRecord(
                    "stored deletion generation metadata is invalid",
                ));
            }
        }
    }
    Ok(())
}

fn next_generation(current: u64) -> Result<u64, StorageError> {
    current.checked_add(1).ok_or(StorageError::GenerationExhausted)
}

fn template_not_found(uid: u32) -> StorageError {
    StorageError::Io(io::Error::new(
        io::ErrorKind::NotFound,
        format!("template for UID {uid} is deleted or absent"),
    ))
}

struct ExclusiveFileLock<'a> {
    file: &'a File,
}

impl<'a> ExclusiveFileLock<'a> {
    fn acquire(file: &'a File) -> Result<Self, StorageError> {
        file.lock()?;
        Ok(Self { file })
    }
}

impl Drop for ExclusiveFileLock<'_> {
    fn drop(&mut self) {
        let _unlock_result = self.file.unlock();
    }
}

fn random_nonce() -> Result<[u8; NONCE_LENGTH], StorageError> {
    let mut nonce = [0_u8; NONCE_LENGTH];
    getrandom::fill(&mut nonce).map_err(|error| StorageError::Random(error.to_string()))?;
    Ok(nonce)
}

fn associated_data(uid: u32) -> Vec<u8> {
    let mut aad = Vec::with_capacity(MAGIC.len() + 2 + 4);
    aad.extend_from_slice(MAGIC);
    aad.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    aad.extend_from_slice(&uid.to_le_bytes());
    aad
}

fn atomic_write(
    path: &Path,
    bytes: &[u8],
    directory: &Path,
    trust: TrustPolicy,
) -> Result<(), StorageError> {
    verify_directory_storage(directory, trust)?;
    verify_replace_target_storage(path, trust)?;
    let temporary = directory.join(format!(
        ".{}.tmp-{}",
        path.file_name().unwrap_or_default().to_string_lossy(),
        Uuid::new_v4()
    ));
    let result: Result<(), StorageError> = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(&temporary)?;
        verify_open_storage_file(&file, &temporary, trust)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        let directory_file = File::open(directory)?;
        directory_file.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn prepare_directory_storage(directory: &Path, trust: TrustPolicy) -> Result<(), StorageError> {
    match fs::symlink_metadata(directory) {
        Ok(_) => verify_directory_storage(directory, trust),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            let mut existing = directory
                .parent()
                .ok_or_else(|| StorageError::UnsafePathType { path: directory.to_owned() })?;
            while !existing.exists() {
                existing = existing
                    .parent()
                    .ok_or_else(|| StorageError::UnsafePathType { path: directory.to_owned() })?;
            }
            verify_directory_storage(existing, trust)?;
            fs::create_dir_all(directory)?;
            fs::set_permissions(directory, fs::Permissions::from_mode(0o700))?;
            verify_directory_storage(directory, trust)
        }
        Err(error) => Err(StorageError::Io(error)),
    }
}

pub(crate) fn prepare_directory_key(directory: &Path, trust: TrustPolicy) -> Result<(), KeyError> {
    match fs::symlink_metadata(directory) {
        Ok(_) => verify_directory_key(directory, trust),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            let mut existing =
                directory.parent().ok_or_else(|| KeyError::InvalidPath(directory.to_owned()))?;
            while !existing.exists() {
                existing =
                    existing.parent().ok_or_else(|| KeyError::InvalidPath(directory.to_owned()))?;
            }
            verify_directory_key(existing, trust)?;
            fs::create_dir_all(directory)?;
            fs::set_permissions(directory, fs::Permissions::from_mode(0o700))?;
            verify_directory_key(directory, trust)
        }
        Err(error) => Err(KeyError::Io(error)),
    }
}

fn verify_directory_storage(path: &Path, trust: TrustPolicy) -> Result<(), StorageError> {
    verify_directory_chain(path, trust).map_err(|violation| match violation {
        TrustViolation::Io(error) => StorageError::Io(error),
        TrustViolation::Owner { path, actual_uid } => {
            StorageError::UnsafeOwner { path, expected_uid: trust.owner_uid, actual_uid }
        }
        TrustViolation::Permissions { path, mode } => {
            StorageError::UnsafePermissions { path, mode }
        }
        TrustViolation::Type { path } => StorageError::UnsafePathType { path },
    })
}

pub(crate) fn verify_directory_key(path: &Path, trust: TrustPolicy) -> Result<(), KeyError> {
    verify_directory_chain(path, trust).map_err(|violation| match violation {
        TrustViolation::Io(error) => KeyError::Io(error),
        TrustViolation::Owner { path, actual_uid } => {
            KeyError::UnsafeOwner { path, expected_uid: trust.owner_uid, actual_uid }
        }
        TrustViolation::Permissions { path, mode } => KeyError::UnsafePermissions { path, mode },
        TrustViolation::Type { path } => KeyError::UnsafePathType { path },
    })
}

fn verify_directory_chain(path: &Path, trust: TrustPolicy) -> Result<(), TrustViolation> {
    let mut directories = path.ancestors();
    loop {
        let Some(directory) = directories.next() else {
            return Ok(());
        };
        let metadata = fs::symlink_metadata(directory).map_err(TrustViolation::Io)?;
        if !metadata.file_type().is_dir() {
            return Err(TrustViolation::Type { path: directory.to_owned() });
        }
        if metadata.uid() != trust.owner_uid {
            return Err(TrustViolation::Owner {
                path: directory.to_owned(),
                actual_uid: metadata.uid(),
            });
        }
        let mode = metadata.permissions().mode() & 0o777;
        if mode & 0o022 != 0 {
            return Err(TrustViolation::Permissions { path: directory.to_owned(), mode });
        }
        if !trust.verify_ancestors {
            return Ok(());
        }
    }
}

pub(crate) fn verify_open_key_file(
    file: &File,
    path: &Path,
    trust: TrustPolicy,
) -> Result<(), KeyError> {
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(KeyError::UnsafePathType { path: path.to_owned() });
    }
    if metadata.uid() != trust.owner_uid {
        return Err(KeyError::UnsafeOwner {
            path: path.to_owned(),
            expected_uid: trust.owner_uid,
            actual_uid: metadata.uid(),
        });
    }
    let mode = metadata.permissions().mode() & 0o777;
    if mode & 0o077 != 0 {
        return Err(KeyError::UnsafePermissions { path: path.to_owned(), mode });
    }
    Ok(())
}

fn verify_open_storage_file(
    file: &File,
    path: &Path,
    trust: TrustPolicy,
) -> Result<(), StorageError> {
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(StorageError::UnsafePathType { path: path.to_owned() });
    }
    if metadata.uid() != trust.owner_uid {
        return Err(StorageError::UnsafeOwner {
            path: path.to_owned(),
            expected_uid: trust.owner_uid,
            actual_uid: metadata.uid(),
        });
    }
    let mode = metadata.permissions().mode() & 0o777;
    if mode & 0o077 != 0 {
        return Err(StorageError::UnsafePermissions { path: path.to_owned(), mode });
    }
    Ok(())
}

fn verify_replace_target_storage(path: &Path, trust: TrustPolicy) -> Result<(), StorageError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            Err(StorageError::Symlink { path: path.to_owned() })
        }
        Ok(metadata) if !metadata.is_file() => {
            Err(StorageError::UnsafePathType { path: path.to_owned() })
        }
        Ok(metadata) if metadata.uid() != trust.owner_uid => Err(StorageError::UnsafeOwner {
            path: path.to_owned(),
            expected_uid: trust.owner_uid,
            actual_uid: metadata.uid(),
        }),
        Ok(metadata) => {
            let mode = metadata.permissions().mode() & 0o777;
            if mode & 0o077 != 0 {
                Err(StorageError::UnsafePermissions { path: path.to_owned(), mode })
            } else {
                Ok(())
            }
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(StorageError::Io(error)),
    }
}

pub(crate) fn verify_replace_target_key(path: &Path, trust: TrustPolicy) -> Result<(), KeyError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            Err(KeyError::Symlink { path: path.to_owned() })
        }
        Ok(metadata) if !metadata.is_file() => {
            Err(KeyError::UnsafePathType { path: path.to_owned() })
        }
        Ok(metadata) if metadata.uid() != trust.owner_uid => Err(KeyError::UnsafeOwner {
            path: path.to_owned(),
            expected_uid: trust.owner_uid,
            actual_uid: metadata.uid(),
        }),
        Ok(metadata) => {
            let mode = metadata.permissions().mode() & 0o777;
            if mode & 0o077 != 0 {
                Err(KeyError::UnsafePermissions { path: path.to_owned(), mode })
            } else {
                Ok(())
            }
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(KeyError::Io(error)),
    }
}

enum TrustViolation {
    Io(io::Error),
    Owner { path: PathBuf, actual_uid: u32 },
    Permissions { path: PathBuf, mode: u32 },
    Type { path: PathBuf },
}

pub(crate) fn open_read_nofollow(path: &Path) -> io::Result<File> {
    OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC).open(path)
}

/// Key loading failure.
#[derive(Debug, Error)]
pub enum KeyError {
    /// The key file did not contain exactly 32 bytes.
    #[error("key length {actual} is not {expected} bytes")]
    InvalidLength {
        /// Required key length.
        expected: usize,
        /// Actual key length.
        actual: usize,
    },
    /// A configured key or blob path is invalid.
    #[error("invalid key path {0}")]
    InvalidPath(PathBuf),
    /// The persisted TPM public/private blob is malformed.
    #[error("invalid TPM sealed-key blob")]
    InvalidBlob,
    /// TPM context or command failed.
    #[error("TPM operation failed: {0}")]
    Tpm(String),
    /// The key file has group or other permissions.
    #[error("key file {path} has unsafe permissions {mode:o}")]
    UnsafePermissions {
        /// Key path.
        path: PathBuf,
        /// Unix permission bits.
        mode: u32,
    },
    /// Key file or containing directory has an unexpected owner.
    #[error("key path {path} owner {actual_uid} does not match required UID {expected_uid}")]
    UnsafeOwner {
        /// Rejected path.
        path: PathBuf,
        /// Required owner UID.
        expected_uid: u32,
        /// Actual owner UID.
        actual_uid: u32,
    },
    /// Key path component is not the required regular-file or directory type.
    #[error("key path has unsafe filesystem type: {path}")]
    UnsafePathType {
        /// Rejected path.
        path: PathBuf,
    },
    /// The key path is a symbolic link.
    #[error("key path {path} must not be a symbolic link")]
    Symlink {
        /// Key path.
        path: PathBuf,
    },
    /// The operating system random source failed.
    #[error("random source failed: {0}")]
    Random(String),
    /// File operation failed.
    #[error("key file operation failed: {0}")]
    Io(#[from] io::Error),
}

/// Storage or authenticated-encryption failure.
#[derive(Debug, Error)]
pub enum StorageError {
    /// Key provider failed.
    #[error(transparent)]
    Key(#[from] KeyError),
    /// A record or encrypted container is invalid.
    #[error("invalid template record: {0}")]
    InvalidRecord(&'static str),
    /// Template was produced by a different model contract or embedding dimension.
    #[error("template is incompatible with the active embedding model contract")]
    IncompatibleTemplate,
    /// A compare-and-swap mutation used an obsolete authenticated generation.
    #[error("template generation changed from expected {expected} to {actual}")]
    StaleGeneration {
        /// Generation observed by the caller.
        expected: u64,
        /// Generation authenticated while holding the UID mutation lock.
        actual: u64,
    },
    /// Replacement or deletion requires a currently live template.
    #[error("template is not present; last authenticated generation is {last_generation:?}")]
    TemplateNotPresent {
        /// Last tombstone generation, or `None` when no state exists.
        last_generation: Option<u64>,
    },
    /// No later generation can be represented safely.
    #[error("template generation is exhausted")]
    GenerationExhausted,
    /// JSON encoding or decoding failed.
    #[error("template serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
    /// Authenticated encryption rejected ciphertext or failed to encrypt.
    #[error("template authentication failed")]
    AuthenticationFailed,
    /// The template path is a symbolic link.
    #[error("template path {path} must not be a symbolic link")]
    Symlink {
        /// Template path.
        path: PathBuf,
    },
    /// The template file has group or other permissions.
    #[error("template file {path} has unsafe permissions {mode:o}")]
    UnsafePermissions {
        /// Template path.
        path: PathBuf,
        /// Unix permission bits.
        mode: u32,
    },
    /// Template file or containing directory has an unexpected owner.
    #[error("template path {path} owner {actual_uid} does not match required UID {expected_uid}")]
    UnsafeOwner {
        /// Rejected path.
        path: PathBuf,
        /// Required owner UID.
        expected_uid: u32,
        /// Actual owner UID.
        actual_uid: u32,
    },
    /// Template path component is not the required regular-file or directory type.
    #[error("template path has unsafe filesystem type: {path}")]
    UnsafePathType {
        /// Rejected path.
        path: PathBuf,
    },
    /// Random nonce generation failed.
    #[error("random source failed: {0}")]
    Random(String),
    /// File operation failed.
    #[error("template storage operation failed: {0}")]
    Io(#[from] io::Error),
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, Barrier},
        thread,
    };

    use super::*;

    #[derive(Clone, Copy)]
    struct TestKeyProvider([u8; KEY_LENGTH]);

    impl KeyProvider for TestKeyProvider {
        fn load_key(&self) -> Result<SecretKey, KeyError> {
            SecretKey::from_slice(&self.0)
        }

        fn strength(&self) -> KeyStrength {
            KeyStrength::TpmBound
        }
    }

    fn record(uid: u32) -> TemplateRecord {
        TemplateRecord {
            schema_version: FORMAT_VERSION,
            uid,
            model_compatibility_sha256:
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
            embedding: vec![1.0, 0.0, 0.0],
        }
    }

    fn alternate_record(uid: u32, axis: usize) -> TemplateRecord {
        let mut template = record(uid);
        template.embedding = vec![0.0; 3];
        template.embedding[axis] = 1.0;
        template
    }

    fn temporary_directory() -> Result<PathBuf, StorageError> {
        let directory = std::env::temp_dir().join(format!("faceauth-storage-{}", Uuid::new_v4()));
        fs::create_dir(&directory)?;
        Ok(directory)
    }

    fn test_store(
        directory: &Path,
    ) -> Result<EncryptedTemplateStore<TestKeyProvider>, StorageError> {
        let owner_uid = fs::metadata(directory)?.uid();
        Ok(EncryptedTemplateStore::for_test(directory, TestKeyProvider([7; KEY_LENGTH]), owner_uid))
    }

    fn write_legacy_record(directory: &Path, record: &TemplateRecord) -> Result<(), StorageError> {
        let owner_uid = fs::metadata(directory)?.uid();
        let key = SecretKey::from_slice(&[7; KEY_LENGTH])?;
        let plaintext = Zeroizing::new(serde_json::to_vec(record)?);
        let nonce = random_nonce()?;
        let cipher = XChaCha20Poly1305::new_from_slice(key.as_bytes())
            .map_err(|_| StorageError::InvalidRecord("invalid encryption key"))?;
        let nonce = XNonce::try_from(nonce.as_slice())
            .map_err(|_| StorageError::InvalidRecord("invalid encryption nonce"))?;
        let ciphertext = cipher
            .encrypt(&nonce, Payload { msg: &plaintext, aad: &associated_data(record.uid) })
            .map_err(|_| StorageError::AuthenticationFailed)?;
        let mut encoded = Vec::with_capacity(HEADER_LENGTH + ciphertext.len());
        encoded.extend_from_slice(MAGIC);
        encoded.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        encoded.extend_from_slice(&nonce);
        encoded.extend_from_slice(&ciphertext);
        atomic_write(
            &directory.join(format!("{}.template", record.uid)),
            &encoded,
            directory,
            TrustPolicy::test(owner_uid),
        )
    }

    #[test]
    fn encrypted_records_round_trip_without_raw_frames() -> Result<(), StorageError> {
        let directory = temporary_directory()?;
        let store = test_store(&directory)?;
        let expected = record(1000);
        assert!(!store.has_authenticated_template(1000)?);
        store.save(&expected)?;
        let actual = store.load(1000)?;

        assert_eq!(actual, expected);
        assert_eq!(store.load_with_generation(1000)?.generation, 1);
        assert_eq!(store.save_with_generation(&expected)?, 2);
        assert!(store.has_authenticated_template(1000)?);
        assert!(!store.has_authenticated_template(1001)?);
        assert_eq!(store.key_strength(), KeyStrength::TpmBound);
        let _ = fs::remove_dir_all(directory);
        Ok(())
    }

    #[test]
    fn legacy_records_load_at_generation_zero_and_upgrade_through_cas() -> Result<(), StorageError>
    {
        let directory = temporary_directory()?;
        let store = test_store(&directory)?;
        write_legacy_record(&directory, &record(1000))?;

        let legacy = store.load_with_generation(1000)?;
        assert_eq!(legacy.generation, 0);
        assert_eq!(legacy.record, record(1000));
        let replacement = alternate_record(1000, 1);
        assert_eq!(store.replace_if_generation(0, &replacement)?, 1);
        let upgraded = store.load_with_generation(1000)?;
        assert_eq!(upgraded.generation, 1);
        assert_eq!(upgraded.record, replacement);

        let _ = fs::remove_dir_all(directory);
        Ok(())
    }

    #[test]
    fn stale_generation_cannot_overwrite_a_newer_template() -> Result<(), StorageError> {
        let directory = temporary_directory()?;
        let store = test_store(&directory)?;
        assert_eq!(store.save_with_generation(&record(1000))?, 1);
        let replacement = alternate_record(1000, 1);
        assert_eq!(store.replace_if_generation(1, &replacement)?, 2);

        let stale = alternate_record(1000, 2);
        assert!(matches!(
            store.replace_if_generation(1, &stale),
            Err(StorageError::StaleGeneration { expected: 1, actual: 2 })
        ));
        assert!(matches!(
            store.delete_if_generation(1000, 1),
            Err(StorageError::StaleGeneration { expected: 1, actual: 2 })
        ));
        let current = store.load_with_generation(1000)?;
        assert_eq!(current.generation, 2);
        assert_eq!(current.record, replacement);

        let _ = fs::remove_dir_all(directory);
        Ok(())
    }

    #[test]
    fn atomic_delete_retains_only_a_generation_tombstone_and_prevents_aba()
    -> Result<(), StorageError> {
        let directory = temporary_directory()?;
        let store = test_store(&directory)?;
        assert_eq!(store.save_with_generation(&record(1000))?, 1);
        assert_eq!(store.delete_if_generation(1000, 1)?, 2);
        assert!(!store.has_authenticated_template(1000)?);
        assert!(matches!(
            store.load(1000),
            Err(StorageError::Io(error)) if error.kind() == io::ErrorKind::NotFound
        ));
        assert!(matches!(
            store.read_payload(1000)?,
            StoredPayload::Deleted { generation: 2, uid: 1000, .. }
        ));
        assert!(matches!(
            store.replace_if_generation(1, &alternate_record(1000, 1)),
            Err(StorageError::TemplateNotPresent { last_generation: Some(2) })
        ));

        assert_eq!(store.save_with_generation(&alternate_record(1000, 1))?, 3);
        assert!(matches!(
            store.replace_if_generation(1, &alternate_record(1000, 2)),
            Err(StorageError::StaleGeneration { expected: 1, actual: 3 })
        ));

        let _ = fs::remove_dir_all(directory);
        Ok(())
    }

    #[test]
    fn concurrent_compare_and_swap_has_exactly_one_winner() -> Result<(), StorageError> {
        let directory = temporary_directory()?;
        let store = test_store(&directory)?;
        assert_eq!(store.save_with_generation(&record(1000))?, 1);
        let barrier = Arc::new(Barrier::new(3));
        let mut workers = Vec::new();
        for axis in [1, 2] {
            let directory = directory.clone();
            let barrier = Arc::clone(&barrier);
            workers.push(thread::spawn(move || {
                let store = test_store(&directory)?;
                barrier.wait();
                store.replace_if_generation(1, &alternate_record(1000, axis))
            }));
        }
        barrier.wait();
        let results = workers
            .into_iter()
            .map(|worker| {
                worker.join().map_err(|_| StorageError::InvalidRecord("CAS test worker panicked"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(results.iter().filter(|result| matches!(result, Ok(2))).count(), 1);
        assert_eq!(
            results
                .iter()
                .filter(|result| matches!(
                    result,
                    Err(StorageError::StaleGeneration { expected: 1, actual: 2 })
                ))
                .count(),
            1
        );
        assert_eq!(store.load_with_generation(1000)?.generation, 2);

        let _ = fs::remove_dir_all(directory);
        Ok(())
    }

    #[test]
    fn deletion_rejects_symlink_targets_before_mutation() -> Result<(), StorageError> {
        let directory = temporary_directory()?;
        let store = test_store(&directory)?;
        store.save(&record(1000))?;
        let target = directory.join("real.template");
        fs::rename(directory.join("1000.template"), &target)?;
        std::os::unix::fs::symlink(&target, directory.join("1000.template"))?;

        assert!(matches!(store.delete(1000), Err(StorageError::Symlink { .. })));
        assert!(target.exists());
        let _ = fs::remove_dir_all(directory);
        Ok(())
    }

    #[test]
    fn uid_lock_symlinks_are_rejected_before_opening() -> Result<(), StorageError> {
        let directory = temporary_directory()?;
        let store = test_store(&directory)?;
        let target = directory.join("attacker-lock-target");
        fs::write(&target, [])?;
        fs::set_permissions(&target, fs::Permissions::from_mode(0o600))?;
        std::os::unix::fs::symlink(&target, directory.join(".1000.template.lock"))?;

        assert!(matches!(store.save(&record(1000)), Err(StorageError::Symlink { .. })));
        assert_eq!(fs::metadata(target)?.len(), 0);
        let _ = fs::remove_dir_all(directory);
        Ok(())
    }

    #[test]
    fn tampering_fails_authenticated_decryption() -> Result<(), StorageError> {
        let directory = temporary_directory()?;
        let store = test_store(&directory)?;
        store.save(&record(1000))?;
        let path = directory.join("1000.template");
        let mut bytes = fs::read(&path)?;
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        fs::write(path, bytes)?;

        assert!(matches!(store.load(1000), Err(StorageError::AuthenticationFailed)));
        assert!(matches!(
            store.has_authenticated_template(1000),
            Err(StorageError::AuthenticationFailed)
        ));
        let _ = fs::remove_dir_all(directory);
        Ok(())
    }

    #[test]
    fn file_key_provider_creates_owner_only_key() -> Result<(), KeyError> {
        let directory = std::env::temp_dir().join(format!("faceauth-key-{}", Uuid::new_v4()));
        fs::create_dir(&directory).map_err(KeyError::Io)?;
        let path = directory.join("machine.key");
        let owner_uid = fs::metadata(&directory).map_err(KeyError::Io)?.uid();
        let provider = FileKeyProvider::for_test(&path, owner_uid);
        let first = provider.load_key()?;
        let second = provider.load_key()?;

        assert_eq!(first.as_bytes(), second.as_bytes());
        assert_eq!(provider.strength(), KeyStrength::RootOnlyFile);
        let mode = fs::metadata(path).map_err(KeyError::Io)?.permissions().mode();
        assert_eq!(mode & 0o077, 0);
        let _ = fs::remove_dir_all(directory);
        Ok(())
    }

    #[test]
    fn template_symlink_is_rejected() -> Result<(), StorageError> {
        let directory = temporary_directory()?;
        let store = test_store(&directory)?;
        store.save(&record(1000))?;
        let target = directory.join("real.template");
        fs::rename(directory.join("1000.template"), &target)?;
        std::os::unix::fs::symlink(&target, directory.join("1000.template"))?;

        assert!(matches!(store.load(1000), Err(StorageError::Symlink { .. })));
        let _ = fs::remove_dir_all(directory);
        Ok(())
    }

    #[test]
    fn template_group_permissions_are_rejected() -> Result<(), StorageError> {
        let directory = temporary_directory()?;
        let store = test_store(&directory)?;
        store.save(&record(1000))?;
        let path = directory.join("1000.template");
        let mut permissions = fs::metadata(&path)?.permissions();
        permissions.set_mode(0o640);
        fs::set_permissions(&path, permissions)?;

        assert!(matches!(store.load(1000), Err(StorageError::UnsafePermissions { .. })));
        let _ = fs::remove_dir_all(directory);
        Ok(())
    }

    #[test]
    fn template_requires_exact_model_contract_and_dimension() -> Result<(), StorageError> {
        let template = record(1000);
        template.require_compatibility(&template.model_compatibility_sha256, 3)?;
        assert!(matches!(
            template.require_compatibility(
                "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                3
            ),
            Err(StorageError::IncompatibleTemplate)
        ));
        assert!(matches!(
            template.require_compatibility(&template.model_compatibility_sha256, 512),
            Err(StorageError::IncompatibleTemplate)
        ));
        Ok(())
    }

    #[test]
    fn non_normalized_templates_are_rejected() {
        let mut template = record(1000);
        template.embedding = vec![0.5, 0.0, 0.0];
        assert!(matches!(template.validate(), Err(StorageError::InvalidRecord(_))));
    }

    #[test]
    fn unsafe_or_wrong_owner_directories_are_rejected() -> Result<(), StorageError> {
        let directory = temporary_directory()?;
        let owner_uid = fs::metadata(&directory)?.uid();
        let wrong_owner_store = EncryptedTemplateStore::for_test(
            &directory,
            TestKeyProvider([7; KEY_LENGTH]),
            owner_uid.saturating_add(1),
        );
        assert!(matches!(
            wrong_owner_store.save(&record(1000)),
            Err(StorageError::UnsafeOwner { .. })
        ));

        let mut permissions = fs::metadata(&directory)?.permissions();
        permissions.set_mode(0o770);
        fs::set_permissions(&directory, permissions)?;
        let unsafe_mode_store = EncryptedTemplateStore::for_test(
            &directory,
            TestKeyProvider([7; KEY_LENGTH]),
            owner_uid,
        );
        assert!(matches!(
            unsafe_mode_store.save(&record(1000)),
            Err(StorageError::UnsafePermissions { .. })
        ));
        let _ = fs::remove_dir_all(directory);
        Ok(())
    }

    #[test]
    fn file_key_symlink_is_rejected_even_when_target_is_owner_only() -> Result<(), KeyError> {
        let directory = std::env::temp_dir().join(format!("faceauth-key-{}", Uuid::new_v4()));
        fs::create_dir(&directory)?;
        let owner_uid = fs::metadata(&directory)?.uid();
        let target = directory.join("target.key");
        fs::write(&target, [7_u8; KEY_LENGTH])?;
        fs::set_permissions(&target, fs::Permissions::from_mode(0o600))?;
        let link = directory.join("machine.key");
        std::os::unix::fs::symlink(&target, &link)?;
        let provider = FileKeyProvider::for_test(&link, owner_uid);
        assert!(matches!(provider.load_key(), Err(KeyError::Symlink { .. })));
        let _ = fs::remove_dir_all(directory);
        Ok(())
    }
}
