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
    /// # Errors
    ///
    /// Returns [`StorageError`] when validation, key loading, encryption, or the atomic write
    /// fails.
    pub fn save(&self, record: &TemplateRecord) -> Result<(), StorageError> {
        record.validate()?;
        prepare_directory_storage(&self.directory, self.trust)?;
        let key = self.key_provider.load_key()?;
        let plaintext = Zeroizing::new(serde_json::to_vec(record)?);
        let nonce = random_nonce()?;
        let aad = associated_data(record.uid);
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
        atomic_write(&self.path_for(record.uid), &encoded, &self.directory, self.trust)
    }

    /// Load and authenticate a record for one UID.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] when the file is malformed, authentication fails, the record is
    /// invalid, or the record belongs to another UID.
    pub fn load(&self, uid: u32) -> Result<TemplateRecord, StorageError> {
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
        let record: TemplateRecord = serde_json::from_slice(&plaintext)?;
        record.validate()?;
        if record.uid != uid {
            return Err(StorageError::InvalidRecord("template UID does not match its path"));
        }
        Ok(record)
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

    #[cfg(test)]
    fn for_test(directory: impl Into<PathBuf>, key_provider: K, owner_uid: u32) -> Self {
        Self { directory: directory.into(), key_provider, trust: TrustPolicy::test(owner_uid) }
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
    use super::*;

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

    #[test]
    fn encrypted_records_round_trip_without_raw_frames() -> Result<(), StorageError> {
        let directory = temporary_directory()?;
        let store = test_store(&directory)?;
        let expected = record(1000);
        assert!(!store.has_authenticated_template(1000)?);
        store.save(&expected)?;
        let actual = store.load(1000)?;

        assert_eq!(actual, expected);
        assert!(store.has_authenticated_template(1000)?);
        assert!(!store.has_authenticated_template(1001)?);
        assert_eq!(store.key_strength(), KeyStrength::TpmBound);
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
