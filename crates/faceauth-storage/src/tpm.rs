use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
    str::FromStr,
};

use tss_esapi::{
    Context, TctiNameConf,
    attributes::ObjectAttributesBuilder,
    interface_types::{
        algorithm::{HashingAlgorithm, PublicAlgorithm},
        resource_handles::Hierarchy,
    },
    structures::{
        Digest, KeyedHashScheme, Private, Public, PublicBuffer, PublicBuilder,
        PublicKeyedHashParameters, SensitiveData, SymmetricCipherParameters,
        SymmetricDefinitionObject,
    },
};
use uuid::Uuid;
use zeroize::Zeroize;

use super::{KEY_LENGTH, KeyError, KeyProvider, KeyStrength, SecretKey};

const BLOB_MAGIC: &[u8; 4] = b"FATK";
const BLOB_VERSION: u16 = 1;
const BLOB_HEADER_LENGTH: usize = BLOB_MAGIC.len() + 2 + 4 + 4;
const MAX_BLOB_LENGTH: usize = 64 * 1024;

/// TPM 2.0 sealed machine-key provider.
///
/// The sealed object has no PCR policy in the initial implementation, so kernel and firmware
/// updates do not invalidate enrollment. It is machine-bound to the TPM owner hierarchy but does
/// not claim Windows Hello Enhanced Sign-in Security equivalence.
#[derive(Clone, Debug)]
pub struct TpmKeyProvider {
    blob_path: PathBuf,
    tcti: String,
}

impl TpmKeyProvider {
    /// Use the kernel resource manager at `/dev/tpmrm0` and persist one sealed-key blob.
    #[must_use]
    pub fn new(blob_path: impl Into<PathBuf>) -> Self {
        Self { blob_path: blob_path.into(), tcti: "device:/dev/tpmrm0".to_owned() }
    }

    /// Override the TCTI string for testing with a TPM simulator or another device.
    #[must_use]
    pub fn with_tcti(blob_path: impl Into<PathBuf>, tcti: impl Into<String>) -> Self {
        Self { blob_path: blob_path.into(), tcti: tcti.into() }
    }

    /// Return the sealed-key blob path.
    #[must_use]
    pub fn blob_path(&self) -> &Path {
        &self.blob_path
    }

    /// Create or open the sealed key and verify that two independent unseal operations agree.
    ///
    /// The secret key is never returned or printed by this diagnostic.
    ///
    /// # Errors
    ///
    /// Returns [`KeyError`] when TPM access, sealed-blob persistence, or key consistency fails.
    pub fn self_test(&self) -> Result<(), KeyError> {
        let first = self.load_key()?;
        let second = self.load_key()?;
        if first.same_secret(&second) {
            Ok(())
        } else {
            Err(KeyError::Tpm("successive unseal operations returned different keys".to_owned()))
        }
    }

    fn context(&self) -> Result<Context, KeyError> {
        let tcti =
            TctiNameConf::from_str(&self.tcti).map_err(|error| KeyError::Tpm(error.to_string()))?;
        Context::new(tcti).map_err(|error| KeyError::Tpm(error.to_string()))
    }

    fn create_sealed_key(&self) -> Result<SecretKey, KeyError> {
        let parent =
            self.blob_path.parent().ok_or_else(|| KeyError::InvalidPath(self.blob_path.clone()))?;
        fs::create_dir_all(parent)?;
        let mut context = self.context()?;
        let primary = create_primary(&mut context)?;
        let mut key_bytes = [0_u8; KEY_LENGTH];
        getrandom::fill(&mut key_bytes).map_err(|error| KeyError::Random(error.to_string()))?;
        let sensitive = SensitiveData::try_from(key_bytes.to_vec())
            .map_err(|error| KeyError::Tpm(error.to_string()))?;
        let public = sealed_object_public()?;
        let created = context
            .execute_with_nullauth_session(|ctx| {
                ctx.create(primary, public, None, Some(sensitive), None, None)
            })
            .map_err(|error| KeyError::Tpm(error.to_string()))?;
        let public_buffer = PublicBuffer::try_from(created.out_public)
            .map_err(|error| KeyError::Tpm(error.to_string()))?;
        let encoded = encode_blob(public_buffer.value(), created.out_private.value())?;
        let write_result = atomic_write_blob(&self.blob_path, &encoded, parent);
        let flush_result =
            context.flush_context(primary.into()).map_err(|error| KeyError::Tpm(error.to_string()));
        if let Err(error) = write_result {
            key_bytes.zeroize();
            return Err(error);
        }
        flush_result?;
        let key = SecretKey::from_slice(&key_bytes);
        key_bytes.zeroize();
        key
    }

    fn unseal_key(&self) -> Result<SecretKey, KeyError> {
        reject_symlink(&self.blob_path)?;
        let (public_bytes, private_bytes) = read_blob(&self.blob_path)?;
        let public_buffer = PublicBuffer::try_from(public_bytes)
            .map_err(|error| KeyError::Tpm(error.to_string()))?;
        let public =
            Public::try_from(public_buffer).map_err(|error| KeyError::Tpm(error.to_string()))?;
        let private =
            Private::try_from(private_bytes).map_err(|error| KeyError::Tpm(error.to_string()))?;
        let mut context = self.context()?;
        let primary = create_primary(&mut context)?;
        let unsealed = context
            .execute_with_nullauth_session(|ctx| {
                let loaded = ctx.load(primary, private, public)?;
                let object = loaded.into();
                let result = ctx.unseal(object);
                let flush_result = ctx.flush_context(object);
                match (result, flush_result) {
                    (Ok(data), Ok(())) => Ok(data),
                    (Err(error), _) | (_, Err(error)) => Err(error),
                }
            })
            .map_err(|error| KeyError::Tpm(error.to_string()))?;
        context.flush_context(primary.into()).map_err(|error| KeyError::Tpm(error.to_string()))?;
        SecretKey::from_slice(unsealed.value())
    }
}

impl KeyProvider for TpmKeyProvider {
    fn load_key(&self) -> Result<SecretKey, KeyError> {
        match fs::symlink_metadata(&self.blob_path) {
            Ok(_) => self.unseal_key(),
            Err(error) if error.kind() == io::ErrorKind::NotFound => self.create_sealed_key(),
            Err(error) => Err(KeyError::Io(error)),
        }
    }

    fn strength(&self) -> KeyStrength {
        KeyStrength::TpmBound
    }
}

fn create_primary(context: &mut Context) -> Result<tss_esapi::handles::KeyHandle, KeyError> {
    let attributes = ObjectAttributesBuilder::new()
        .with_fixed_tpm(true)
        .with_fixed_parent(true)
        .with_st_clear(false)
        .with_sensitive_data_origin(true)
        .with_user_with_auth(true)
        .with_decrypt(true)
        .with_restricted(true)
        .build()
        .map_err(|error| KeyError::Tpm(error.to_string()))?;
    let public = PublicBuilder::new()
        .with_public_algorithm(PublicAlgorithm::SymCipher)
        .with_name_hashing_algorithm(HashingAlgorithm::Sha256)
        .with_object_attributes(attributes)
        .with_symmetric_cipher_parameters(SymmetricCipherParameters::new(
            SymmetricDefinitionObject::AES_128_CFB,
        ))
        .with_symmetric_cipher_unique_identifier(Digest::default())
        .build()
        .map_err(|error| KeyError::Tpm(error.to_string()))?;
    context
        .execute_with_nullauth_session(|ctx| {
            ctx.create_primary(Hierarchy::Owner, public, None, None, None, None)
        })
        .map(|result| result.key_handle)
        .map_err(|error| KeyError::Tpm(error.to_string()))
}

fn sealed_object_public() -> Result<Public, KeyError> {
    let attributes = ObjectAttributesBuilder::new()
        .with_fixed_tpm(true)
        .with_fixed_parent(true)
        .with_st_clear(false)
        .with_user_with_auth(true)
        .build()
        .map_err(|error| KeyError::Tpm(error.to_string()))?;
    PublicBuilder::new()
        .with_public_algorithm(PublicAlgorithm::KeyedHash)
        .with_name_hashing_algorithm(HashingAlgorithm::Sha256)
        .with_object_attributes(attributes)
        .with_keyed_hash_parameters(PublicKeyedHashParameters::new(KeyedHashScheme::Null))
        .with_keyed_hash_unique_identifier(Digest::default())
        .build()
        .map_err(|error| KeyError::Tpm(error.to_string()))
}

fn encode_blob(public: &[u8], private: &[u8]) -> Result<Vec<u8>, KeyError> {
    let public_length = u32::try_from(public.len()).map_err(|_| KeyError::InvalidBlob)?;
    let private_length = u32::try_from(private.len()).map_err(|_| KeyError::InvalidBlob)?;
    let total = BLOB_HEADER_LENGTH
        .checked_add(public.len())
        .and_then(|size| size.checked_add(private.len()))
        .ok_or(KeyError::InvalidBlob)?;
    if total > MAX_BLOB_LENGTH {
        return Err(KeyError::InvalidBlob);
    }
    let mut encoded = Vec::with_capacity(total);
    encoded.extend_from_slice(BLOB_MAGIC);
    encoded.extend_from_slice(&BLOB_VERSION.to_le_bytes());
    encoded.extend_from_slice(&public_length.to_le_bytes());
    encoded.extend_from_slice(&private_length.to_le_bytes());
    encoded.extend_from_slice(public);
    encoded.extend_from_slice(private);
    Ok(encoded)
}

fn read_blob(path: &Path) -> Result<(Vec<u8>, Vec<u8>), KeyError> {
    let file = File::open(path)?;
    let mode = file.metadata()?.permissions().mode();
    if mode & 0o077 != 0 {
        return Err(KeyError::UnsafePermissions { path: path.to_owned(), mode: mode & 0o777 });
    }
    let mut encoded = Vec::new();
    file.take(u64::try_from(MAX_BLOB_LENGTH + 1).map_err(|_| KeyError::InvalidBlob)?)
        .read_to_end(&mut encoded)?;
    if encoded.len() < BLOB_HEADER_LENGTH || encoded.len() > MAX_BLOB_LENGTH {
        return Err(KeyError::InvalidBlob);
    }
    if &encoded[..BLOB_MAGIC.len()] != BLOB_MAGIC {
        return Err(KeyError::InvalidBlob);
    }
    let version_offset = BLOB_MAGIC.len();
    let version = u16::from_le_bytes([encoded[version_offset], encoded[version_offset + 1]]);
    if version != BLOB_VERSION {
        return Err(KeyError::InvalidBlob);
    }
    let public_length_offset = version_offset + 2;
    let private_length_offset = public_length_offset + 4;
    let public_length = u32::from_le_bytes(
        encoded[public_length_offset..public_length_offset + 4]
            .try_into()
            .map_err(|_| KeyError::InvalidBlob)?,
    ) as usize;
    let private_length = u32::from_le_bytes(
        encoded[private_length_offset..private_length_offset + 4]
            .try_into()
            .map_err(|_| KeyError::InvalidBlob)?,
    ) as usize;
    let public_start = BLOB_HEADER_LENGTH;
    let public_end = public_start.checked_add(public_length).ok_or(KeyError::InvalidBlob)?;
    let private_end = public_end.checked_add(private_length).ok_or(KeyError::InvalidBlob)?;
    if private_end != encoded.len() {
        return Err(KeyError::InvalidBlob);
    }
    Ok((encoded[public_start..public_end].to_vec(), encoded[public_end..private_end].to_vec()))
}

fn atomic_write_blob(path: &Path, bytes: &[u8], directory: &Path) -> Result<(), KeyError> {
    let temporary = directory.join(format!(".faceauth-tpm-key.tmp-{}", Uuid::new_v4()));
    let result = (|| {
        let mut file =
            OpenOptions::new().write(true).create_new(true).mode(0o600).open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        File::open(directory)?.sync_all()?;
        Ok::<(), io::Error>(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result.map_err(KeyError::Io)
}

fn reject_symlink(path: &Path) -> Result<(), KeyError> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() {
        Err(KeyError::Symlink { path: path.to_owned() })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blob_round_trip_is_bounded() -> Result<(), KeyError> {
        let public = vec![1, 2, 3];
        let private = vec![4, 5, 6, 7];
        let path = std::env::temp_dir().join(format!("faceauth-tpm-blob-{}", Uuid::new_v4()));
        let encoded = encode_blob(&public, &private)?;
        fs::write(&path, encoded)?;
        let mut permissions = fs::metadata(&path)?.permissions();
        permissions.set_mode(0o600);
        fs::set_permissions(&path, permissions)?;
        let decoded = read_blob(&path)?;

        assert_eq!(decoded, (public, private));
        let _ = fs::remove_file(path);
        Ok(())
    }
}
