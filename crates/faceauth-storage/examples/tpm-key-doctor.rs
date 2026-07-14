//! Non-destructive TPM sealed-key diagnostic.

use std::{env, error::Error, path::PathBuf};

use faceauth_storage::TpmKeyProvider;

fn main() -> Result<(), Box<dyn Error>> {
    let path = env::args_os()
        .nth(1)
        .map_or_else(|| PathBuf::from("/var/lib/faceauth/machine-key.tpm"), PathBuf::from);
    let provider = TpmKeyProvider::new(&path);
    provider.self_test()?;
    println!("TPM sealed-key self-test passed for {}", path.display());
    Ok(())
}
