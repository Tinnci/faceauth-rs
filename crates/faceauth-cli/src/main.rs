//! Command-line diagnostics and future enrollment client for faceauth-rs.

use anyhow::Result;
use clap::{Parser, Subcommand};
use faceauth_core::AuthPolicy;

#[derive(Debug, Parser)]
#[command(about = "Face authentication enrollment and diagnostics client", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Print the conservative default authentication policy.
    Policy,
    /// Explain why enrollment is not enabled in the initial scaffold.
    Enroll,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Policy => println!("{}", serde_json::to_string_pretty(&AuthPolicy::default())?),
        Command::Enroll => anyhow::bail!(
            "enrollment is disabled until encrypted template storage and liveness are implemented"
        ),
    }
    Ok(())
}
