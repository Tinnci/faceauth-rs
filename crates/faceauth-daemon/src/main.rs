//! Read-only diagnostics and the future privileged face-authentication daemon entry point.

use std::{fs, path::Path};

use anyhow::Result;
use clap::{Parser, Subcommand};
use serde::Serialize;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(about = "Privileged face authentication service", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Print read-only hardware and readiness diagnostics.
    Doctor,
    /// Refuse to start the production service until the transport is implemented.
    Serve,
}

#[derive(Debug, Serialize)]
struct CameraDevice {
    device: String,
    name: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
enum ProductionStatus {
    ScaffoldOnly,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
enum DeviceStatus {
    Available,
    Unavailable,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
enum PasswordFallbackPolicy {
    Mandatory,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
enum HpdRole {
    HintOnly,
}

#[derive(Debug, Serialize)]
struct DoctorReport {
    version: &'static str,
    production_status: ProductionStatus,
    cameras: Vec<CameraDevice>,
    tpm2_resource_manager: DeviceStatus,
    password_fallback: PasswordFallbackPolicy,
    hpd_role: HpdRole,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    match Cli::parse().command {
        Command::Doctor => {
            let report = DoctorReport {
                version: env!("CARGO_PKG_VERSION"),
                production_status: ProductionStatus::ScaffoldOnly,
                cameras: discover_cameras(),
                tpm2_resource_manager: if Path::new("/dev/tpmrm0").exists() {
                    DeviceStatus::Available
                } else {
                    DeviceStatus::Unavailable
                },
                password_fallback: PasswordFallbackPolicy::Mandatory,
                hpd_role: HpdRole::HintOnly,
            };
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Serve => {
            info!(
                "service start refused: authenticated IPC and template storage are not implemented"
            );
            anyhow::bail!("faceauth-daemon is not production-ready")
        }
    }
    Ok(())
}

fn discover_cameras() -> Vec<CameraDevice> {
    let root = Path::new("/sys/class/video4linux");
    let Ok(entries) = fs::read_dir(root) else {
        return Vec::new();
    };
    let mut cameras = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let node = entry.file_name().to_string_lossy().into_owned();
            let name = fs::read_to_string(entry.path().join("name")).ok()?;
            Some(CameraDevice { device: format!("/dev/{node}"), name: name.trim().to_owned() })
        })
        .collect::<Vec<_>>();
    cameras.sort_by(|left, right| left.device.cmp(&right.device));
    cameras
}
