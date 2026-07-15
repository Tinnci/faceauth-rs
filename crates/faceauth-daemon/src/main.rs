//! Read-only diagnostics and the future privileged face-authentication daemon entry point.

use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use faceauth_camera::{CameraDevice, CameraPairSelector};
use faceauth_capture::{
    CaptureSpec, FrameSummary, PairingPolicy, PixelFormat, V4l2CaptureDevice, capture_pair,
};
use faceauth_core::CaptureModality;
use faceauth_storage::TpmKeyProvider;
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
    /// Exercise TPM sealing and unsealing without enrolling a face.
    StorageDoctor {
        /// Root-only path for the TPM public/private sealed-key blob.
        #[arg(long, default_value = "/var/lib/faceauth/machine-key.tpm")]
        blob: PathBuf,
    },
    /// Capture and pair one ephemeral IR/RGB observation without inference or enrollment.
    CaptureDoctor {
        /// Root-controlled stable USB and physical-path camera selectors.
        #[arg(long, default_value = "/etc/faceauth/cameras.json")]
        config: PathBuf,
    },
    /// Refuse to start the production service until the transport is implemented.
    Serve,
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
#[serde(rename_all = "kebab-case")]
enum StorageKeyPolicy {
    TpmPreferredFileFallbackExplicit,
}

#[derive(Debug, Serialize)]
struct DoctorReport {
    version: &'static str,
    production_status: ProductionStatus,
    cameras: Vec<CameraDevice>,
    tpm2_resource_manager: DeviceStatus,
    password_fallback: PasswordFallbackPolicy,
    hpd_role: HpdRole,
    storage_key_policy: StorageKeyPolicy,
}

#[derive(Debug, Serialize)]
struct CaptureDoctorReport<'a> {
    infrared_device: &'a CameraDevice,
    visible_device: &'a CameraDevice,
    infrared: &'a FrameSummary,
    visible: &'a FrameSummary,
    skew_micros: u64,
    raw_frames_retained: bool,
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
                cameras: faceauth_camera::discover()?
                    .into_iter()
                    .filter(|camera| camera.capture_capable)
                    .collect(),
                tpm2_resource_manager: if Path::new("/dev/tpmrm0").exists() {
                    DeviceStatus::Available
                } else {
                    DeviceStatus::Unavailable
                },
                password_fallback: PasswordFallbackPolicy::Mandatory,
                hpd_role: HpdRole::HintOnly,
                storage_key_policy: StorageKeyPolicy::TpmPreferredFileFallbackExplicit,
            };
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::StorageDoctor { blob } => {
            let provider = TpmKeyProvider::new(&blob);
            provider.self_test()?;
            println!("TPM sealed-key self-test passed for {}", blob.display());
        }
        Command::CaptureDoctor { config } => {
            let selector_bytes = fs::read(&config).with_context(|| {
                format!("unable to read camera selectors from {}", config.display())
            })?;
            let selectors: CameraPairSelector = serde_json::from_slice(&selector_bytes)
                .with_context(|| format!("invalid camera selector JSON in {}", config.display()))?;
            let inventory = faceauth_camera::discover()?;
            let cameras = faceauth_camera::resolve_pair(&inventory, &selectors)?;
            let infrared_device = V4l2CaptureDevice::open(
                &cameras.infrared.node,
                CaptureModality::Infrared,
                CaptureSpec {
                    width: 640,
                    height: 360,
                    pixel_format: PixelFormat::Gray8,
                    frames_per_second: 15,
                    buffer_count: 4,
                    warmup_frames: 2,
                    frame_timeout_millis: 1_500,
                    max_frame_bytes: 1024 * 1024,
                },
            )?;
            let visible_device = V4l2CaptureDevice::open(
                &cameras.visible.node,
                CaptureModality::Visible,
                CaptureSpec {
                    width: 640,
                    height: 360,
                    pixel_format: PixelFormat::Yuyv,
                    frames_per_second: 30,
                    buffer_count: 4,
                    warmup_frames: 2,
                    frame_timeout_millis: 1_500,
                    max_frame_bytes: 1024 * 1024,
                },
            )?;
            let mut infrared_stream = infrared_device.stream()?;
            let mut visible_stream = visible_device.stream()?;
            let pair = capture_pair(
                &mut infrared_stream,
                &mut visible_stream,
                PairingPolicy { max_skew_micros: 100_000, max_replacements: 8 },
            )?;
            let report = CaptureDoctorReport {
                infrared_device: &cameras.infrared,
                visible_device: &cameras.visible,
                infrared: pair.infrared.summary(),
                visible: pair.visible.summary(),
                skew_micros: pair
                    .infrared
                    .summary()
                    .timestamp_micros
                    .abs_diff(pair.visible.summary().timestamp_micros),
                raw_frames_retained: false,
            };
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Serve => {
            info!(
                "service start refused: inference, passive PAD, active-liveness integration, and enrollment are incomplete"
            );
            anyhow::bail!("faceauth-daemon is not production-ready")
        }
    }
    Ok(())
}
