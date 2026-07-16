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
use faceauth_daemon::{ReadinessReport, readiness_from_config_path};
use faceauth_presence::HpdClient;
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
    Doctor {
        /// Root-controlled versioned production configuration.
        #[arg(long, default_value = "/etc/faceauth/faceauth.json")]
        config: PathBuf,
    },
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
    /// Read the optional thinkpad-hpd presence hint; never used for authentication.
    PresenceDoctor,
    /// Evaluate all production gates and refuse startup while any remain unproven.
    Serve {
        /// Root-controlled versioned production configuration.
        #[arg(long, default_value = "/etc/faceauth/faceauth.json")]
        config: PathBuf,
    },
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
    TpmRequiredFileFallbackDiagnosticOnly,
}

#[derive(Debug, Serialize)]
struct DoctorReport {
    version: &'static str,
    readiness: ReadinessReport,
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

#[derive(Debug, Serialize)]
struct PresenceDoctorReport {
    available: bool,
    present: bool,
    raw_value: i32,
    authentication_evidence: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    match Cli::parse().command {
        Command::Doctor { config } => run_doctor(&config)?,
        Command::StorageDoctor { blob } => run_storage_doctor(&blob)?,
        Command::CaptureDoctor { config } => run_capture_doctor(&config)?,
        Command::PresenceDoctor => run_presence_doctor()?,
        Command::Serve { config } => run_serve(&config)?,
    }
    Ok(())
}

fn run_doctor(config: &Path) -> Result<()> {
    let report = DoctorReport {
        version: env!("CARGO_PKG_VERSION"),
        readiness: readiness_from_config_path(config),
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
        storage_key_policy: StorageKeyPolicy::TpmRequiredFileFallbackDiagnosticOnly,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn run_storage_doctor(blob: &Path) -> Result<()> {
    let provider = TpmKeyProvider::new(blob);
    provider.self_test()?;
    println!("TPM sealed-key self-test passed for {}", blob.display());
    Ok(())
}

fn run_capture_doctor(config: &Path) -> Result<()> {
    let selector_bytes = fs::read(config)
        .with_context(|| format!("unable to read camera selectors from {}", config.display()))?;
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
    Ok(())
}

fn run_serve(config: &Path) -> Result<()> {
    let readiness = readiness_from_config_path(config);
    if !readiness.production_ready {
        let blocking = readiness
            .blocking_gates()
            .into_iter()
            .map(|gate| format!("{gate:?}"))
            .collect::<Vec<_>>()
            .join(", ");
        info!(blocking_gates = %blocking, "service start refused by production readiness gates");
        anyhow::bail!("faceauth-daemon is not production-ready; blocking gates: {blocking}")
    }
    anyhow::bail!(
        "faceauth-daemon readiness contradiction: this build has no production service composition"
    )
}

fn run_presence_doctor() -> Result<()> {
    let snapshot = HpdClient::system()?.get_state(0)?;
    let report = PresenceDoctorReport {
        available: snapshot.available,
        present: snapshot.present,
        raw_value: snapshot.raw_value,
        authentication_evidence: false,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
