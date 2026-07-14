//! Stable Linux camera discovery for the face authentication service.

use std::{ffi::OsStr, io, path::PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// USB identity used to bind configuration to one camera device.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct UsbIdentity {
    /// Four-character lowercase hexadecimal USB vendor identifier.
    pub vendor_id: String,
    /// Four-character lowercase hexadecimal USB product identifier.
    pub product_id: String,
    /// Manufacturer-provided serial string, when available.
    pub serial: Option<String>,
}

/// One V4L2 node discovered through udev.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CameraDevice {
    /// Device node, such as `/dev/video0`.
    pub node: PathBuf,
    /// Kernel-reported camera name.
    pub name: String,
    /// Stable USB identity, when the node belongs to a USB camera.
    pub usb: Option<UsbIdentity>,
    /// Stable physical path reported by udev.
    pub physical_path: Option<String>,
    /// Whether udev identifies this node as video-capture capable.
    pub capture_capable: bool,
}

impl CameraDevice {
    /// Return whether this device matches an explicit USB camera selector.
    #[must_use]
    pub fn matches(&self, selector: &CameraSelector) -> bool {
        let Some(usb) = &self.usb else {
            return false;
        };

        usb.vendor_id.eq_ignore_ascii_case(&selector.vendor_id)
            && usb.product_id.eq_ignore_ascii_case(&selector.product_id)
            && selector.serial.as_ref().is_none_or(|expected| usb.serial.as_ref() == Some(expected))
            && selector
                .physical_path
                .as_ref()
                .is_none_or(|expected| self.physical_path.as_ref() == Some(expected))
    }
}

/// Explicit selector stored in administrator-controlled camera configuration.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CameraSelector {
    /// Required USB vendor identifier.
    pub vendor_id: String,
    /// Required USB product identifier.
    pub product_id: String,
    /// Optional serial constraint for otherwise-identical devices.
    pub serial: Option<String>,
    /// Optional physical-path constraint when a serial is unavailable or duplicated.
    pub physical_path: Option<String>,
}

/// Camera inventory discovery failure.
#[derive(Debug, Error)]
pub enum DiscoveryError {
    /// libudev could not create or scan the video4linux enumerator.
    #[error("unable to enumerate video4linux devices: {0}")]
    Udev(#[from] io::Error),
}

/// Discover all V4L2 nodes and return them in deterministic device-node order.
///
/// # Errors
///
/// Returns [`DiscoveryError`] when libudev cannot scan the `video4linux` subsystem.
pub fn discover() -> Result<Vec<CameraDevice>, DiscoveryError> {
    let mut enumerator = udev::Enumerator::new()?;
    enumerator.match_subsystem("video4linux")?;

    let mut devices = enumerator
        .scan_devices()?
        .filter_map(|device| camera_from_udev(&device))
        .collect::<Vec<_>>();
    devices.sort_by(|left, right| left.node.cmp(&right.node));
    Ok(devices)
}

fn camera_from_udev(device: &udev::Device) -> Option<CameraDevice> {
    let node = device.devnode()?.to_path_buf();
    let name = device
        .attribute_value("name")
        .and_then(OsStr::to_str)
        .unwrap_or("unknown video device")
        .to_owned();
    let property = |key| device.property_value(key).and_then(OsStr::to_str).map(str::to_owned);
    let vendor_id = property("ID_VENDOR_ID");
    let product_id = property("ID_MODEL_ID");
    let usb = vendor_id.zip(product_id).map(|(vendor_id, product_id)| UsbIdentity {
        vendor_id,
        product_id,
        serial: property("ID_SERIAL_SHORT").or_else(|| property("ID_SERIAL")),
    });
    let capabilities = property("ID_V4L_CAPABILITIES").unwrap_or_default();

    Some(CameraDevice {
        node,
        name,
        usb,
        physical_path: property("ID_PATH"),
        capture_capable: capabilities.split(':').any(|capability| capability == "capture"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera() -> CameraDevice {
        CameraDevice {
            node: PathBuf::from("/dev/video0"),
            name: "Integrated IR Camera".to_owned(),
            usb: Some(UsbIdentity {
                vendor_id: "04f2".to_owned(),
                product_id: "b769".to_owned(),
                serial: Some("01.00.00".to_owned()),
            }),
            physical_path: Some("pci-0000:05:00.0-usb-0:1:1.0".to_owned()),
            capture_capable: true,
        }
    }

    #[test]
    fn selector_can_bind_usb_identity_and_location() {
        let selector = CameraSelector {
            vendor_id: "04F2".to_owned(),
            product_id: "B769".to_owned(),
            serial: Some("01.00.00".to_owned()),
            physical_path: Some("pci-0000:05:00.0-usb-0:1:1.0".to_owned()),
        };

        assert!(camera().matches(&selector));
    }

    #[test]
    fn selector_rejects_a_different_physical_device() {
        let selector = CameraSelector {
            vendor_id: "04f2".to_owned(),
            product_id: "b769".to_owned(),
            serial: None,
            physical_path: Some("pci-0000:00:00.0-usb-0:9:1.0".to_owned()),
        };

        assert!(!camera().matches(&selector));
    }
}
