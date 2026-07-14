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

    /// Build the most specific persistent selector available for this USB camera.
    #[must_use]
    pub fn selector(&self) -> Option<CameraSelector> {
        let usb = self.usb.as_ref()?;
        Some(CameraSelector {
            vendor_id: usb.vendor_id.clone(),
            product_id: usb.product_id.clone(),
            serial: usb.serial.clone(),
            physical_path: self.physical_path.clone(),
        })
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

/// Explicit selectors for the two camera modalities.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CameraPairSelector {
    /// Selector for the near-infrared capture node.
    pub infrared: CameraSelector,
    /// Selector for the visible-light capture node.
    pub visible: CameraSelector,
}

/// Camera role used in selection errors and diagnostics.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CameraRole {
    /// Near-infrared camera.
    Infrared,
    /// Visible-light camera.
    Visible,
}

/// A uniquely resolved IR and visible-light capture pair.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ResolvedCameraPair {
    /// Resolved near-infrared capture node.
    pub infrared: CameraDevice,
    /// Resolved visible-light capture node.
    pub visible: CameraDevice,
}

/// Camera inventory discovery failure.
#[derive(Debug, Error)]
pub enum DiscoveryError {
    /// libudev could not create or scan the video4linux enumerator.
    #[error("unable to enumerate video4linux devices: {0}")]
    Udev(#[from] io::Error),
}

/// Failure to resolve explicit selectors to a unique capture pair.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum SelectionError {
    /// No capture-capable node matched a required selector.
    #[error("no capture-capable {role:?} camera matched the configured selector")]
    NotFound {
        /// Camera role that could not be resolved.
        role: CameraRole,
    },
    /// More than one capture-capable node matched a selector.
    #[error("{matches} capture-capable nodes matched the {role:?} camera selector")]
    Ambiguous {
        /// Camera role with ambiguous matches.
        role: CameraRole,
        /// Number of matching capture nodes.
        matches: usize,
    },
    /// Both selectors resolved to the same V4L2 node.
    #[error("infrared and visible selectors resolved to the same camera node")]
    SameDevice,
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

/// Resolve administrator-controlled selectors against a camera inventory.
///
/// Metadata-only V4L2 nodes are ignored. Each role must resolve to exactly one capture node and
/// the two roles must not resolve to the same node.
///
/// # Errors
///
/// Returns [`SelectionError`] when a role is missing or ambiguous, or both roles resolve to the
/// same node.
pub fn resolve_pair(
    devices: &[CameraDevice],
    selectors: &CameraPairSelector,
) -> Result<ResolvedCameraPair, SelectionError> {
    let infrared = resolve_one(devices, &selectors.infrared, CameraRole::Infrared)?;
    let visible = resolve_one(devices, &selectors.visible, CameraRole::Visible)?;
    if infrared.node == visible.node {
        return Err(SelectionError::SameDevice);
    }
    Ok(ResolvedCameraPair { infrared, visible })
}

fn resolve_one(
    devices: &[CameraDevice],
    selector: &CameraSelector,
    role: CameraRole,
) -> Result<CameraDevice, SelectionError> {
    let matches = devices
        .iter()
        .filter(|device| device.capture_capable && device.matches(selector))
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [] => Err(SelectionError::NotFound { role }),
        [device] => Ok((*device).clone()),
        _ => Err(SelectionError::Ambiguous { role, matches: matches.len() }),
    }
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

    fn visible_camera() -> CameraDevice {
        CameraDevice {
            node: PathBuf::from("/dev/video2"),
            name: "Integrated Camera".to_owned(),
            usb: Some(UsbIdentity {
                vendor_id: "04f2".to_owned(),
                product_id: "b768".to_owned(),
                serial: Some("01.00.00".to_owned()),
            }),
            physical_path: Some("pci-0000:04:00.3-usb-0:4:1.0".to_owned()),
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

    #[test]
    fn pair_resolution_ignores_metadata_only_nodes() -> Result<(), SelectionError> {
        let infrared = camera();
        let mut infrared_metadata = infrared.clone();
        infrared_metadata.node = PathBuf::from("/dev/video1");
        infrared_metadata.capture_capable = false;
        let visible = visible_camera();
        let selectors = CameraPairSelector {
            infrared: infrared
                .selector()
                .ok_or(SelectionError::NotFound { role: CameraRole::Infrared })?,
            visible: visible
                .selector()
                .ok_or(SelectionError::NotFound { role: CameraRole::Visible })?,
        };

        let pair =
            resolve_pair(&[infrared.clone(), infrared_metadata, visible.clone()], &selectors)?;

        assert_eq!(pair, ResolvedCameraPair { infrared, visible });
        Ok(())
    }

    #[test]
    fn ambiguous_capture_nodes_fail_closed() {
        let infrared = camera();
        let mut duplicate = infrared.clone();
        duplicate.node = PathBuf::from("/dev/video4");
        let selector = CameraSelector {
            vendor_id: "04f2".to_owned(),
            product_id: "b769".to_owned(),
            serial: None,
            physical_path: None,
        };
        let selectors = CameraPairSelector {
            infrared: selector,
            visible: visible_camera().selector().unwrap_or_else(|| CameraSelector {
                vendor_id: String::new(),
                product_id: String::new(),
                serial: None,
                physical_path: None,
            }),
        };

        assert_eq!(
            resolve_pair(&[infrared, duplicate, visible_camera()], &selectors),
            Err(SelectionError::Ambiguous { role: CameraRole::Infrared, matches: 2 })
        );
    }

    #[test]
    fn the_same_node_cannot_supply_both_modalities() {
        let infrared = camera();
        let selector = infrared.selector().unwrap_or_else(|| CameraSelector {
            vendor_id: String::new(),
            product_id: String::new(),
            serial: None,
            physical_path: None,
        });
        let selectors = CameraPairSelector { infrared: selector.clone(), visible: selector };

        assert_eq!(resolve_pair(&[infrared], &selectors), Err(SelectionError::SameDevice));
    }
}
