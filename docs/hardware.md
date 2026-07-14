# Target hardware baseline

Captured on a ThinkPad Z13 Gen 1 with Linux `7.1.3-2-cachyos`.

| Role | USB ID | Mode |
| --- | --- | --- |
| IR camera | `04f2:b769` | GREY 640x360 at 15 fps |
| Visible camera | `04f2:b768` | MJPEG/YUYV, including 640x360 at 30 fps |
| TPM | TPM 2.0 | `/dev/tpmrm0` present |

The IR device incorrectly advertises UVC 1.50 while returning a 26-byte probe
control. The local `uvcvideo-b769` DKMS module forces UVC 1.00 semantics for this
device and has been verified with full-size frame capture.

Device node numbers are not stable and must not appear in persisted policy.

