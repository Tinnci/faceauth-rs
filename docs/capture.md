# Camera capture boundary

`faceauth-capture` owns bounded V4L2 streaming and timestamp pairing. It requires
an exact resolution, pixel format, and frame rate; rejects corrupt frames,
non-monotonic kernel timestamps, empty or oversized payloads, and IR/RGB pairs
outside the configured skew and replacement budgets.

Frame bytes are copied only into an ephemeral `Zeroizing<Vec<u8>>`. Diagnostics
serialize metadata but never pixels, decoded images, embeddings, or templates.
Production preprocessing and inference must borrow these bytes and must not
persist them.

## Stable camera selection

Persisted configuration contains USB identity and optional serial and physical
path constraints, never `/dev/videoN` node numbers. The daemon discovers the
current udev inventory and fails closed unless each selector resolves to exactly
one capture-capable node and the two modalities resolve to different nodes.

Example `/etc/faceauth/cameras.json` for the initial ThinkPad hardware:

```json
{
  "infrared": {
    "vendor_id": "04f2",
    "product_id": "b769",
    "serial": null,
    "physical_path": "replace-with-ID_PATH-from-doctor"
  },
  "visible": {
    "vendor_id": "04f2",
    "product_id": "b768",
    "serial": null,
    "physical_path": "replace-with-ID_PATH-from-doctor"
  }
}
```

Generate the selectors from the capture-capable entries printed by `doctor`.
Keep this file root-owned and non-writable by unprivileged users. Then run the
ephemeral hardware diagnostic:

```bash
cargo run -p faceauth-daemon -- capture-doctor --config /etc/faceauth/cameras.json
```

This command captures one aligned IR/RGB observation, prints safe metadata, and
drops and zeroizes both raw frame buffers. It performs no enrollment or
authentication.
