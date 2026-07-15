# Active liveness boundary

Production policy combines passive presentation-attack detection with a
randomized active challenge. Neither mechanism is sufficient alone, and neither
can override missing IR, poor quality, or a failed identity comparison.

`faceauth-liveness` owns the parts of the active challenge that must not be left
to a landmark model or desktop client:

- unbiased operating-system random selection among blink, turn-left, and
  turn-right actions;
- a centered, eyes-open baseline before the action prompt;
- observation of the requested action followed by centered recovery;
- strictly increasing monotonic timestamps that do not predate challenge issue;
- fresh sequence numbers from both IR and visible cameras;
- exact one-face and dual-camera pairing requirements;
- a fixed total deadline and maximum observation budget.

The detector/landmark adapter supplies only face count, normalized aggregate eye
openness, and signed yaw measurements for each fresh paired observation. It does
not supply a `passed` flag. The daemon converts terminal state-machine success to
active-liveness evidence only after passive PAD and all other evidence have been
computed independently.

## Calibration and UI contract

The crate deliberately has no default thresholds. Eye-openness and yaw bounds
must be calibrated for the admitted landmark model, preprocessing contract, and
target camera hardware, then versioned with that model configuration. Changing a
model or preprocessing path invalidates those bounds.

Desktop clients and the future lock-screen OSD receive only progress prompts:
waiting for neutral baseline, perform the selected action, or return to neutral.
They never receive raw frames or landmark values and cannot report challenge
success back to the daemon. Cancellation and password fallback remain available
throughout the bounded transaction.

The state machine also exposes a session-independent cancellable observation
method. It checks the callback before validating or mutating state, so a cancelled
worker does not consume a frame sequence, advance a challenge phase, or retain
landmark-derived state. The daemon can map its private transaction token to this
callback without making the liveness crate depend on daemon/session types.

Active challenges raise replay cost but do not establish presentation-attack
resistance by themselves. Print, screen, video, mask, injection, and virtual
camera attacks still require passive PAD, device trust policy, and independent
evaluation on the exact hardware and model set.
