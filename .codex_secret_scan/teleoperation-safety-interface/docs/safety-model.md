# Safety Model

This prototype models a layered teleoperation safety policy. It is useful for demos, portfolio work, and interface design discussions, but it is not suitable for direct robot control.

## Gate states

- Standby: output is zero because the console is not armed.
- Output hold: output is zero because the deadman control is released.
- Limited: output is reduced because a warning condition is active.
- Safety hold: output is zero because at least one interlock failed.
- Emergency stop: output is zero until the stop is reset.
- Live: output follows the operator command within the configured speed limit.

## Interlocks

| Interlock | Warning | Stop |
| --- | --- | --- |
| Control link | Latency above 150 ms or quality below 70% | Latency above 220 ms or quality below 50% |
| Human proximity | Human range below 2.0 m | Human range below 1.2 m |
| Platform attitude | Tilt above 9 deg | Tilt above 14 deg |
| Battery reserve | Battery below 20% | Battery below 10% |
| Motor thermal | Motor temperature above 62 C | Motor temperature above 76 C |
| Geofence | Handover envelope command exceeds reduced range | Not modeled as a hard stop |

## Production considerations

For a real robot, the browser should never be the only safety mechanism. Production systems should include independent safety enforcement, authenticated command channels, watchdog timers, replay protection, operator authorization, audit logging, and formal hazard analysis.
