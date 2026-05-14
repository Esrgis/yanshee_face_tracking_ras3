# Session Notes - 2026-05-14

## Context

Project: `yanshee_face_tracking_ras3`

Robot: Yanshee on Raspberry Pi. Production entrypoint is `main_tracker_robot.py`.

Pipeline chot: Haar/MOSSE/Kalman/PID/adaptive scheduler. The goal is live face tracking on robot while a human watches the robot camera stream remotely, without local OpenCV UI causing lag.

Robot real IP confirmed by user:

```text
192.168.31.239
```

## Problems Investigated

1. Running `main_tracker_robot.py` with local UI on robot caused heavy lag.
2. Robot head initially appeared to rotate in the wrong direction or not physically move.
3. `python` on Pi showed Python 2-style errors even though the target runtime is Python 3.5; we made import/encoding compatibility fixes where encountered.
4. MOSSE sometimes drifted to background/table and stayed locked.
5. Bbox was too large, increasing drift.
6. Tracking became smoother after tuning, but the target is still lost during abrupt human movement and search recovery can be slow.

## Files Changed

### Runtime / Streaming

- `main_tracker_robot.py`
  - Added `--stream` and `--stream-port`.
  - Uses `FrameStreamServer` to send processed frames to `client_monitor.py`.
  - Local OpenCV UI is now only enabled by `--ui`.
  - Added latency logging:

```text
[latency] cap=...ms vision=...ms stream=...ms loop=...ms
```

- `hardware/stream_server.py`
  - New file.
  - Streams latest annotated frame over TCP port 5555.
  - Can receive `G:<clip>` command from monitor and record video without opening camera again.

- `hardware/server_record.py`
  - Fixed missing `fourcc`.

### Package Compatibility

- `core/__init__.py`
- `hardware/__init__.py`
  - Added package markers for import compatibility.

- `core/vision.py`
  - Added UTF-8 header.
  - Replaced `class VisionSystem(ABC)` with `class VisionSystem(object)` because Pi raised:

```text
ImportError: cannot import name ABC
```

### Yanshee Hardware

- `hardware/yanshee_interface.py`
  - Added config-driven servo parameters:
    - `servo_direction`
    - `servo_center`
    - `servo_min_abs`
    - `servo_max_abs`
    - `min_command_interval_sec`
    - `servo_duration_ms`
    - `min_angle_step`
    - hardware debug response logging
  - Uses `target_angle` incremental control.
  - Skips micro commands if target angle change is less than `min_angle_step`.
  - Tracks YanAPI response with `get_last_response()`.

- `hardware/yanshee_servo_test.py`
  - New test script for direct servo testing.
  - Supports `--sweep`.

Useful commands:

```bash
python3 hardware/yanshee_servo_test.py \
  --ip 192.168.31.239 \
  --servo NeckLR \
  --duration 1200 \
  --sweep
```

```bash
python3 hardware/yanshee_servo_test.py \
  --ip 192.168.31.239 \
  --servo NeckLR \
  --angles 90,45,135,90 \
  --duration 1200 \
  --list-api
```

Observed API response:

```text
{'msg': 'success', 'data': {'NeckLR': True}, 'code': 0}
```

### Vision Drift Fixes

- `core/vision_haarcascade.py`
- `core/vision_lbp.py`

Added detector confirmation:

- MOSSE tracker cannot keep reporting success forever if Haar/LBP does not confirm a face.
- `max_tracker_confirm_misses=2` means after 2 failed detector confirmations, tracker resets and state should go SEARCH.
- `_reset_tracker()` now clears tracker, bbox, last center, and confirmation miss count.

### Config

Current important `config.json` values:

```json
{
  "hardware": {
    "robot_ip": "192.168.31.239",
    "use_multithreading": true,
    "servo_direction": 1,
    "min_command_interval_sec": 0.25,
    "servo_duration_ms": 250,
    "min_angle_step": 2.0,
    "debug_hardware": false
  },
  "vision": {
    "active_model": "haarcascade",
    "frame_width": 320,
    "frame_height": 240,
    "pad_ratio": 0.12,
    "iou_reinit_threshold": 0.6
  },
  "vision_params": {
    "min_size": 40,
    "max_size": 400,
    "min_neighbors_haar": 6,
    "min_neighbors_lbp": 5,
    "scale_factor_haar": 1.08,
    "scale_factor_lbp": 1.05,
    "max_tracker_confirm_misses": 2
  },
  "adaptive_scheduler": {
    "active_config": "D",
    "configs": {
      "D": {"base_skip": 2, "alpha": 0.10, "beta": 0.25}
    },
    "min_skip": 1,
    "max_skip": 6
  },
  "robot_yanshee": {
    "servo_min_abs": 30,
    "servo_max_abs": 150,
    "servo_center": 90
  },
  "controller_pid": {
    "Kp": 0.025,
    "Ki": 0.0,
    "Kd": 0.004,
    "max_integral": 20.0,
    "deadzone": 6.0,
    "output_limit": 4.0
  }
}
```

## Servo Mapping

Confirmed by physical test while user looked into robot's eyes:

- `90` = center.
- `90 -> 45` = robot turns to user's right, which is robot's left.
- `90 -> 135` = robot turns to user's left, which is robot's right.
- `15` = deep to user's right / robot's left.
- `165` = deep to user's left / robot's right.

Production range was narrowed from `15..165` to:

```text
30..150
```

Reason: avoid hitting deep limits during tracking lag or drift.

`servo_direction=1` is currently correct.

## Runtime Commands

Run robot tracking:

```bash
python3 main_tracker_robot.py --real --stream
```

Run laptop monitor:

```bash
python hardware/client_monitor.py --ip 192.168.31.239
```

Run with robot local UI only for debug:

```bash
python3 main_tracker_robot.py --real --stream --ui
```

## Important Observations

Latency sample after streaming integration:

```text
[latency] cap=1.3ms vision=7.8ms stream=1.6ms loop=13.2ms
[latency] cap=0.8ms vision=7.3ms stream=2.6ms loop=12.6ms
```

Interpretation:

- Camera read is not the bottleneck.
- Vision at current 320x240 is not the bottleneck.
- Stream is not the bottleneck.
- Occasional low FPS was likely due to frequent YanAPI/servo commands and micro corrections. Mitigated with `min_command_interval_sec`, `servo_duration_ms`, and `min_angle_step`.

Tracking improved after:

- changing to Haar,
- lowering detection skip,
- adding detector confirmation,
- reducing bbox padding,
- narrowing servo range,
- lowering PID output.

Remaining issue:

- If the human moves abruptly, target can still leave bbox and Haar can take noticeable time to reacquire.
- This is likely near the practical limit of Haar + MOSSE on fast movement/blur.

## Current Assessment

The pipeline is now much more stable for normal/slow movement, but abrupt movement still exposes limits:

- MOSSE is fast but can drift or lose target when scale/pose/blur changes abruptly.
- Haar is more reliable than LBP but weak for motion blur, face angles, and partial faces.
- Search recovery speed is limited by detector robustness, not just scheduler speed.

## Recommended Next Step

Add SSD fallback only in `SEARCH`/`LOST` states:

- Keep Haar/MOSSE/Kalman/PID/adaptive scheduler as the main pipeline.
- When target is lost or SEARCH lasts more than a short threshold, run SSD OpenCV DNN as a stronger reacquisition detector.
- Once SSD finds face, reinitialize MOSSE and return to normal Haar/MOSSE tracking.

This preserves the agreed pipeline for normal tracking while improving recovery.

Alternative smaller next steps:

1. Add servo scan behavior in SEARCH: sweep slightly around current angle while detector runs every frame.
2. Tune `pad_ratio` between `0.08` and `0.15`.
3. Tune `max_tracker_confirm_misses` between `1` and `3`.
4. Improve lighting/exposure to reduce motion blur.

## Notes For Next Session

If continuing from this file:

1. Inspect `git diff` first because several files were edited.
2. Do not assume `.235`; real robot IP is `.239`.
3. Servo direction is already confirmed: `servo_direction=1`.
4. User is concerned primarily with demo responsiveness and reacquisition after losing face.
5. Best next engineering move is SSD fallback for SEARCH/LOST or SEARCH servo sweep.
