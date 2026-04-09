# Gesture Recognition System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.31-green?logo=google&logoColor=white)](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10.0-red?logo=opencv&logoColor=white)](https://opencv.org/)
[![PyQt6](https://img.shields.io/badge/PyQt6-%E2%89%A56.5-41CD52?logo=qt&logoColor=white)](https://pypi.org/project/PyQt6/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows-0078D6?logo=windows&logoColor=white)](https://www.microsoft.com/windows)

A real-time hand gesture recognition system that translates webcam hand movements and gestures into keyboard and mouse inputs. Built with MediaPipe and OpenCV, it enables hands-free control of games or any application — no additional hardware required.

---

## Table of Contents

- [Demo](#demo)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [How It Works](#how-it-works)
- [Controls](#controls)
- [Supported Key Bindings](#supported-key-bindings)
- [GUI Settings](#gui-settings)
- [Configuration File](#configuration-file)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)

---

## Demo

> Work in progress!

---

## Features

- **Dual-Hand Control** — Left hand for WASD directional movement, right hand for 4 distinct gesture-triggered actions, both operating simultaneously
- **4 Right-Hand Gestures** — Pinch, Thumbs Up, Point, and Flat Palm, each mapped to a configurable key or mouse button
- **Full Keyboard & Mouse Binding** — Bind any gesture or direction to any key on a standard keyboard (F1–F12, arrows, modifiers, navigation keys, etc.) or mouse button (left, right, middle)
- **Gesture Debouncing** — Confirmation-frame system prevents flickering: 3 frames to activate, 5 frames to release
- **Movement Hysteresis** — Separate activate/release thresholds prevent WASD key chatter near the movement boundary
- **Picture-in-Picture Overlay** — Frameless, always-on-top, draggable camera feed window that stays visible over any application
- **PyQt6 Configuration GUI** — 5-tab settings interface for camera, thresholds, key bindings, display, and colours
- **Headless Mode** — Run from the terminal with an OpenCV window for lightweight usage
- **Async Detection** — Non-blocking MediaPipe pipeline with result caching eliminates landmark flickering
- **Immediate Key Release** — All held keys are released instantly when a hand leaves the camera frame

---

## Requirements

- Python 3.8+
- A working webcam
- Windows OS (required for the Picture-in-Picture / always-on-top overlay)

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/LLBahamut/Final-Year-Project.git
cd "Final-Year-Project"
```

2. **Create and activate a virtual environment**

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Running the Application

### GUI Mode (Recommended)

Launches the settings window. Configure everything visually, then click **Start**.

```bash
python gui.py
```

### Headless Mode

Runs the detection pipeline directly in a terminal + OpenCV window, using saved settings from `config.json` (or defaults if none exist).

```bash
python main.py
```

---

## How It Works

```
Webcam → Frame Capture → MediaPipe HandLandmarker → Gesture Classification → Key/Mouse Injection
                              (async, 21 3D landmarks per hand)
```

### Pipeline

1. **Capture** — Frames are captured from the webcam at up to 1920x1080, horizontally flipped for a natural mirror view, and capped at 30 fps.
2. **Detect** — Each frame is queued asynchronously to MediaPipe's `HandLandmarker`, which returns up to 2 sets of 21 3D landmarks per frame via a background callback thread.
3. **Cache** — The latest valid detection result is cached so landmarks are drawn on every frame, even when the async callback hasn't fired yet. This eliminates landmark flickering.
4. **Classify** — Hands are identified as left or right (closest to camera by Z-depth). The left hand controls WASD movement; the right hand is classified into one of four gestures.
5. **Inject** — Active keys or mouse buttons are sent to the OS in real time via `pynput`.
6. **Render** — The annotated camera feed (hand skeletons, gesture labels, WASD overlay, direction arrows) is displayed in the PiP overlay or OpenCV window.

### Left Hand — Movement Control

1. **Palm opens** → Reference point locked at palm centre
2. **Hand moves** → Displacement compared to reference; beyond `movement_threshold_activate` triggers a directional key
3. **Hysteresis** → Key releases only when displacement drops below `movement_threshold_release` (lower threshold prevents chatter)
4. **Palm closes** → All WASD keys released, reference point cleared
5. **Hand lost** → All keys released immediately

### Right Hand — Gesture Control

1. **Classify** → Priority order: Thumbs Up → Point → Pinch → Flat Palm
2. **Confirm** → New gesture must persist for `gesture_confirm_frames` (3) consecutive frames before activating
3. **Hold** → Mapped key/mouse button is held while gesture is maintained
4. **Release** → Gesture must be absent for `gesture_release_frames` (5) consecutive frames before releasing (resistant to accidental drops)
5. **Hand lost** → Held key released immediately

---

## Controls

### Left Hand — Movement

| Gesture | Default Key | Action |
|---|---|---|
| Open palm | — | Activate control and lock reference point |
| Move hand up | `W` | Move forward |
| Move hand down | `S` | Move backward |
| Move hand left | `A` | Move left |
| Move hand right | `D` | Move right |
| Diagonal movement | Two keys | Two directional keys pressed simultaneously |
| Close palm | — | Release all keys and clear reference point |

### Right Hand — Gestures

| Gesture | Default Key | Description |
|---|---|---|
| Pinch | `Z` | Thumb tip and index tip close together |
| Thumbs Up | `X` | Thumb extended upward, all other fingers curled |
| Point | `Q` | Index finger extended, others curled |
| Flat Palm | `C` | Open palm (3+ fingers extended) |

> All gesture-to-key bindings are fully remappable in the GUI or via `config.json`. You can also bind mouse clicks (left, right, middle).

### Headless Mode Keyboard Shortcuts

These shortcuts apply only when running `main.py` (the OpenCV window must be in focus).

| Key | Action |
|---|---|
| `B` | Quit the application |
| `R` | Reset reference point and release all keys |

---

## Supported Key Bindings

Any gesture or direction can be bound to:

| Category | Keys |
|---|---|
| **Letters** | `a`–`z` |
| **Numbers** | `0`–`9` |
| **Punctuation** | All printable single characters |
| **Function keys** | `f1`–`f12` |
| **Arrow keys** | `up`, `down`, `left`, `right` |
| **Navigation** | `home`, `end`, `page_up`, `page_down`, `insert`, `delete` |
| **Modifiers** | `shift`, `ctrl`, `alt`, `caps_lock`, `cmd` |
| **Common keys** | `space`, `enter`, `tab`, `backspace`, `esc` |
| **Lock / Special** | `num_lock`, `scroll_lock`, `pause`, `print_screen`, `menu` |
| **Mouse buttons** | `mouse_left`, `mouse_right`, `mouse_middle` |

In the GUI, click a key binding button then press any key or click a mouse button to bind it. Escape cancels.

---

## GUI Settings

Launching `gui.py` opens a settings window with five tabs:

| Tab | What you can configure |
|---|---|
| **Camera & Detection** | Camera index, resolution (up to 3840x2160), MediaPipe confidence thresholds (detection, presence, tracking) |
| **Gesture Thresholds** | Palm detection sensitivity, movement activate/release thresholds, hand loss grace period, hand proximity, pinch distance, finger curl ratio |
| **Keys & Controls** | Enable/disable actual keypresses and debug output; remap all WASD keys and right-hand gesture keys (keyboard or mouse) |
| **Display & Overlay** | PiP scale factor, WASD overlay toggle, key box size/spacing, overlay position |
| **Colors** | Hand skeleton colours (left/right), WASD key active/inactive colours, label text colours |

**Controls:**
- **Start / Stop** — Toggle gesture detection and PiP overlay
- **Restore Defaults** — Reset all fields to default values without stopping a running session

Settings are not auto-saved to disk. Use `config.json` to persist them between runs.

---

## Configuration File

Settings can be persisted by saving a `config.json` in the project root. `main.py` automatically loads it on startup if present.

```python
from config import GestureConfig
GestureConfig().to_json("config.json")        # write defaults
cfg = GestureConfig.from_json("config.json")  # load back
```

### Full Config Reference

<details>
<summary>Click to expand all configuration fields</summary>

#### Camera

| Field | Default | Description |
|---|---|---|
| `camera_index` | `0` | Index of the webcam to use |
| `desired_width` | `1920` | Camera capture width in pixels |
| `desired_height` | `1080` | Camera capture height in pixels |

#### MediaPipe Detection

| Field | Default | Description |
|---|---|---|
| `min_hand_detection_confidence` | `0.7` | Minimum confidence for initial hand detection |
| `min_hand_presence_confidence` | `0.5` | Minimum confidence to confirm hand presence |
| `min_tracking_confidence` | `0.5` | Minimum confidence for landmark tracking |

#### Palm & Movement

| Field | Default | Description |
|---|---|---|
| `palm_extension_threshold` | `1.1` | Tip-to-wrist / base-to-wrist ratio for a finger to count as extended |
| `palm_min_fingers` | `3` | Minimum extended fingers to classify as an open palm |
| `movement_threshold_activate` | `0.12` | Normalised displacement from reference to press a directional key |
| `movement_threshold_release` | `0.08` | Normalised displacement to release a directional key (hysteresis) |

#### Hand Tracking

| Field | Default | Description |
|---|---|---|
| `hand_loss_grace_period` | `2.0` | Seconds before control resets when hand disappears |
| `hand_proximity_threshold` | `0.2` | Max 3D distance to identify a re-appearing hand as the same one |

#### Keyboard & Control

| Field | Default | Description |
|---|---|---|
| `enable_actual_keypresses` | `true` | `false` = print keys to console only (useful for testing) |
| `enable_debug_output` | `true` | `false` = suppress press/release log lines |
| `key_forward` | `"w"` | Key sent when moving hand up |
| `key_backward` | `"s"` | Key sent when moving hand down |
| `key_left` | `"a"` | Key sent when moving hand left |
| `key_right` | `"d"` | Key sent when moving hand right |

#### Right-Hand Gestures

| Field | Default | Description |
|---|---|---|
| `enable_right_hand_gestures` | `true` | Enable/disable right-hand gesture detection |
| `gesture_pinch_key` | `"z"` | Key sent when pinch is detected |
| `gesture_thumbsup_key` | `"x"` | Key sent when thumbs-up is detected |
| `gesture_palm_key` | `"c"` | Key sent when flat palm is detected |
| `gesture_point_key` | `"q"` | Key sent when point gesture is detected |
| `pinch_distance_threshold` | `0.06` | Max 3D distance between thumb and index tips |
| `finger_curl_max_ratio` | `0.95` | Max tip/base ratio for a finger to be curled |
| `thumb_extended_min_ratio` | `1.2` | Stricter thumb extension ratio for thumbs-up |
| `thumbs_up_y_margin` | `0.02` | Thumb must be this far above MCP in y-axis |
| `thumbs_up_min_thumb_openness` | `0.10` | Thumb must be this far from index MCP |

#### Gesture Debouncing

| Field | Default | Description |
|---|---|---|
| `gesture_confirm_frames` | `3` | Consecutive frames required to confirm a new gesture |
| `gesture_release_frames` | `5` | Consecutive frames required to confirm gesture release |

#### Display & Overlay

| Field | Default | Description |
|---|---|---|
| `pip_scale` | `0.4` | PiP overlay scale relative to capture resolution |
| `wasd_overlay_enabled` | `true` | Show/hide the on-screen WASD key overlay |
| `wasd_key_size` | `50` | Pixel size of each key box |
| `wasd_key_spacing` | `10` | Gap in pixels between key boxes |
| `wasd_overlay_x` | `20` | X position from left edge |
| `wasd_overlay_y_offset` | `150` | Y offset from bottom edge |

#### Colors (BGR format)

| Field | Default | Description |
|---|---|---|
| `color_left_hand` | `[255, 0, 0]` | Left-hand skeleton colour |
| `color_right_hand` | `[0, 0, 255]` | Right-hand skeleton colour |
| `wasd_key_color_inactive` | `[60, 60, 60]` | Key box colour when not pressed |
| `wasd_key_color_active` | `[0, 255, 0]` | Key box colour when pressed |
| `wasd_text_color_inactive` | `[180, 180, 180]` | Key label colour when not pressed |
| `wasd_text_color_active` | `[0, 0, 0]` | Key label colour when pressed |

</details>

---

## Project Structure

```
Final Year Project/
├── main.py                 # Headless entry point — OpenCV window, terminal output
├── gui.py                  # PyQt6 settings window + entry point
├── processor.py            # GestureProcessor — detection, gesture logic, rendering
├── pip_overlay.py          # CameraWorker (QThread) + PipOverlay (frameless Qt window)
├── config.py               # GestureConfig dataclass with JSON load/save
├── config.json             # Saved user settings (auto-generated, not tracked by git)
├── hand_landmarker.task    # Pre-trained MediaPipe hand landmark model (~7.5 MB)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Tech Stack

| Library | Version | Role |
|---|---|---|
| [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) | 0.10.31 | Hand landmark detection — 21 3D points per hand, async LIVE_STREAM mode |
| [OpenCV](https://opencv.org/) | 4.10.0 | Video capture, frame processing, on-screen annotation |
| [NumPy](https://numpy.org/) | >= 1.24 | Array operations |
| [pynput](https://pynput.readthedocs.io/) | latest | OS-level keyboard and mouse input injection |
| [PyQt6](https://pypi.org/project/PyQt6/) | >= 6.5 | Settings GUI, PiP overlay window, camera worker thread |