# Gesture Recognition System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-orange)
![PyQt6](https://img.shields.io/badge/PyQt6-6.5.0-red)


A real-time hand gesture recognition system that translates webcam hand movements into WASD keyboard inputs. Built with MediaPipe and OpenCV, it enables hands-free control of games or any application that uses WASD movement keys.

---

## Demo

> Work in progress!

---

## Requirements

- Python 3.8+
- A working webcam
- Windows OS (required for Picture-in-Picture / always-on-top functionality)

---

## Installation

1. Clone the repository
```bash
git clone https://github.com/LLBahamut/Final-Year-Project.git
cd "Final-Year-Project"
```

2. Create and activate a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
python main.py
```

---

## How It Works

1. **Capture:** Frames are read from the webcam at 1280×720 and horizontally flipped for a natural mirror view.
2. **Detect:** Each frame is passed asynchronously to MediaPipe's `HandLandmarker`, which returns up to 2 sets of 21 3D landmarks.
3. **Palm State:** The system checks whether at least 3 fingers are extended beyond their base joints. If so, the hand is considered an open palm and control is activated.
4. **Reference Lock:** When the palm first opens, the current palm-centre position is saved as a reference point.
5. **Movement Mapping:** Subsequent palm-centre positions are compared to the reference. Displacement beyond the activation threshold (`MOVEMENT_THRESHOLD_ACTIVATE`) triggers the corresponding WASD key; displacement below the release threshold (`MOVEMENT_THRESHOLD_RELEASE`) releases it. This hysteresis prevents key jitter.
6. **Key Injection:** Active keys are sent to the OS via `pynput`.

---

## Controls

### Gesture Controls

| Gesture | Action |
|---|---|
| Open palm (left hand) | Activate keyboard control; lock reference point |
| Move hand up | Press **W** (forward) |
| Move hand down | Press **S** (backward) |
| Move hand left | Press **A** (left) |
| Move hand right | Press **D** (right) |
| Diagonal movement | Press two keys simultaneously |
| Close palm | Release all keys |

### Keyboard Shortcuts

| Key | Action |
|---|---|
| `Q` / `Esc` | Quit the application |
| `R` | Reset reference point and release all keys |
| `P` | Toggle Picture-in-Picture (always-on-top) mode |

---

## Configuration

All tunable constants are defined at the top of [main.py](main.py).

| Constant | Default | Description |
|---|---|---|
| `CAMERA_INDEX` | `0` | Index of the webcam to use |
| `DESIRED_WIDTH` | `1280` | Camera capture width in pixels |
| `DESIRED_HEIGHT` | `720` | Camera capture height in pixels |
| `MIN_HAND_DETECTION_CONFIDENCE` | `0.7` | Minimum confidence for initial hand detection |
| `MIN_HAND_PRESENCE_CONFIDENCE` | `0.5` | Minimum confidence to confirm hand presence |
| `MIN_TRACKING_CONFIDENCE` | `0.5` | Minimum confidence for landmark tracking |
| `PALM_EXTENSION_THRESHOLD` | `1.1` | Ratio of tip-to-wrist vs. base-to-wrist distance required to count a finger as extended |
| `PALM_MIN_FINGERS` | `3` | Number of extended fingers required to count as an open palm |
| `MOVEMENT_THRESHOLD_ACTIVATE` | `0.12` | Normalised displacement from reference to activate a key |
| `MOVEMENT_THRESHOLD_RELEASE` | `0.08` | Normalised displacement from reference to release a key (hysteresis) |
| `HAND_LOSS_GRACE_PERIOD` | `2.0` | Seconds before tracking state resets when the hand leaves view |
| `HAND_PROXIMITY_THRESHOLD` | `0.2` | Maximum 3D distance (normalised) to identify a returning hand as the same hand |
| `ENABLE_ACTUAL_KEYPRESSES` | `True` | Set to `False` to print keys instead of sending them (useful for testing) |
| `ENABLE_DEBUG_OUTPUT` | `True` | Set to `False` to suppress debug console output |
| `KEY_MAPPING` | `w/s/a/d` | Dict mapping `"forward"`, `"backward"`, `"left"`, `"right"` to key strings |
| `PIP_SCALE` | `0.4` | Window scale factor in Picture-in-Picture mode (40% of original) |
| `WASD_OVERLAY_ENABLED` | `True` | Show/hide the on-screen WASD key overlay |
| `WASD_KEY_SIZE` | `50` | Pixel size of each key box in the overlay |

---

## Project Structure

```
Final Year Project/
├── main.py                  # Main application — detection pipeline, gesture logic, rendering
├── hand_landmarker.task     # Pre-trained MediaPipe hand landmark model (7.8 MB)
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## Tech Stack

| Library | Version | Role |
|---|---|---|
| [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) | 0.10.31 | Hand landmark detection (21 points per hand) |
| [OpenCV](https://opencv.org/) | 4.10.0 | Video capture, frame processing, on-screen rendering |
| [NumPy](https://numpy.org/) | ≥1.24 | Numerical operations and distance calculations |
| [pynput](https://pynput.readthedocs.io/) | latest | OS-level keyboard input injection |
| [PyQt6](https://pypi.org/project/PyQt6/) | latest | GUI rendering |