# Gesture Recognition System

Real-time hand gesture recognition system using MediaPipe and OpenCV.

## Features
- Tracks up to 2 hands simultaneously
- 21 landmarks per hand for detailed hand pose tracking
- Real-time performance (30+ FPS on most systems)
- Color-coded left (blue) and right (red) hand visualization
- FPS counter for performance monitoring

## Installation

### 1. Model File
The hand landmark model (`hand_landmarker.task`) should already be downloaded. If not, download it from:
```
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```
Save it in the `Gesture Recognition` folder.

### 2. Activate Virtual Environment

**Windows:**
```bash
cd "Gesture Recognition"
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
cd "Gesture Recognition"
source .venv/bin/activate
```

### 3. Verify Installation
```bash
python -c "import cv2; import mediapipe; from mediapipe.tasks.python import vision; print('All packages installed successfully!')"
```

## Usage

### Run the Application
```bash
python main.py
```

### Controls
- **Q** or **ESC** - Quit the application
- Show your hands to the camera for tracking

## System Requirements
- Python 3.13+
- Webcam (laptop or USB)
- Windows/Linux/Mac OS

## Performance
- **Target FPS**: 30+
- **Resolution**: 1280x720 (or your webcam's maximum)
- **Latency**: <100ms
- **CPU Usage**: <50% on modern laptops

## How It Works
1. Captures video from webcam at high resolution (1280x720)
2. Converts frames to RGB and creates MediaPipe Image objects
3. Processes frames asynchronously with MediaPipe HandLandmarker (LIVE_STREAM mode)
4. Receives detection results via callback function
5. Detects up to 2 hands and extracts 21 landmarks per hand
6. Visualizes hand skeleton with connections (manually drawn)
7. Displays real-time FPS counter

## Technical Details
- **API Version**: MediaPipe Tasks API 0.10.31 (new task-based architecture)
- **Running Mode**: LIVE_STREAM with async callback
- **Model**: hand_landmarker.task (float16 precision, ~7.5 MB)
- **Landmarks**: 21 points per hand (x, y, z coordinates normalized 0-1)
- **Handedness Detection**: Automatically identifies left vs right hand

## Next Steps
This system provides the foundation for gesture recognition. Future enhancements:
- Custom gesture classification (e.g., fist, open palm, peace sign)
- Gesture-to-game-control mapping
- Dynamic gesture recognition (swipes, circles, waves)
- Gesture recording and playback

## Troubleshooting

**Camera not opening:**
- Ensure no other application is using the webcam
- Check camera permissions in Windows settings
- Try changing `CAMERA_INDEX` in main.py (0, 1, 2, etc.)

**Low FPS:**
- Reduce resolution by changing `DESIRED_WIDTH` and `DESIRED_HEIGHT` in main.py
- Ensure good lighting conditions
- Close other resource-intensive applications

**Hands not detected:**
- Ensure good lighting
- Keep hands within camera frame
- Try adjusting `MIN_DETECTION_CONFIDENCE` (lower = more sensitive)
