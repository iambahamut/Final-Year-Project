import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# Constants
CAMERA_INDEX = 0
DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720
MIN_HAND_DETECTION_CONFIDENCE = 0.7
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
MODEL_PATH = 'hand_landmarker.task'

# Open palm detection thresholds
PALM_EXTENSION_THRESHOLD = 1.1  # Finger tip must be farther from wrist than base
PALM_MIN_FINGERS = 3  # Minimum fingers extended to count as open palm

# Movement thresholds (normalized coordinates)
MOVEMENT_THRESHOLD_ACTIVATE = 0.12   # Distance from reference to activate key
MOVEMENT_THRESHOLD_RELEASE = 0.08    # Distance from reference to release key (hysteresis)

# Hand tracking loss grace period
HAND_LOSS_GRACE_PERIOD = 2.0  # seconds

# Colors (BGR format)
COLOR_LEFT_HAND = (255, 0, 0)   # Blue
COLOR_RIGHT_HAND = (0, 0, 255)  # Red
COLOR_FPS = (0, 255, 0)         # Green

# Hand connections (pairs of landmark indices to draw lines between)
HAND_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),      # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),    # Ring finger
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
])

# Global variable to store detection results from callback
detection_result = None
result_lock = False

# Keyboard control state
keyboard_controller = None
right_hand_state = {
    'is_palm_open': False,
    'reference_point': None,  # (x, y, z) when open palm first detected
    'active_keys': set(),      # Currently pressed keys
    'last_seen_time': None,    # Last time right hand was detected
    'control_active': False    # Whether keyboard control is active
}


def is_hand_open_palm(hand_landmarks):
    """
    Detect if hand is in open palm position by checking if fingers are extended.
    Returns True if at least 3 of 4 fingers (index, middle, ring, pinky) are extended.
    Thumb is ignored for relaxed palm detection.
    """
    wrist = hand_landmarks[0]
    fingers_extended = 0

    # Check each finger (excluding thumb)
    finger_tips = [8, 12, 16, 20]    # Index, Middle, Ring, Pinky tips
    finger_bases = [5, 9, 13, 17]     # Corresponding MCP joints

    for tip_idx, base_idx in zip(finger_tips, finger_bases):
        tip = hand_landmarks[tip_idx]
        base = hand_landmarks[base_idx]

        # Calculate distance from wrist
        tip_dist = ((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2 + (tip.z - wrist.z)**2)**0.5
        base_dist = ((base.x - wrist.x)**2 + (base.y - wrist.y)**2 + (base.z - wrist.z)**2)**0.5

        # Finger is extended if tip is farther from wrist than base
        if tip_dist > base_dist * PALM_EXTENSION_THRESHOLD:
            fingers_extended += 1

    return fingers_extended >= PALM_MIN_FINGERS


def update_keyboard_controls(hand_landmarks):
    """
    Update keyboard controls based on right hand position relative to reference point.
    Supports diagonal movement by pressing multiple keys.
    """
    global right_hand_state, keyboard_controller

    if not right_hand_state['control_active'] or not right_hand_state['reference_point']:
        return

    wrist = hand_landmarks[0]
    ref_x, ref_y = right_hand_state['reference_point']

    # Calculate deltas from reference point
    delta_x = wrist.x - ref_x
    delta_y = wrist.y - ref_y

    # Determine which keys should be active based on current thresholds
    target_keys = set()

    # Check X-axis (left/right)
    if delta_x < -MOVEMENT_THRESHOLD_ACTIVATE:
        target_keys.add('a')  # Move left
    elif delta_x > MOVEMENT_THRESHOLD_ACTIVATE:
        target_keys.add('d')  # Move right
    else:
        # Hysteresis: only release if within smaller threshold
        if 'a' in right_hand_state['active_keys'] and delta_x > -MOVEMENT_THRESHOLD_RELEASE:
            pass  # Will be removed below
        elif 'a' in right_hand_state['active_keys']:
            target_keys.add('a')

        if 'd' in right_hand_state['active_keys'] and delta_x < MOVEMENT_THRESHOLD_RELEASE:
            pass  # Will be removed below
        elif 'd' in right_hand_state['active_keys']:
            target_keys.add('d')

    # Check Y-axis (up/down - vertical movement)
    # Smaller Y value = palm moved UP = W key (forward)
    # Larger Y value = palm moved DOWN = S key (backward)
    if delta_y < -MOVEMENT_THRESHOLD_ACTIVATE:
        target_keys.add('w')  # Palm moved up
    elif delta_y > MOVEMENT_THRESHOLD_ACTIVATE:
        target_keys.add('s')  # Palm moved down
    else:
        # Hysteresis for Y-axis
        if 'w' in right_hand_state['active_keys'] and delta_y > -MOVEMENT_THRESHOLD_RELEASE:
            pass
        elif 'w' in right_hand_state['active_keys']:
            target_keys.add('w')

        if 's' in right_hand_state['active_keys'] and delta_y < MOVEMENT_THRESHOLD_RELEASE:
            pass
        elif 's' in right_hand_state['active_keys']:
            target_keys.add('s')

    # Press new keys
    for key in target_keys - right_hand_state['active_keys']:
        print(f"KEY PRESS: {key.upper()}")

    # Release old keys
    for key in right_hand_state['active_keys'] - target_keys:
        print(f"KEY RELEASE: {key.upper()}")

    # Update active keys
    right_hand_state['active_keys'] = target_keys


def process_right_hand_control(detection_result):
    """
    Process right hand for keyboard control.
    Manages fist detection, reference point setting, and hand tracking loss.
    """
    global right_hand_state

    current_time = time.time()
    right_hand_found = False

    if detection_result and detection_result.hand_landmarks:
        # Find right hand
        for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            handedness = detection_result.handedness[idx][0]
            if handedness.category_name == "Right":
                right_hand_found = True
                right_hand_state['last_seen_time'] = current_time

                # Check palm state
                is_palm_open = is_hand_open_palm(hand_landmarks)

                # Set reference point only if not already set
                if is_palm_open and right_hand_state['reference_point'] is None:
                    wrist = hand_landmarks[0]
                    right_hand_state['reference_point'] = (wrist.x, wrist.y)
                    print(f"Reference point set - static position locked at ({wrist.x:.3f}, {wrist.y:.3f})")

                # Open palm detected - activate control
                if is_palm_open and not right_hand_state['is_palm_open']:
                    right_hand_state['control_active'] = True
                    right_hand_state['is_palm_open'] = True
                    print("Control activated - open palm detected")

                # Palm is open - update controls (no deactivation on close)
                if is_palm_open and right_hand_state['is_palm_open']:
                    update_keyboard_controls(hand_landmarks)

                # Palm closed but control stays active if reference exists
                elif not is_palm_open and right_hand_state['is_palm_open']:
                    right_hand_state['is_palm_open'] = False
                    # Note: control_active and reference_point stay set
                    # Keys will continue to be controlled based on hand position

                break

    # Handle hand tracking loss with grace period
    if not right_hand_found and right_hand_state['control_active']:
        if right_hand_state['last_seen_time']:
            time_since_seen = current_time - right_hand_state['last_seen_time']
            if time_since_seen > HAND_LOSS_GRACE_PERIOD:
                print("Hand lost - releasing controls")
                release_all_keys()
                right_hand_state['control_active'] = False
                right_hand_state['is_palm_open'] = False
                # Keep reference_point for potential return


def release_all_keys():
    """Release all currently pressed keys."""
    global right_hand_state, keyboard_controller

    for key in right_hand_state['active_keys']:
        print(f"KEY RELEASE: {key.upper()}")

    right_hand_state['active_keys'] = set()


def get_optimal_camera_settings(cap):
    """
    Configure camera for optimal performance.
    Returns the actual resolution achieved.
    """
    # Try to set desired resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)

    # Set buffer size to 1 to minimize latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Try to set FPS to 30
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Get actual values
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Camera configured:")
    print(f"  Resolution: {actual_width}x{actual_height}")
    print(f"  FPS: {actual_fps}")

    return actual_width, actual_height


def result_callback(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    """
    Callback function to receive hand landmark detection results.
    """
    global detection_result, result_lock

    # Simple spinlock to prevent race conditions
    while result_lock:
        pass

    result_lock = True
    detection_result = result
    result_lock = False


def draw_landmarks_on_image(image, detection_result):
    """
    Draw hand landmarks and connections on the image.
    """
    if not detection_result or not detection_result.hand_landmarks:
        return image

    height, width, _ = image.shape

    # Draw each detected hand
    for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
        # Get handedness (Left or Right)
        handedness = detection_result.handedness[idx][0]
        hand_label = handedness.category_name

        # Determine color based on hand
        color = COLOR_LEFT_HAND if hand_label == "Left" else COLOR_RIGHT_HAND

        # Draw connections
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection

            start_landmark = hand_landmarks[start_idx]
            end_landmark = hand_landmarks[end_idx]

            # Convert normalized coordinates to pixel coordinates
            start_x = int(start_landmark.x * width)
            start_y = int(start_landmark.y * height)
            end_x = int(end_landmark.x * width)
            end_y = int(end_landmark.y * height)

            # Draw line
            cv2.line(image, (start_x, start_y), (end_x, end_y), color, 2)

        # Draw landmarks
        for landmark in hand_landmarks:
            # Convert normalized coordinates to pixel coordinates
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            # Draw landmark point
            cv2.circle(image, (x, y), 5, color, -1)
            cv2.circle(image, (x, y), 7, (255, 255, 255), 1)

        # Draw hand label at wrist (landmark 0)
        wrist = hand_landmarks[0]
        wrist_x = int(wrist.x * width)
        wrist_y = int(wrist.y * height)

        cv2.putText(
            image,
            f"{hand_label} Hand",
            (wrist_x - 50, wrist_y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA
        )

        # Draw control status for right hand
        if hand_label == "Right" and right_hand_state['control_active']:
            status_text = "CONTROLLING - PALM" if right_hand_state['is_palm_open'] else "CONTROLLING"
            cv2.putText(
                image,
                status_text,
                (wrist_x - 50, wrist_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0) if right_hand_state['is_palm_open'] else (0, 200, 200),
                2,
                cv2.LINE_AA
            )

            # Draw reference point if active
            if right_hand_state['reference_point'] and right_hand_state['control_active']:
                ref_x, ref_y = right_hand_state['reference_point']
                ref_pixel_x = int(ref_x * width)
                ref_pixel_y = int(ref_y * height)
                cv2.circle(image, (ref_pixel_x, ref_pixel_y), 10, (0, 255, 255), 2)
                cv2.putText(image, "REF", (ref_pixel_x + 15, ref_pixel_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    return image


def draw_fps(image, fps):
    """
    Draw FPS counter on the image.
    """
    cv2.putText(
        image,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        COLOR_FPS,
        2,
        cv2.LINE_AA
    )


def main():
    """
    Main function to run the gesture recognition system.
    """
    global detection_result

    # Initialize camera
    print("Initializing camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        print("Please check that your webcam is connected and not in use by another application.")
        return

    # Configure camera settings
    width, height = get_optimal_camera_settings(cap)

    # Initialize keyboard controller (using print for now)
    global keyboard_controller
    keyboard_controller = True  # Flag to indicate controller is ready
    print("Keyboard controller initialized (print mode)")

    # Initialize MediaPipe HandLandmarker
    print("Initializing HandLandmarker...")

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        result_callback=result_callback
    )

    try:
        landmarker = vision.HandLandmarker.create_from_options(options)
    except Exception as e:
        print(f"Error initializing HandLandmarker: {e}")
        print(f"Please ensure '{MODEL_PATH}' exists in the current directory.")
        cap.release()
        return

    print("\nGesture Recognition System Started!")
    print("Controls:")
    print("  - Press 'Q' or 'ESC' to quit")
    print("  - Press 'R' to reset reference point")
    print("  - Show OPEN PALM with RIGHT hand to set/activate keyboard control")
    print("  - Move hand: LEFT/RIGHT (A/D), FORWARD/BACK (W/S)")
    print("  - Control stays active once reference point is set")
    print("\nStarting camera feed...\n")

    # FPS calculation variables
    prev_time = time.time()
    fps = 0
    frame_count = 0

    try:
        while True:
            # Capture frame
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Calculate timestamp in milliseconds
            timestamp_ms = int(time.time() * 1000)

            # Process frame asynchronously
            landmarker.detect_async(mp_image, timestamp_ms)

            # Draw landmarks from the latest detection result
            if detection_result:
                frame = draw_landmarks_on_image(frame, detection_result)
                # Process right hand for keyboard control
                process_right_hand_control(detection_result)

            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            # Draw FPS on frame
            draw_fps(frame, fps)

            # Display frame
            cv2.imshow('Gesture Recognition System', frame)

            # Check for quit commands
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:  # 27 is ESC
                print("\nShutting down...")
                break
            elif key == ord('r') or key == ord('R'):
                # Reset reference point
                release_all_keys()
                right_hand_state['reference_point'] = None
                right_hand_state['control_active'] = False
                right_hand_state['is_palm_open'] = False
                print("\nReference point reset - show open palm to set new reference")

            frame_count += 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Shutting down...")

    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        print("Cleaning up...")
        # Release any active keys before cleanup
        release_all_keys()
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Cleanup complete. Goodbye!")


if __name__ == "__main__":
    main()
