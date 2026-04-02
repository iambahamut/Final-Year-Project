import time
import ctypes

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Import pynput for keyboard control
try:
    from pynput.keyboard import Controller, Key
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("Warning: pynput library not found. Keyboard control will be disabled.")
    print("Install with: pip install pynput")

# Constants
CAMERA_INDEX = 0
DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720
MIN_HAND_DETECTION_CONFIDENCE = 0.7
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
MODEL_PATH = "hand_landmarker.task"

# Open palm detection thresholds
PALM_EXTENSION_THRESHOLD = 1.1  # Finger tip must be farther from wrist than base
PALM_MIN_FINGERS = 3  # Minimum fingers extended to count as open palm

# Movement thresholds (normalized coordinates)
MOVEMENT_THRESHOLD_ACTIVATE = 0.12  # Distance from reference to activate key
MOVEMENT_THRESHOLD_RELEASE = 0.08  # Distance from reference to release key (hysteresis)

# Hand tracking loss grace period
HAND_LOSS_GRACE_PERIOD = 2.0  # seconds

# Hand proximity threshold (normalized 3D distance)
HAND_PROXIMITY_THRESHOLD = 0.2  # Maximum distance to consider same hand

# Keyboard control configuration
ENABLE_ACTUAL_KEYPRESSES = True  # Set to False to disable actual key presses (print only)
ENABLE_DEBUG_OUTPUT = True  # Set to False to disable debug print statements

# Key mapping configuration (easily customizable)
KEY_MAPPING = {
    "forward": "w",
    "backward": "s",
    "left": "a",
    "right": "d"
}

# Picture-in-Picture mode configuration
PIP_SCALE = 0.4  # Scale factor for PiP mode (40% of original size)
PIP_WINDOW_NAME = "Gesture Recognition System"

# Colors (BGR format)
COLOR_LEFT_HAND = (255, 0, 0)  # Blue
COLOR_RIGHT_HAND = (0, 0, 255)  # Red
COLOR_FPS = (0, 255, 0)  # Green

# WASD Overlay configuration
WASD_OVERLAY_ENABLED = True  # Set to False to disable overlay
WASD_KEY_SIZE = 50  # Size of each key box
WASD_KEY_SPACING = 10  # Spacing between keys
WASD_OVERLAY_X = 20  # X position (bottom-left corner)
WASD_OVERLAY_Y_OFFSET = 150  # Y offset from bottom of screen
WASD_KEY_COLOR_INACTIVE = (60, 60, 60)  # Dark gray when inactive
WASD_KEY_COLOR_ACTIVE = (0, 255, 0)  # Bright green when pressed
WASD_TEXT_COLOR_INACTIVE = (180, 180, 180)  # Light gray text
WASD_TEXT_COLOR_ACTIVE = (0, 0, 0)  # Black text when active

# Hand connections (pairs of landmark indices to draw lines between)
HAND_CONNECTIONS = frozenset(
    [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),  # Thumb
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),  # Index finger
        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),  # Middle finger
        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),  # Ring finger
        (13, 17),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),  # Pinky
    ]
)

# Global variable to store detection results from callback
detection_result = None
result_lock = False

# Keyboard control state
keyboard_controller = None
controlling_hand_state = {
    "is_palm_open": False,
    "reference_point": None,  # (x, y) when open palm first detected
    "last_palm_position": None,  # (x, y, z) 3D position for tracking same hand
    "active_keys": set(),  # Currently pressed keys
    "last_seen_time": None,  # Last time controlling hand was detected
    "control_active": False,  # Whether keyboard control is active
}

# Picture-in-Picture mode state
pip_mode = False


def set_window_always_on_top(window_name, enable=True):
    """
    Set the OpenCV window to always stay on top of other windows (Windows only).

    Args:
        window_name: Name of the OpenCV window
        enable: True to enable always-on-top, False to disable
    """
    try:
        # Windows API constants
        HWND_TOPMOST = -1
        HWND_NOTOPMOST = -2
        SWP_NOMOVE = 0x0002
        SWP_NOSIZE = 0x0001
        SWP_NOACTIVATE = 0x0010  # Don't activate/focus the window

        # Get window handle using FindWindow
        hwnd = ctypes.windll.user32.FindWindowW(None, window_name)

        if hwnd:
            # Set window position with TOPMOST or NOTOPMOST flag
            # Include SWP_NOACTIVATE to prevent stealing focus from game
            flag = HWND_TOPMOST if enable else HWND_NOTOPMOST
            ctypes.windll.user32.SetWindowPos(
                hwnd, flag, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE
            )
            return True
        return False
    except Exception:
        # Silent failure for non-Windows platforms or errors
        return False


def set_window_no_focus(window_name):
    """
    Make the OpenCV window non-focusable so it never steals focus (Windows only).
    This prevents the window from accepting keyboard focus while still being visible.

    Args:
        window_name: Name of the OpenCV window

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Windows API constants
        GWL_EXSTYLE = -20  # Extended window style
        WS_EX_NOACTIVATE = 0x08000000  # Window won't be activated when clicked

        # Get window handle
        hwnd = ctypes.windll.user32.FindWindowW(None, window_name)

        if hwnd:
            # Get current extended window style
            current_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)

            # Add WS_EX_NOACTIVATE to prevent focus stealing
            new_style = current_style | WS_EX_NOACTIVATE
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)
            return True
        return False
    except Exception:
        # Silent failure for non-Windows platforms or errors
        return False


def is_hand_open_palm(hand_landmarks):
    """
    Detect if hand is in open palm position by checking if fingers are extended.
    Returns True if at least 3 of 4 fingers (index, middle, ring, pinky) are extended.
    Thumb is ignored for relaxed palm detection.
    """
    wrist = hand_landmarks[0]
    fingers_extended = 0

    # Check each finger (excluding thumb)
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
    finger_bases = [5, 9, 13, 17]  # Corresponding MCP joints

    for tip_idx, base_idx in zip(finger_tips, finger_bases):
        tip = hand_landmarks[tip_idx]
        base = hand_landmarks[base_idx]

        # Calculate distance from wrist
        tip_dist = (
            (tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2 + (tip.z - wrist.z) ** 2
        ) ** 0.5
        base_dist = (
            (base.x - wrist.x) ** 2 + (base.y - wrist.y) ** 2 + (base.z - wrist.z) ** 2
        ) ** 0.5

        # Finger is extended if tip is farther from wrist than base
        if tip_dist > base_dist * PALM_EXTENSION_THRESHOLD:
            fingers_extended += 1

    return fingers_extended >= PALM_MIN_FINGERS


def calculate_palm_center(hand_landmarks):
    """
    Calculate the center of the palm using a 5-point average.
    Uses wrist (0) and all four finger bases (5, 9, 13, 17) for a stable,
    geometrically centered tracking point.

    Args:
        hand_landmarks: MediaPipe hand landmarks (21 points)

    Returns:
        tuple: (center_x, center_y) normalized coordinates
    """
    # Key palm landmarks: wrist + finger bases
    wrist = hand_landmarks[0]
    index_base = hand_landmarks[5]
    middle_base = hand_landmarks[9]
    ring_base = hand_landmarks[13]
    pinky_base = hand_landmarks[17]

    # Calculate average position
    center_x = (wrist.x + index_base.x + middle_base.x + ring_base.x + pinky_base.x) / 5
    center_y = (wrist.y + index_base.y + middle_base.y + ring_base.y + pinky_base.y) / 5

    return center_x, center_y


def update_keyboard_controls(hand_landmarks):
    """
    Update keyboard controls based on controlling hand position relative to reference point.
    Supports diagonal movement by pressing multiple keys.
    """
    global controlling_hand_state, keyboard_controller

    if (
        not controlling_hand_state["control_active"]
        or not controlling_hand_state["reference_point"]
    ):
        return

    # Calculate current palm center position
    palm_center_x, palm_center_y = calculate_palm_center(hand_landmarks)
    ref_x, ref_y = controlling_hand_state["reference_point"]

    # Calculate deltas from reference point
    delta_x = palm_center_x - ref_x
    delta_y = palm_center_y - ref_y

    # Determine which keys should be active based on current thresholds
    target_keys = set()

    # Get configured keys
    key_right = KEY_MAPPING["right"]
    key_left = KEY_MAPPING["left"]
    key_forward = KEY_MAPPING["forward"]
    key_backward = KEY_MAPPING["backward"]

    # Check X-axis (left/right)
    if delta_x < -MOVEMENT_THRESHOLD_ACTIVATE:
        target_keys.add(key_left)  # Palm moved left → A key
    elif delta_x > MOVEMENT_THRESHOLD_ACTIVATE:
        target_keys.add(key_right)  # Palm moved right → D key
    else:
        # Hysteresis: only release if within smaller threshold
        if (
            key_left in controlling_hand_state["active_keys"]
            and delta_x > -MOVEMENT_THRESHOLD_RELEASE
        ):
            pass  # Will be removed below
        elif key_left in controlling_hand_state["active_keys"]:
            target_keys.add(key_left)

        if (
            key_right in controlling_hand_state["active_keys"]
            and delta_x < MOVEMENT_THRESHOLD_RELEASE
        ):
            pass  # Will be removed below
        elif key_right in controlling_hand_state["active_keys"]:
            target_keys.add(key_right)

    # Check Y-axis (up/down - vertical movement)
    # Smaller Y value = palm moved UP = forward key
    # Larger Y value = palm moved DOWN = backward key
    if delta_y < -MOVEMENT_THRESHOLD_ACTIVATE:
        target_keys.add(key_forward)  # Palm moved up
    elif delta_y > MOVEMENT_THRESHOLD_ACTIVATE:
        target_keys.add(key_backward)  # Palm moved down
    else:
        # Hysteresis for Y-axis
        if (
            key_forward in controlling_hand_state["active_keys"]
            and delta_y > -MOVEMENT_THRESHOLD_RELEASE
        ):
            pass
        elif key_forward in controlling_hand_state["active_keys"]:
            target_keys.add(key_forward)

        if (
            key_backward in controlling_hand_state["active_keys"]
            and delta_y < MOVEMENT_THRESHOLD_RELEASE
        ):
            pass
        elif key_backward in controlling_hand_state["active_keys"]:
            target_keys.add(key_backward)

    # Press new keys
    for key in target_keys - controlling_hand_state["active_keys"]:
        if ENABLE_DEBUG_OUTPUT:
            print(f"KEY PRESS: {key.upper()}")

        # Actually press the key
        if keyboard_controller is not None:
            try:
                keyboard_controller.press(key)
            except:
                pass  # Silent failure

    # Release old keys
    for key in controlling_hand_state["active_keys"] - target_keys:
        if ENABLE_DEBUG_OUTPUT:
            print(f"KEY RELEASE: {key.upper()}")

        # Actually release the key
        if keyboard_controller is not None:
            try:
                keyboard_controller.release(key)
            except:
                pass  # Silent failure

    # Update active keys
    controlling_hand_state["active_keys"] = target_keys


def find_closest_left_hand(detection_result):
    """
    Find the closest left hand to the camera based on Z-depth.

    Args:
        detection_result: MediaPipe hand detection result

    Returns:
        tuple: (hand_landmarks, hand_index) for closest left hand, or (None, None) if no left hand found
    """
    if not detection_result or not detection_result.hand_landmarks:
        return None, None

    closest_hand = None
    closest_index = None
    min_z_depth = float('inf')

    for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
        handedness = detection_result.handedness[idx][0]

        # Convert to user's perspective (mirrored camera inverts left/right)
        # MediaPipe sees "Right" when user shows their left hand
        if handedness.category_name == "Right":
            # Calculate average Z-depth using palm center points
            palm_center_x, palm_center_y = calculate_palm_center(hand_landmarks)

            # Get Z values from the palm landmarks (wrist and finger bases)
            wrist = hand_landmarks[0]
            index_base = hand_landmarks[5]
            middle_base = hand_landmarks[9]
            ring_base = hand_landmarks[13]
            pinky_base = hand_landmarks[17]

            avg_z = (wrist.z + index_base.z + middle_base.z + ring_base.z + pinky_base.z) / 5

            # Smaller Z = closer to camera
            if avg_z < min_z_depth:
                min_z_depth = avg_z
                closest_hand = hand_landmarks
                closest_index = idx

    return closest_hand, closest_index


def is_same_hand(hand_landmarks, last_position):
    """
    Check if the current hand is the same as the previously tracked hand using proximity.

    Args:
        hand_landmarks: Current hand landmarks
        last_position: Last known (x, y, z) position of tracked hand

    Returns:
        bool: True if hand is within proximity threshold of last position
    """
    if last_position is None or hand_landmarks is None:
        return False

    # Calculate current palm center position (including Z)
    palm_center_x, palm_center_y = calculate_palm_center(hand_landmarks)

    # Get average Z from palm landmarks
    wrist = hand_landmarks[0]
    index_base = hand_landmarks[5]
    middle_base = hand_landmarks[9]
    ring_base = hand_landmarks[13]
    pinky_base = hand_landmarks[17]
    palm_center_z = (wrist.z + index_base.z + middle_base.z + ring_base.z + pinky_base.z) / 5

    # Calculate 3D distance from last position
    last_x, last_y, last_z = last_position
    distance = (
        (palm_center_x - last_x) ** 2 +
        (palm_center_y - last_y) ** 2 +
        (palm_center_z - last_z) ** 2
    ) ** 0.5

    return distance < HAND_PROXIMITY_THRESHOLD


def process_left_hand_control(detection_result):
    """
    Process left hand for keyboard control with multi-person support.
    Uses closest left hand to camera. Manages palm detection, reference point setting,
    and hand tracking loss with proximity-based persistence.
    """
    global controlling_hand_state

    current_time = time.time()

    # Find the closest left hand to camera
    left_hand_landmarks, left_hand_index = find_closest_left_hand(detection_result)

    if left_hand_landmarks is not None:
        # Calculate current palm position (3D)
        palm_center_x, palm_center_y = calculate_palm_center(left_hand_landmarks)
        wrist = left_hand_landmarks[0]
        index_base = left_hand_landmarks[5]
        middle_base = left_hand_landmarks[9]
        ring_base = left_hand_landmarks[13]
        pinky_base = left_hand_landmarks[17]
        palm_center_z = (wrist.z + index_base.z + middle_base.z + ring_base.z + pinky_base.z) / 5
        current_palm_position = (palm_center_x, palm_center_y, palm_center_z)

        # Check if this is the same hand we were tracking (if control is active)
        if controlling_hand_state["control_active"]:
            # Verify this is the same hand using proximity
            if not is_same_hand(left_hand_landmarks, controlling_hand_state["last_palm_position"]):
                # Different hand detected - check if we're in grace period
                if controlling_hand_state["last_seen_time"]:
                    time_since_seen = current_time - controlling_hand_state["last_seen_time"]
                    if time_since_seen <= HAND_LOSS_GRACE_PERIOD:
                        # In grace period - ignore different hand
                        return
                    # Grace period expired - this new hand can take control

        # Update tracking
        controlling_hand_state["last_seen_time"] = current_time
        controlling_hand_state["last_palm_position"] = current_palm_position

        # Check palm state
        is_palm_open = is_hand_open_palm(left_hand_landmarks)

        # Set reference point only if not already set
        if is_palm_open and controlling_hand_state["reference_point"] is None:
            controlling_hand_state["reference_point"] = (palm_center_x, palm_center_y)
            print(
                f"Reference point set - palm center locked at ({palm_center_x:.3f}, {palm_center_y:.3f})"
            )

        # Open palm detected - activate control
        if is_palm_open and not controlling_hand_state["is_palm_open"]:
            controlling_hand_state["control_active"] = True
            controlling_hand_state["is_palm_open"] = True
            print("Control activated - open palm detected")

        # Palm is open - update controls (no deactivation on close)
        if is_palm_open and controlling_hand_state["is_palm_open"]:
            update_keyboard_controls(left_hand_landmarks)

        # Palm closed - release keys and reset reference point
        elif not is_palm_open and controlling_hand_state["is_palm_open"]:
            controlling_hand_state["is_palm_open"] = False
            release_all_keys()
            controlling_hand_state["reference_point"] = None

    else:
        # No left hand found - handle tracking loss with grace period
        if controlling_hand_state["control_active"]:
            if controlling_hand_state["last_seen_time"]:
                time_since_seen = current_time - controlling_hand_state["last_seen_time"]
                if time_since_seen > HAND_LOSS_GRACE_PERIOD:
                    print("Hand lost - releasing controls")
                    release_all_keys()
                    controlling_hand_state["control_active"] = False
                    controlling_hand_state["is_palm_open"] = False
                    # Keep reference_point and last_palm_position for potential return


def release_all_keys():
    """Release all currently pressed keys."""
    global controlling_hand_state, keyboard_controller

    for key in controlling_hand_state["active_keys"]:
        if ENABLE_DEBUG_OUTPUT:
            print(f"KEY RELEASE: {key.upper()}")

        # Actually release the key
        if keyboard_controller is not None:
            try:
                keyboard_controller.release(key)
            except:
                pass  # Silent failure

    controlling_hand_state["active_keys"] = set()


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

    # Try to set desired resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)

    # Get actual values
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    actual_width = 1920
    actual_height = 1080
    print(f"Camera configured:")
    print(f"  Resolution: {actual_width}x{actual_height}")
    print(f"  FPS: {actual_fps}")

    return actual_width, actual_height


def result_callback(
    result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int
):
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

        # Convert to user's perspective (mirrored camera inverts left/right)
        # MediaPipe says "Right" when user shows their left hand
        hand_label = "Left" if handedness.category_name == "Right" else "Right"

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

        # Draw current palm center on the hand
        palm_center_x, palm_center_y = calculate_palm_center(hand_landmarks)
        palm_pixel_x = int(palm_center_x * width)
        palm_pixel_y = int(palm_center_y * height)

        # Draw palm center circle (magenta color to stand out)
        cv2.circle(image, (palm_pixel_x, palm_pixel_y), 8, (255, 0, 255), -1)  # Filled magenta circle
        cv2.circle(image, (palm_pixel_x, palm_pixel_y), 10, (255, 255, 255), 2)  # White outline

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
            cv2.LINE_AA,
        )

        # Draw control status for left hand (user's perspective)
        if hand_label == "Left" and controlling_hand_state["control_active"]:
            # Check if this is the controlling hand by proximity
            is_controlling_hand = is_same_hand(hand_landmarks, controlling_hand_state["last_palm_position"])

            if is_controlling_hand or controlling_hand_state["last_palm_position"] is None:
                status_text = (
                    "CONTROLLING - PALM"
                    if controlling_hand_state["is_palm_open"]
                    else "CONTROLLING"
                )
                cv2.putText(
                    image,
                    status_text,
                    (wrist_x - 50, wrist_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0) if controlling_hand_state["is_palm_open"] else (0, 200, 200),
                    2,
                    cv2.LINE_AA,
                )

                # Draw reference point if active
                if (
                    controlling_hand_state["reference_point"]
                    and controlling_hand_state["control_active"]
                ):
                    ref_x, ref_y = controlling_hand_state["reference_point"]
                    ref_pixel_x = int(ref_x * width)
                    ref_pixel_y = int(ref_y * height)
                    cv2.circle(image, (ref_pixel_x, ref_pixel_y), 10, (0, 255, 255), 2)
                    cv2.putText(
                        image,
                        "REF",
                        (ref_pixel_x + 15, ref_pixel_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    # Draw directional arrow when keys are pressed
                    draw_direction_arrow(
                        image,
                        controlling_hand_state["reference_point"],
                        controlling_hand_state["active_keys"]
                    )

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
        cv2.LINE_AA,
    )


def draw_wasd_overlay(image):
    """
    Draw WASD keyboard overlay showing which keys are currently pressed.
    Layout:
        W
      A S D
    """
    if not WASD_OVERLAY_ENABLED:
        return

    height, width, _ = image.shape
    active_keys = controlling_hand_state["active_keys"]

    # Get configured keys
    key_w = KEY_MAPPING["forward"]
    key_a = KEY_MAPPING["left"]
    key_s = KEY_MAPPING["backward"]
    key_d = KEY_MAPPING["right"]

    # Calculate base position (bottom-left area)
    base_x = WASD_OVERLAY_X
    base_y = height - WASD_OVERLAY_Y_OFFSET

    # Define key positions in WASD layout
    # W is centered above A S D
    keys = {
        key_w: {
            "pos": (base_x + WASD_KEY_SIZE + WASD_KEY_SPACING, base_y - WASD_KEY_SIZE - WASD_KEY_SPACING),
            "label": "W"
        },
        key_a: {
            "pos": (base_x, base_y),
            "label": "A"
        },
        key_s: {
            "pos": (base_x + WASD_KEY_SIZE + WASD_KEY_SPACING, base_y),
            "label": "S"
        },
        key_d: {
            "pos": (base_x + 2 * (WASD_KEY_SIZE + WASD_KEY_SPACING), base_y),
            "label": "D"
        }
    }

    # Draw each key
    for key, info in keys.items():
        x, y = info["pos"]
        label = info["label"]
        is_active = key in active_keys

        # Choose colors based on active state
        box_color = WASD_KEY_COLOR_ACTIVE if is_active else WASD_KEY_COLOR_INACTIVE
        text_color = WASD_TEXT_COLOR_ACTIVE if is_active else WASD_TEXT_COLOR_INACTIVE

        # Draw key box (filled rectangle)
        cv2.rectangle(
            image,
            (x, y),
            (x + WASD_KEY_SIZE, y + WASD_KEY_SIZE),
            box_color,
            -1  # Filled
        )

        # Draw border (white outline)
        cv2.rectangle(
            image,
            (x, y),
            (x + WASD_KEY_SIZE, y + WASD_KEY_SIZE),
            (255, 255, 255),
            2  # Border thickness
        )

        # Draw key label (centered)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = x + (WASD_KEY_SIZE - text_size[0]) // 2
        text_y = y + (WASD_KEY_SIZE + text_size[1]) // 2

        cv2.putText(
            image,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            text_color,
            2,
            cv2.LINE_AA,
        )


def draw_direction_arrow(image, reference_point, active_keys):
    """
    Draw a directional arrow near the reference point showing current movement direction.
    Only draws when keys are actively pressed.

    Args:
        image: The image to draw on
        reference_point: Tuple of (x, y) normalized coordinates
        active_keys: Set of currently pressed keys
    """
    if not active_keys or not reference_point:
        return

    # Arrow visual properties
    ARROW_COLOR = (0, 255, 255)  # Cyan to match reference point
    ARROW_LENGTH = 120  # pixels
    ARROW_THICKNESS = 3
    ARROW_TIP_LENGTH = 0.3  # Proportion of arrow length

    # Get image dimensions
    height, width, _ = image.shape

    # Convert reference point to pixel coordinates
    ref_x = int(reference_point[0] * width)
    ref_y = int(reference_point[1] * height)

    # Calculate direction vector based on active keys
    direction_x = 0
    direction_y = 0

    # Get configured keys
    key_right = KEY_MAPPING["right"]
    key_left = KEY_MAPPING["left"]
    key_forward = KEY_MAPPING["forward"]
    key_backward = KEY_MAPPING["backward"]

    # X-axis mapping (left/right in screen space)
    if key_right in active_keys:
        direction_x += 1  # Right on screen
    if key_left in active_keys:
        direction_x -= 1  # Left on screen

    # Y-axis mapping (up/down in screen space)
    if key_forward in active_keys:
        direction_y -= 1  # Up on screen
    if key_backward in active_keys:
        direction_y += 1  # Down on screen

    # If no direction, don't draw
    if direction_x == 0 and direction_y == 0:
        return

    # Normalize direction vector and scale to arrow length
    magnitude = (direction_x ** 2 + direction_y ** 2) ** 0.5
    norm_x = direction_x / magnitude
    norm_y = direction_y / magnitude

    # Calculate arrow end point
    end_x = int(ref_x + norm_x * ARROW_LENGTH)
    end_y = int(ref_y + norm_y * ARROW_LENGTH)

    # Draw the arrow
    cv2.arrowedLine(
        image,
        (ref_x, ref_y),  # Start at reference point
        (end_x, end_y),  # End at calculated direction
        ARROW_COLOR,
        ARROW_THICKNESS,
        tipLength=ARROW_TIP_LENGTH
    )


def main():
    """
    Main function to run the gesture recognition system.
    """
    global detection_result, pip_mode

    # Initialize camera
    print("Initializing camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        print(
            "Please check that your webcam is connected and not in use by another application."
        )
        return

    # Configure camera settings
    width, height = get_optimal_camera_settings(cap)

    # Initialize keyboard controller
    global keyboard_controller
    if PYNPUT_AVAILABLE and ENABLE_ACTUAL_KEYPRESSES:
        try:
            keyboard_controller = Controller()
            print("Keyboard controller initialized (actual key press mode)")
        except Exception as e:
            keyboard_controller = None
            print("Keyboard controller initialization failed - continuing in print-only mode")
    else:
        keyboard_controller = None
        if not PYNPUT_AVAILABLE:
            print("Keyboard controller not available - print-only mode")
        else:
            print("Keyboard controller disabled - print-only mode")

    # Initialize MediaPipe HandLandmarker
    print("Initializing HandLandmarker...")

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        result_callback=result_callback,
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
    print("  - Show OPEN PALM with LEFT hand (closest to camera) to set/activate keyboard control")
    print("  - Move hand: LEFT/RIGHT (A/D), FORWARD/BACK (W/S)")
    print("  - Control stays active once reference point is set")
    print("  - Multi-person support: Closest left hand to camera controls")
    print("  - Press 'P' to toggle Picture-in-Picture mode (always-on-top)")
    print("\nStarting camera feed...\n")

    # Create OpenCV window with NORMAL flag (resizable)
    cv2.namedWindow(PIP_WINDOW_NAME, cv2.WINDOW_NORMAL)

    # Set initial window size (full size by default)
    cv2.resizeWindow(PIP_WINDOW_NAME, width, height)

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

            # Mirror flip the frame horizontally
            frame = cv2.flip(frame, 1)

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
                # Process left hand for keyboard control
                process_left_hand_control(detection_result)

            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            # Draw FPS on frame
            draw_fps(frame, fps)

            # Draw WASD overlay
            draw_wasd_overlay(frame)

            # Scale frame for PiP mode if enabled (keep original for processing)
            if pip_mode:
                # Scale down for display
                pip_width = int(frame.shape[1] * PIP_SCALE)
                pip_height = int(frame.shape[0] * PIP_SCALE)
                display_frame = cv2.resize(frame, (pip_width, pip_height), interpolation=cv2.INTER_AREA)
            else:
                display_frame = frame

            # Display frame
            cv2.imshow(PIP_WINDOW_NAME, display_frame)

            # Check for quit commands
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q") or key == 27:  # 27 is ESC
                print("\nShutting down...")
                break
            elif key == ord("r") or key == ord("R"):
                # Reset reference point
                release_all_keys()
                controlling_hand_state["reference_point"] = None
                controlling_hand_state["last_palm_position"] = None
                controlling_hand_state["control_active"] = False
                controlling_hand_state["is_palm_open"] = False
                print("\nReference point reset - show open palm with left hand to set new reference")
            elif key == ord("p") or key == ord("P"):
                # Toggle Picture-in-Picture mode
                pip_mode = not pip_mode

                if pip_mode:
                    # Enable PiP: scale down window and set always-on-top
                    pip_width = int(width * PIP_SCALE)
                    pip_height = int(height * PIP_SCALE)
                    cv2.resizeWindow(PIP_WINDOW_NAME, pip_width, pip_height)
                    # Small delay to let window resize complete
                    time.sleep(0.05)
                    set_window_always_on_top(PIP_WINDOW_NAME, True)
                    print("\nPicture-in-Picture mode ENABLED (always-on-top, no-focus)")
                else:
                    # Disable PiP: restore full size and remove always-on-top
                    cv2.resizeWindow(PIP_WINDOW_NAME, width, height)
                    set_window_always_on_top(PIP_WINDOW_NAME, False)
                    print("\nPicture-in-Picture mode DISABLED (normal window)")

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
