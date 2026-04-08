import os
import time
import threading

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    from pynput.keyboard import Controller, Key as PynputKey
    from pynput.mouse import Controller as MouseController, Button as MouseButton
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    PynputKey = None
    MouseButton = None

from config import GestureConfig

# Hand connections (pairs of landmark indices to draw lines between)
HAND_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4),                    # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),                    # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),                # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),              # Ring finger
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),   # Pinky
])

COLOR_FPS = (0, 255, 0)

# Map config key strings to pynput Key enum values for special keys
SPECIAL_KEY_MAP = {}
if PynputKey:
    SPECIAL_KEY_MAP = {
        # Common keys
        "space": PynputKey.space, "enter": PynputKey.enter, "tab": PynputKey.tab,
        "backspace": PynputKey.backspace, "esc": PynputKey.esc,
        # Modifiers
        "shift": PynputKey.shift, "ctrl": PynputKey.ctrl, "alt": PynputKey.alt,
        "caps_lock": PynputKey.caps_lock, "cmd": PynputKey.cmd,
        # Function keys
        "f1": PynputKey.f1, "f2": PynputKey.f2, "f3": PynputKey.f3,
        "f4": PynputKey.f4, "f5": PynputKey.f5, "f6": PynputKey.f6,
        "f7": PynputKey.f7, "f8": PynputKey.f8, "f9": PynputKey.f9,
        "f10": PynputKey.f10, "f11": PynputKey.f11, "f12": PynputKey.f12,
        # Arrow keys
        "up": PynputKey.up, "down": PynputKey.down,
        "left": PynputKey.left, "right": PynputKey.right,
        # Navigation
        "home": PynputKey.home, "end": PynputKey.end,
        "page_up": PynputKey.page_up, "page_down": PynputKey.page_down,
        "insert": PynputKey.insert, "delete": PynputKey.delete,
        # Lock / special
        "num_lock": PynputKey.num_lock, "scroll_lock": PynputKey.scroll_lock,
        "pause": PynputKey.pause, "print_screen": PynputKey.print_screen,
        "menu": PynputKey.menu,
    }
if MouseButton:
    SPECIAL_KEY_MAP["mouse_left"]   = MouseButton.left
    SPECIAL_KEY_MAP["mouse_right"]  = MouseButton.right
    SPECIAL_KEY_MAP["mouse_middle"] = MouseButton.middle


def resolve_key(key_string):
    """Map 'space' -> Key.space, single chars returned as-is."""
    return SPECIAL_KEY_MAP.get(key_string, key_string)


GESTURE_DISPLAY_NAMES = {
    "pinch":    "PINCH",
    "thumbsup": "THUMBS UP",
    "palm":     "FLAT PALM",
    "point":    "POINT",
}


class GestureProcessor:
    def __init__(self, cfg: GestureConfig):
        self.cfg = cfg

        # Thread-safe detection result
        self._result_lock = threading.Lock()
        self._detection_result = None

        # FIX (flickering): cache the last valid detection result so landmarks
        # are drawn every frame even when detect_async hasn't fired yet.
        self._last_drawn_result = None

        # Input controllers
        self.keyboard_controller = None
        self.mouse_controller = None

        # Hand control state
        self.controlling_hand_state = {
            "is_palm_open":       False,
            "reference_point":    None,
            "last_palm_position": None,
            "active_keys":        set(),
            "last_seen_time":     None,
            "control_active":     False,
        }

        # Right-hand gesture state (with debounce tracking)
        self.right_hand_gesture_state = {
            "active_gesture":  None,
            "active_key":      None,
            "pending_gesture": None,   # gesture being confirmed
            "confirm_count":   0,      # consecutive frames pending gesture seen
            "release_count":   0,      # consecutive frames active gesture absent
        }

        # MediaPipe landmarker (set by init_landmarker)
        self.landmarker = None

        # Timestamp tracking for process_frame
        self._last_timestamp_ms = 0

    @property
    def key_mapping(self):
        return {
            "forward":  self.cfg.key_forward,
            "backward": self.cfg.key_backward,
            "left":     self.cfg.key_left,
            "right":    self.cfg.key_right,
        }

    @property
    def right_hand_key_mapping(self):
        return {
            "pinch":    self.cfg.gesture_pinch_key,
            "thumbsup": self.cfg.gesture_thumbsup_key,
            "palm":     self.cfg.gesture_palm_key,
            "point":    self.cfg.gesture_point_key,
        }

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_camera(self):
        """Open camera and configure settings. Returns (cap, width, height)."""
        cap = cv2.VideoCapture(self.cfg.camera_index)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.cfg.desired_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.desired_height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)

        actual_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps    = cap.get(cv2.CAP_PROP_FPS)

        print("Camera configured:")
        print(f"  Resolution: {actual_width}x{actual_height}")
        print(f"  FPS: {actual_fps}")

        return cap, actual_width, actual_height

    def init_keyboard(self):
        """Initialize pynput keyboard and mouse controllers."""
        if PYNPUT_AVAILABLE and self.cfg.enable_actual_keypresses:
            try:
                self.keyboard_controller = Controller()
                print("Keyboard controller initialized (actual key press mode)")
            except Exception:
                self.keyboard_controller = None
                print("Keyboard controller initialization failed - print-only mode")
            try:
                self.mouse_controller = MouseController()
                print("Mouse controller initialized")
            except Exception:
                self.mouse_controller = None
                print("Mouse controller initialization failed")
        else:
            self.keyboard_controller = None
            self.mouse_controller = None
            if not PYNPUT_AVAILABLE:
                print("Keyboard controller not available - print-only mode")
            else:
                print("Keyboard controller disabled - print-only mode")

    def _press_action(self, resolved_key):
        """Press a key or mouse button via the appropriate controller."""
        if MouseButton and isinstance(resolved_key, MouseButton):
            if self.mouse_controller is not None:
                self.mouse_controller.press(resolved_key)
        elif self.keyboard_controller is not None:
            self.keyboard_controller.press(resolved_key)

    def _release_action(self, resolved_key):
        """Release a key or mouse button via the appropriate controller."""
        if MouseButton and isinstance(resolved_key, MouseButton):
            if self.mouse_controller is not None:
                self.mouse_controller.release(resolved_key)
        elif self.keyboard_controller is not None:
            self.keyboard_controller.release(resolved_key)

    def init_landmarker(self):
        """Create and return MediaPipe HandLandmarker. Also stores it as self.landmarker."""
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task"
        )

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=self.cfg.min_hand_detection_confidence,
            min_hand_presence_confidence=self.cfg.min_hand_presence_confidence,
            min_tracking_confidence=self.cfg.min_tracking_confidence,
            result_callback=self.result_callback,
        )

        self.landmarker = vision.HandLandmarker.create_from_options(options)
        return self.landmarker

    # ------------------------------------------------------------------
    # MediaPipe callback (runs on MediaPipe's internal thread)
    # ------------------------------------------------------------------

    def result_callback(self, result, output_image, timestamp_ms):
        with self._result_lock:
            self._detection_result = result

    def get_latest_result(self):
        with self._result_lock:
            result = self._detection_result
            self._detection_result = None
        return result

    # ------------------------------------------------------------------
    # Gesture detection
    # ------------------------------------------------------------------

    def is_hand_open_palm(self, hand_landmarks):
        """Detect open palm by checking finger extension."""
        wrist = hand_landmarks[0]
        fingers_extended = 0

        finger_tips  = [8, 12, 16, 20]
        finger_bases = [5,  9, 13, 17]

        for tip_idx, base_idx in zip(finger_tips, finger_bases):
            tip  = hand_landmarks[tip_idx]
            base = hand_landmarks[base_idx]

            tip_dist = (
                (tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2 + (tip.z - wrist.z) ** 2
            ) ** 0.5
            base_dist = (
                (base.x - wrist.x) ** 2 + (base.y - wrist.y) ** 2 + (base.z - wrist.z) ** 2
            ) ** 0.5

            if tip_dist > base_dist * self.cfg.palm_extension_threshold:
                fingers_extended += 1

        return fingers_extended >= self.cfg.palm_min_fingers

    @staticmethod
    def _landmark_distance(hand_landmarks, idx_a, idx_b):
        """3D Euclidean distance between two landmarks."""
        a = hand_landmarks[idx_a]
        b = hand_landmarks[idx_b]
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5

    def _finger_extended(self, hand_landmarks, tip_idx, base_idx):
        """Check if a finger is extended (tip far from wrist relative to base)."""
        wrist = hand_landmarks[0]
        tip   = hand_landmarks[tip_idx]
        base  = hand_landmarks[base_idx]
        tip_dist  = ((tip.x  - wrist.x) ** 2 + (tip.y  - wrist.y) ** 2 + (tip.z  - wrist.z) ** 2) ** 0.5
        base_dist = ((base.x - wrist.x) ** 2 + (base.y - wrist.y) ** 2 + (base.z - wrist.z) ** 2) ** 0.5
        return tip_dist > base_dist * self.cfg.palm_extension_threshold

    def _finger_curled(self, hand_landmarks, tip_idx, base_idx):
        """Check if a finger is curled (tip close to wrist relative to base)."""
        wrist = hand_landmarks[0]
        tip   = hand_landmarks[tip_idx]
        base  = hand_landmarks[base_idx]
        tip_dist  = ((tip.x  - wrist.x) ** 2 + (tip.y  - wrist.y) ** 2 + (tip.z  - wrist.z) ** 2) ** 0.5
        base_dist = ((base.x - wrist.x) ** 2 + (base.y - wrist.y) ** 2 + (base.z - wrist.z) ** 2) ** 0.5
        if base_dist < 1e-7:
            return False
        return tip_dist < base_dist * self.cfg.finger_curl_max_ratio

    def _is_fist(self, hand_landmarks):
        """True when all four fingers (index, middle, ring, pinky) are curled."""
        finger_tips  = [8, 12, 16, 20]
        finger_bases = [5,  9, 13, 17]
        curled_count = sum(
            1 for tip, base in zip(finger_tips, finger_bases)
            if self._finger_curled(hand_landmarks, tip, base)
        )
        return curled_count >= 4

    def _thumb_truly_extended(self, hand_landmarks):
        """Stricter thumb extension check that resists fist false positives."""
        wrist     = hand_landmarks[0]
        thumb_tip = hand_landmarks[4]
        thumb_mcp = hand_landmarks[2]
        index_mcp = hand_landmarks[5]

        # Check 1: Wrist-relative distance ratio with stricter threshold
        tip_dist  = ((thumb_tip.x - wrist.x)**2 + (thumb_tip.y - wrist.y)**2 + (thumb_tip.z - wrist.z)**2) ** 0.5
        base_dist = ((thumb_mcp.x - wrist.x)**2 + (thumb_mcp.y - wrist.y)**2 + (thumb_mcp.z - wrist.z)**2) ** 0.5
        if base_dist < 1e-7 or tip_dist <= base_dist * self.cfg.thumb_extended_min_ratio:
            return False

        # Check 2: Thumb tip must be meaningfully above MCP (y decreases upward)
        if thumb_tip.y >= thumb_mcp.y - self.cfg.thumbs_up_y_margin:
            return False

        # Check 3: Thumb tip must be away from index MCP (not wrapped around fist)
        thumb_to_index_mcp = ((thumb_tip.x - index_mcp.x)**2 + (thumb_tip.y - index_mcp.y)**2 + (thumb_tip.z - index_mcp.z)**2) ** 0.5
        if thumb_to_index_mcp < self.cfg.thumbs_up_min_thumb_openness:
            return False

        return True

    def is_pinch(self, hand_landmarks):
        """Thumb tip (4) and index tip (8) close together, but not a fist."""
        if self._is_fist(hand_landmarks):
            return False
        if self._landmark_distance(hand_landmarks, 4, 8) >= self.cfg.pinch_distance_threshold:
            return False
        # At least 1 of middle/ring/pinky must not be curled (real pinch has relaxed fingers)
        non_curled = sum(
            1 for tip, base in zip([12, 16, 20], [9, 13, 17])
            if not self._finger_curled(hand_landmarks, tip, base)
        )
        return non_curled >= 1

    def is_thumbs_up(self, hand_landmarks):
        """Thumb extended upward, all other fingers curled.
        Uses stricter thumb check to reject fist false positives."""
        if not self._thumb_truly_extended(hand_landmarks):
            return False
        return (
            self._finger_curled(hand_landmarks, 8,  5)
            and self._finger_curled(hand_landmarks, 12, 9)
            and self._finger_curled(hand_landmarks, 16, 13)
            and self._finger_curled(hand_landmarks, 20, 17)
        )

    def is_point(self, hand_landmarks):
        """Index extended, middle/ring/pinky curled, thumb must not be extended."""
        if not self._finger_extended(hand_landmarks, 8, 5):
            return False
        if self._finger_extended(hand_landmarks, 4, 2):
            return False
        return (
            self._finger_curled(hand_landmarks, 12, 9)
            and self._finger_curled(hand_landmarks, 16, 13)
            and self._finger_curled(hand_landmarks, 20, 17)
        )

    def classify_right_hand_gesture(self, hand_landmarks):
        """Classify gesture in priority order (most specific first). Returns name or None."""
        if self.is_thumbs_up(hand_landmarks):
            return "thumbsup"
        if self.is_point(hand_landmarks):
            return "point"
        if self.is_pinch(hand_landmarks):
            return "pinch"
        if self.is_hand_open_palm(hand_landmarks):
            return "palm"
        return None

    @staticmethod
    def calculate_palm_center(hand_landmarks):
        """Calculate palm center using 5-point average."""
        wrist       = hand_landmarks[0]
        index_base  = hand_landmarks[5]
        middle_base = hand_landmarks[9]
        ring_base   = hand_landmarks[13]
        pinky_base  = hand_landmarks[17]

        center_x = (wrist.x + index_base.x + middle_base.x + ring_base.x + pinky_base.x) / 5
        center_y = (wrist.y + index_base.y + middle_base.y + ring_base.y + pinky_base.y) / 5
        return center_x, center_y

    def find_closest_left_hand(self, detection_result):
        """Find closest left hand to camera based on Z-depth.
        NOTE: after flipping the frame, MediaPipe labels the user's left hand as 'Right'."""
        if not detection_result or not detection_result.hand_landmarks:
            return None, None

        closest_hand  = None
        closest_index = None
        min_z_depth   = float("inf")

        for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            handedness = detection_result.handedness[idx][0]
            if handedness.category_name == "Right":
                wrist       = hand_landmarks[0]
                index_base  = hand_landmarks[5]
                middle_base = hand_landmarks[9]
                ring_base   = hand_landmarks[13]
                pinky_base  = hand_landmarks[17]
                avg_z = (wrist.z + index_base.z + middle_base.z + ring_base.z + pinky_base.z) / 5
                if avg_z < min_z_depth:
                    min_z_depth   = avg_z
                    closest_hand  = hand_landmarks
                    closest_index = idx

        return closest_hand, closest_index

    def find_closest_right_hand(self, detection_result):
        """Find closest right hand to camera (user's right = MediaPipe 'Left')."""
        if not detection_result or not detection_result.hand_landmarks:
            return None, None

        closest_hand  = None
        closest_index = None
        min_z_depth   = float("inf")

        for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            handedness = detection_result.handedness[idx][0]
            if handedness.category_name == "Left":
                wrist = hand_landmarks[0]
                avg_z = (
                    wrist.z + hand_landmarks[5].z + hand_landmarks[9].z
                    + hand_landmarks[13].z + hand_landmarks[17].z
                ) / 5
                if avg_z < min_z_depth:
                    min_z_depth   = avg_z
                    closest_hand  = hand_landmarks
                    closest_index = idx

        return closest_hand, closest_index

    def is_same_hand(self, hand_landmarks, last_position):
        """Check if hand matches last tracked position using proximity."""
        if last_position is None or hand_landmarks is None:
            return False

        palm_center_x, palm_center_y = self.calculate_palm_center(hand_landmarks)

        wrist       = hand_landmarks[0]
        index_base  = hand_landmarks[5]
        middle_base = hand_landmarks[9]
        ring_base   = hand_landmarks[13]
        pinky_base  = hand_landmarks[17]
        palm_center_z = (wrist.z + index_base.z + middle_base.z + ring_base.z + pinky_base.z) / 5

        last_x, last_y, last_z = last_position
        distance = (
            (palm_center_x - last_x) ** 2
            + (palm_center_y - last_y) ** 2
            + (palm_center_z - last_z) ** 2
        ) ** 0.5

        return distance < self.cfg.hand_proximity_threshold

    # ------------------------------------------------------------------
    # Keyboard control
    # ------------------------------------------------------------------

    def update_keyboard_controls(self, hand_landmarks):
        """Update WASD keys based on hand position relative to reference point."""
        if (
            not self.controlling_hand_state["control_active"]
            or not self.controlling_hand_state["reference_point"]
        ):
            return

        palm_center_x, palm_center_y = self.calculate_palm_center(hand_landmarks)
        ref_x, ref_y = self.controlling_hand_state["reference_point"]

        delta_x = palm_center_x - ref_x
        delta_y = palm_center_y - ref_y

        target_keys  = set()
        km           = self.key_mapping
        key_right    = km["right"]
        key_left     = km["left"]
        key_forward  = km["forward"]
        key_backward = km["backward"]

        activate = self.cfg.movement_threshold_activate
        release  = self.cfg.movement_threshold_release
        active   = self.controlling_hand_state["active_keys"]

        # X-axis
        if delta_x < -activate:
            target_keys.add(key_left)
        elif delta_x > activate:
            target_keys.add(key_right)
        else:
            if key_left in active and delta_x > -release:
                pass
            elif key_left in active:
                target_keys.add(key_left)
            if key_right in active and delta_x < release:
                pass
            elif key_right in active:
                target_keys.add(key_right)

        # Y-axis
        if delta_y < -activate:
            target_keys.add(key_forward)
        elif delta_y > activate:
            target_keys.add(key_backward)
        else:
            if key_forward in active and delta_y > -release:
                pass
            elif key_forward in active:
                target_keys.add(key_forward)
            if key_backward in active and delta_y < release:
                pass
            elif key_backward in active:
                target_keys.add(key_backward)

        # Press new keys
        for key in target_keys - active:
            if self.cfg.enable_debug_output:
                print(f"KEY PRESS: {key.upper()}")
            try:
                self._press_action(key)
            except Exception:
                pass

        # Release old keys
        for key in active - target_keys:
            if self.cfg.enable_debug_output:
                print(f"KEY RELEASE: {key.upper()}")
            try:
                self._release_action(key)
            except Exception:
                pass

        self.controlling_hand_state["active_keys"] = target_keys

    def process_left_hand_control(self, detection_result):
        """Process left hand for keyboard control with multi-person support."""
        current_time = time.time()

        left_hand_landmarks, _ = self.find_closest_left_hand(detection_result)

        if left_hand_landmarks is not None:
            palm_center_x, palm_center_y = self.calculate_palm_center(left_hand_landmarks)
            wrist       = left_hand_landmarks[0]
            index_base  = left_hand_landmarks[5]
            middle_base = left_hand_landmarks[9]
            ring_base   = left_hand_landmarks[13]
            pinky_base  = left_hand_landmarks[17]
            palm_center_z = (wrist.z + index_base.z + middle_base.z + ring_base.z + pinky_base.z) / 5
            current_palm_position = (palm_center_x, palm_center_y, palm_center_z)

            if self.controlling_hand_state["control_active"]:
                if not self.is_same_hand(left_hand_landmarks, self.controlling_hand_state["last_palm_position"]):
                    if self.controlling_hand_state["last_seen_time"]:
                        time_since_seen = current_time - self.controlling_hand_state["last_seen_time"]
                        if time_since_seen <= self.cfg.hand_loss_grace_period:
                            return

            self.controlling_hand_state["last_seen_time"]     = current_time
            self.controlling_hand_state["last_palm_position"] = current_palm_position

            is_palm_open = self.is_hand_open_palm(left_hand_landmarks)

            if is_palm_open and self.controlling_hand_state["reference_point"] is None:
                self.controlling_hand_state["reference_point"] = (palm_center_x, palm_center_y)
                print(f"Reference point set - palm center locked at ({palm_center_x:.3f}, {palm_center_y:.3f})")

            if is_palm_open and not self.controlling_hand_state["is_palm_open"]:
                self.controlling_hand_state["control_active"] = True
                self.controlling_hand_state["is_palm_open"]   = True
                print("Control activated - open palm detected")

            if is_palm_open and self.controlling_hand_state["is_palm_open"]:
                self.update_keyboard_controls(left_hand_landmarks)

            elif not is_palm_open and self.controlling_hand_state["is_palm_open"]:
                self.controlling_hand_state["is_palm_open"] = False
                self.release_all_keys()
                self.controlling_hand_state["reference_point"] = None

        else:
            if self.controlling_hand_state["control_active"]:
                if self.controlling_hand_state["last_seen_time"]:
                    time_since_seen = current_time - self.controlling_hand_state["last_seen_time"]
                    if time_since_seen > self.cfg.hand_loss_grace_period:
                        print("Hand lost - releasing controls")
                        self.release_all_keys()
                        self.controlling_hand_state["control_active"] = False
                        self.controlling_hand_state["is_palm_open"]   = False

    def release_all_keys(self):
        """Release all currently pressed keys."""
        for key in self.controlling_hand_state["active_keys"]:
            if self.cfg.enable_debug_output:
                print(f"KEY RELEASE: {key.upper()}")
            try:
                self._release_action(key)
            except Exception:
                pass
        self.controlling_hand_state["active_keys"] = set()

    def reset_control(self):
        """Release keys and reset all control state."""
        self.release_all_keys()
        self.release_right_hand_gesture_key()
        self.controlling_hand_state["reference_point"]    = None
        self.controlling_hand_state["last_palm_position"] = None
        self.controlling_hand_state["control_active"]     = False
        self.controlling_hand_state["is_palm_open"]       = False
        print("\nReference point reset - show open palm with left hand to set new reference")

    # ------------------------------------------------------------------
    # Right-hand gesture control
    # ------------------------------------------------------------------

    def process_right_hand_gestures(self, detection_result):
        """Detect gesture on right hand with confirmation-frame debouncing."""
        if not self.cfg.enable_right_hand_gestures:
            return

        right_hand, _ = self.find_closest_right_hand(detection_result)

        detected_gesture = None
        if right_hand is not None:
            detected_gesture = self.classify_right_hand_gesture(right_hand)

        state  = self.right_hand_gesture_state
        active = state["active_gesture"]

        if detected_gesture == active:
            # Still seeing the active gesture — reset pending/release counters
            state["pending_gesture"] = None
            state["confirm_count"]   = 0
            state["release_count"]   = 0
            return

        # Detected something different from the active gesture
        # Track consecutive frames for the new candidate
        if detected_gesture == state["pending_gesture"]:
            state["confirm_count"] += 1
        else:
            state["pending_gesture"] = detected_gesture
            state["confirm_count"]   = 1

        # Determine the confirmation threshold
        if detected_gesture is None:
            # Releasing requires more frames (resistant to accidental drops)
            threshold = self.cfg.gesture_release_frames
        else:
            threshold = self.cfg.gesture_confirm_frames

        if state["confirm_count"] < threshold:
            return  # Not yet confirmed — wait for more frames

        # Confirmed: activate the new gesture
        # Release previous key
        if state["active_key"] is not None:
            if self.cfg.enable_debug_output:
                print(f"RIGHT GESTURE RELEASE: {active}")
            try:
                self._release_action(state["active_key"])
            except Exception:
                pass

        # Press new key
        if detected_gesture is not None:
            key_string = self.right_hand_key_mapping[detected_gesture]
            resolved   = resolve_key(key_string)
            if self.cfg.enable_debug_output:
                print(f"RIGHT GESTURE PRESS: {detected_gesture} -> {key_string}")
            try:
                self._press_action(resolved)
            except Exception:
                pass
            state["active_gesture"] = detected_gesture
            state["active_key"]     = resolved
        else:
            state["active_gesture"] = None
            state["active_key"]     = None

        state["pending_gesture"] = None
        state["confirm_count"]   = 0
        state["release_count"]   = 0

    def release_right_hand_gesture_key(self):
        """Release any currently held right-hand gesture key."""
        if self.right_hand_gesture_state["active_key"] is not None:
            try:
                self._release_action(self.right_hand_gesture_state["active_key"])
            except Exception:
                pass
        self.right_hand_gesture_state["active_gesture"]  = None
        self.right_hand_gesture_state["active_key"]      = None
        self.right_hand_gesture_state["pending_gesture"] = None
        self.right_hand_gesture_state["confirm_count"]   = 0
        self.right_hand_gesture_state["release_count"]   = 0

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw_landmarks_on_image(self, image, detection_result):
        """Draw hand landmarks and connections on the image."""
        if not detection_result or not detection_result.hand_landmarks:
            return image

        height, width, _ = image.shape
        color_left  = tuple(self.cfg.color_left_hand)
        color_right = tuple(self.cfg.color_right_hand)

        for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            handedness = detection_result.handedness[idx][0]
            hand_label = "Left" if handedness.category_name == "Right" else "Right"
            color = color_left if hand_label == "Left" else color_right

            # Draw connections
            for start_idx, end_idx in HAND_CONNECTIONS:
                s = hand_landmarks[start_idx]
                e = hand_landmarks[end_idx]
                cv2.line(
                    image,
                    (int(s.x * width), int(s.y * height)),
                    (int(e.x * width), int(e.y * height)),
                    color, 2,
                )

            # Draw landmark dots
            for landmark in hand_landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(image, (x, y), 5, color, -1)
                cv2.circle(image, (x, y), 7, (255, 255, 255), 1)

            # Draw palm centre
            pcx, pcy = self.calculate_palm_center(hand_landmarks)
            ppx, ppy = int(pcx * width), int(pcy * height)
            cv2.circle(image, (ppx, ppy), 8,  (255, 0, 255), -1)
            cv2.circle(image, (ppx, ppy), 10, (255, 255, 255), 2)

            # Draw hand label near wrist
            wrist = hand_landmarks[0]
            wx, wy = int(wrist.x * width), int(wrist.y * height)
            cv2.putText(image, f"{hand_label} Hand", (wx - 50, wy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            # Draw control status for left hand
            if hand_label == "Left" and self.controlling_hand_state["control_active"]:
                is_ctrl = self.is_same_hand(
                    hand_landmarks, self.controlling_hand_state["last_palm_position"]
                )
                if is_ctrl or self.controlling_hand_state["last_palm_position"] is None:
                    status_text = (
                        "CONTROLLING - PALM"
                        if self.controlling_hand_state["is_palm_open"]
                        else "CONTROLLING"
                    )
                    status_color = (
                        (0, 255, 0) if self.controlling_hand_state["is_palm_open"] else (0, 200, 200)
                    )
                    cv2.putText(image, status_text, (wx - 50, wy + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)

            # Draw reference point + direction arrow
            if self.controlling_hand_state["reference_point"] and self.controlling_hand_state["control_active"]:
                ref_x, ref_y = self.controlling_hand_state["reference_point"]
                rpx, rpy = int(ref_x * width), int(ref_y * height)
                cv2.circle(image, (rpx, rpy), 10, (0, 255, 255), 2)
                cv2.putText(image, "REF", (rpx + 15, rpy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                self.draw_direction_arrow(
                    image,
                    self.controlling_hand_state["reference_point"],
                    self.controlling_hand_state["active_keys"],
                )

            # Draw gesture status for right hand
            if hand_label == "Right":
                gesture = self.right_hand_gesture_state["active_gesture"]
                if gesture:
                    display_name = GESTURE_DISPLAY_NAMES.get(gesture, gesture.upper())
                    key_name     = self.right_hand_key_mapping[gesture]
                    cv2.putText(
                        image,
                        f"{display_name} [{key_name.upper()}]",
                        (wx - 50, wy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
                    )

        return image

    @staticmethod
    def draw_fps(image, fps):
        """Draw FPS counter on the image."""
        cv2.putText(image, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_FPS, 2, cv2.LINE_AA)

    def draw_wasd_overlay(self, image):
        """Draw WASD keyboard overlay showing which keys are currently pressed."""
        if not self.cfg.wasd_overlay_enabled:
            return

        height, width, _ = image.shape
        active_keys = self.controlling_hand_state["active_keys"]

        km    = self.key_mapping
        key_w = km["forward"]
        key_a = km["left"]
        key_s = km["backward"]
        key_d = km["right"]

        ks     = self.cfg.wasd_key_size
        sp     = self.cfg.wasd_key_spacing
        base_x = self.cfg.wasd_overlay_x
        base_y = height - self.cfg.wasd_overlay_y_offset

        color_active   = tuple(self.cfg.wasd_key_color_active)
        color_inactive = tuple(self.cfg.wasd_key_color_inactive)
        text_active    = tuple(self.cfg.wasd_text_color_active)
        text_inactive  = tuple(self.cfg.wasd_text_color_inactive)

        keys = {
            key_w: {"pos": (base_x + ks + sp,          base_y - ks - sp), "label": "W"},
            key_a: {"pos": (base_x,                    base_y),            "label": "A"},
            key_s: {"pos": (base_x + ks + sp,          base_y),            "label": "S"},
            key_d: {"pos": (base_x + 2 * (ks + sp),   base_y),            "label": "D"},
        }

        for key, info in keys.items():
            x, y  = info["pos"]
            label = info["label"]
            is_active = key in active_keys

            box_color  = color_active   if is_active else color_inactive
            text_color = text_active    if is_active else text_inactive

            cv2.rectangle(image, (x, y), (x + ks, y + ks), box_color, -1)
            cv2.rectangle(image, (x, y), (x + ks, y + ks), (255, 255, 255), 2)

            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            tx = x + (ks - text_size[0]) // 2
            ty = y + (ks + text_size[1]) // 2
            cv2.putText(image, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)

    def draw_direction_arrow(self, image, reference_point, active_keys):
        """Draw a directional arrow near the reference point."""
        if not active_keys or not reference_point:
            return

        ARROW_COLOR      = (0, 255, 255)
        ARROW_LENGTH     = 120
        ARROW_THICKNESS  = 3
        ARROW_TIP_LENGTH = 0.3

        height, width, _ = image.shape
        ref_x = int(reference_point[0] * width)
        ref_y = int(reference_point[1] * height)

        km          = self.key_mapping
        direction_x = 0
        direction_y = 0

        if km["right"]    in active_keys: 
            direction_x += 1
        if km["left"]     in active_keys: 
            direction_x -= 1
        if km["forward"]  in active_keys: 
            direction_y -= 1
        if km["backward"] in active_keys: 
            direction_y += 1

        if direction_x == 0 and direction_y == 0:
            return

        magnitude = (direction_x ** 2 + direction_y ** 2) ** 0.5
        norm_x = direction_x / magnitude
        norm_y = direction_y / magnitude

        end_x = int(ref_x + norm_x * ARROW_LENGTH)
        end_y = int(ref_y + norm_y * ARROW_LENGTH)

        cv2.arrowedLine(image, (ref_x, ref_y), (end_x, end_y),
                        ARROW_COLOR, ARROW_THICKNESS, tipLength=ARROW_TIP_LENGTH)

    # ------------------------------------------------------------------
    # Frame pipeline
    # ------------------------------------------------------------------

    def process_frame(self, frame):
        """Full processing pipeline for a single frame.

        Flips the frame, queues it with MediaPipe detect_async, then:
        - Updates the cached result when a fresh callback has fired.
        - Runs gesture/control logic ONLY on fresh results (no redundant updates).
        - Draws landmarks from the CACHED result every frame (fixes flickering).

        Background: detect_async is non-blocking — the result_callback fires on
        MediaPipe's internal thread some time later.  Calling get_latest_result()
        immediately after detect_async frequently returns None (callback hasn't
        fired yet), which used to cause every other frame to skip drawing, creating
        the visible flicker.  Caching the last valid result and always drawing from
        it eliminates the skip entirely.
        """
        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int(time.time() * 1000)
        # MediaPipe requires strictly increasing timestamps
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        self.landmarker.detect_async(mp_image, timestamp_ms)

        result = self.get_latest_result()
        if result is not None:
            self._last_drawn_result = result        # refresh cache
            self.process_left_hand_control(result)  # control logic on fresh data only
            self.process_right_hand_gestures(result)

        # Always draw from cache — never skips a frame
        if self._last_drawn_result is not None:
            frame = self.draw_landmarks_on_image(frame, self._last_drawn_result)

        self.draw_wasd_overlay(frame)

        return frame

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Release keys and close landmarker."""
        self.release_all_keys()
        self.release_right_hand_gesture_key()
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None