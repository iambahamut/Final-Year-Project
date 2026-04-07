import dataclasses
from dataclasses import dataclass, field, asdict
import json


@dataclass
class GestureConfig:
    # Camera
    camera_index: int = 0
    desired_width: int = 1920
    desired_height: int = 1080

    # MediaPipe detection thresholds
    min_hand_detection_confidence: float = 0.7
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # Palm detection
    palm_extension_threshold: float = 1.1
    palm_min_fingers: int = 3

    # Movement thresholds
    movement_threshold_activate: float = 0.12
    movement_threshold_release: float = 0.08

    # Hand tracking
    hand_loss_grace_period: float = 2.0
    hand_proximity_threshold: float = 0.2

    # Keyboard control
    enable_actual_keypresses: bool = True
    enable_debug_output: bool = True
    key_forward: str = "w"
    key_backward: str = "s"
    key_left: str = "a"
    key_right: str = "d"

    # Right-hand gesture keys
    enable_right_hand_gestures: bool = True
    gesture_pinch_key: str = "e"
    gesture_thumbsup_key: str = "space"
    gesture_palm_key: str = "f"
    gesture_point_key: str = "q"

    # Right-hand gesture thresholds
    pinch_distance_threshold: float = 0.06
    finger_curl_max_ratio: float = 0.9

    # PiP
    pip_scale: float = 0.4

    # Colors (BGR format as lists for JSON compatibility)
    color_left_hand: list = field(default_factory=lambda: [255, 0, 0])
    color_right_hand: list = field(default_factory=lambda: [0, 0, 255])

    # WASD overlay
    wasd_overlay_enabled: bool = True
    wasd_key_size: int = 50
    wasd_key_spacing: int = 10
    wasd_overlay_x: int = 20
    wasd_overlay_y_offset: int = 150
    wasd_key_color_inactive: list = field(default_factory=lambda: [60, 60, 60])
    wasd_key_color_active: list = field(default_factory=lambda: [0, 255, 0])
    wasd_text_color_inactive: list = field(default_factory=lambda: [180, 180, 180])
    wasd_text_color_active: list = field(default_factory=lambda: [0, 0, 0])

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "GestureConfig":
        with open(path, "r") as f:
            data = json.load(f)
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})