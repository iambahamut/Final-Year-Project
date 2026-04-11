import sys
import time

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QSpinBox, QDoubleSpinBox, QSlider, QCheckBox, QPushButton,
    QLabel, QColorDialog, QGroupBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QKeyEvent

from config import GestureConfig
from pip_overlay import CameraWorker, PipOverlay

# ---------------------------------------------------------------------------
# Custom widgets
# ---------------------------------------------------------------------------

_SPECIAL_KEY_NAMES = {
    Qt.Key.Key_Space: "space",
    Qt.Key.Key_Return: "enter",
    Qt.Key.Key_Enter: "enter",
    Qt.Key.Key_Tab: "tab",
    Qt.Key.Key_Backspace: "backspace",
    Qt.Key.Key_Escape: "esc",
    Qt.Key.Key_Shift: "shift",
    Qt.Key.Key_Control: "ctrl",
    Qt.Key.Key_Alt: "alt",
    Qt.Key.Key_CapsLock: "caps_lock",
    Qt.Key.Key_Meta: "cmd",
    Qt.Key.Key_F1: "f1", Qt.Key.Key_F2: "f2", Qt.Key.Key_F3: "f3",
    Qt.Key.Key_F4: "f4", Qt.Key.Key_F5: "f5", Qt.Key.Key_F6: "f6",
    Qt.Key.Key_F7: "f7", Qt.Key.Key_F8: "f8", Qt.Key.Key_F9: "f9",
    Qt.Key.Key_F10: "f10", Qt.Key.Key_F11: "f11", Qt.Key.Key_F12: "f12",
    Qt.Key.Key_Up: "up", Qt.Key.Key_Down: "down",
    Qt.Key.Key_Left: "left", Qt.Key.Key_Right: "right",
    Qt.Key.Key_Home: "home", Qt.Key.Key_End: "end",
    Qt.Key.Key_PageUp: "page_up", Qt.Key.Key_PageDown: "page_down",
    Qt.Key.Key_Insert: "insert", Qt.Key.Key_Delete: "delete",
    Qt.Key.Key_NumLock: "num_lock", Qt.Key.Key_ScrollLock: "scroll_lock",
    Qt.Key.Key_Pause: "pause", Qt.Key.Key_Print: "print_screen",
    Qt.Key.Key_Menu: "menu",
}


def _stacked_label(name: str, hint: str = "") -> QWidget:
    """Label widget: name in normal font, hint below in smaller muted text."""
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 2, 0, 2)
    layout.setSpacing(1)

    name_lbl = QLabel(name)
    layout.addWidget(name_lbl)

    if hint:
        hint_lbl = QLabel(hint)
        hint_lbl.setWordWrap(True)
        f = hint_lbl.font()
        f.setPointSize(max(7, f.pointSize() - 2))
        hint_lbl.setFont(f)
        hint_lbl.setStyleSheet("color: #888888;")
        layout.addWidget(hint_lbl)

    return container


class KeyCaptureButton(QPushButton):
    """Button that captures a single keypress or mouse click for binding."""

    def __init__(self, key: str, parent=None):
        super().__init__(self._display_text(key), parent)
        self._key = key
        self._listening = False
        self._listen_start_time = 0.0
        self.setMinimumWidth(80)
        self.clicked.connect(self._start_listening)

    @staticmethod
    def _display_text(key: str) -> str:
        return key.upper().replace("_", " ")

    def _start_listening(self):
        self._listening = True
        self._listen_start_time = time.time()
        self.setText("...")
        self.grabKeyboard()

    def mousePressEvent(self, event):
        if self._listening and (time.time() - self._listen_start_time) > 0.3:
            button = event.button()
            if button == Qt.MouseButton.LeftButton:
                self._key = "mouse_left"
                self._stop_listening()
                return
            elif button == Qt.MouseButton.RightButton:
                self._key = "mouse_right"
                self._stop_listening()
                return
            elif button == Qt.MouseButton.MiddleButton:
                self._key = "mouse_middle"
                self._stop_listening()
                return
        super().mousePressEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        if not self._listening:
            super().keyPressEvent(event)
            return
        if event.key() == Qt.Key.Key_Escape:
            self._stop_listening()
            return
        if event.key() in _SPECIAL_KEY_NAMES:
            self._key = _SPECIAL_KEY_NAMES[event.key()]
            self._stop_listening()
            return
        text = event.text()
        if text and text.isprintable() and len(text) == 1:
            self._key = text.lower()
            self._stop_listening()

    def _stop_listening(self):
        self._listening = False
        self.releaseKeyboard()
        self.setText(self._display_text(self._key))

    def get_key(self) -> str:
        return self._key

    def set_key(self, key: str):
        self._key = key
        self.setText(self._display_text(key))


class ColorPickerButton(QPushButton):
    """Button that shows a color swatch and opens a color picker dialog."""

    def __init__(self, bgr: list, parent=None):
        super().__init__(parent)
        self._bgr = list(bgr)
        self.setFixedSize(60, 30)
        self._update_swatch()
        self.clicked.connect(self._pick_color)

    def _update_swatch(self):
        b, g, r = self._bgr
        self.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); border: 1px solid #888;"
        )

    def _pick_color(self):
        b, g, r = self._bgr
        initial = QColor(r, g, b)
        color = QColorDialog.getColor(initial, self, "Pick a color")
        if color.isValid():
            self._bgr = [color.blue(), color.green(), color.red()]
            self._update_swatch()

    def get_bgr(self) -> list:
        return list(self._bgr)

    def set_bgr(self, bgr: list):
        self._bgr = list(bgr)
        self._update_swatch()


# ---------------------------------------------------------------------------
# Helper: confidence slider row (label stacked above slider)
# ---------------------------------------------------------------------------

def _make_confidence_row(name: str, hint: str, value: float):
    """Return (layout, slider, value_label) for a 0-100 int slider mapped to 0.0-1.0."""
    row = QHBoxLayout()
    lbl = _stacked_label(name, hint)
    lbl.setFixedWidth(180)
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(0, 100)
    slider.setValue(int(value * 100))
    val_label = QLabel(f"{value:.2f}")
    val_label.setFixedWidth(40)
    slider.valueChanged.connect(lambda v: val_label.setText(f"{v / 100:.2f}"))
    row.addWidget(lbl)
    row.addWidget(slider)
    row.addWidget(val_label)
    return row, slider, val_label


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Settings")
        self.setFixedWidth(520)
        self.worker = None
        self.pip_overlay = None

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs)

        self._build_camera_tab()
        self._build_gesture_tab()
        self._build_preprocess_tab()
        self._build_keys_tab()
        self._build_display_tab()
        self._build_colors_tab()

        bottom = QHBoxLayout()
        self.btn_defaults = QPushButton("Restore Defaults")
        self.btn_defaults.clicked.connect(self._restore_defaults)
        self.btn_save = QPushButton("Save Settings")
        self.btn_save.clicked.connect(self._save_settings)
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.btn_start = QPushButton("Start")
        self.btn_start.setFixedWidth(100)
        self.btn_start.clicked.connect(self._toggle_start_stop)
        bottom.addWidget(self.btn_defaults)
        bottom.addWidget(self.btn_save)
        bottom.addStretch()
        bottom.addWidget(self.status_label)
        bottom.addStretch()
        bottom.addWidget(self.btn_start)
        root_layout.addLayout(bottom)

        try:
            self._populate(GestureConfig.from_json("config.json"))
        except Exception:
            self._populate(GestureConfig())

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------

    def _build_camera_tab(self):
        tab = QWidget()
        layout = QFormLayout(tab)
        layout.setVerticalSpacing(8)

        self.spin_camera_index = QSpinBox()
        self.spin_camera_index.setRange(0, 10)
        layout.addRow(
            _stacked_label("Camera Index", "0 = built-in webcam\n1+ = external camera"),
            self.spin_camera_index,
        )

        self.spin_width = QSpinBox()
        self.spin_width.setRange(320, 3840)
        self.spin_width.setSingleStep(160)
        layout.addRow(
            _stacked_label("Resolution Width", "higher = sharper image, uses more CPU"),
            self.spin_width,
        )

        self.spin_height = QSpinBox()
        self.spin_height.setRange(240, 2160)
        self.spin_height.setSingleStep(120)
        layout.addRow(
            _stacked_label("Resolution Height", "higher = sharper image, uses more CPU"),
            self.spin_height,
        )

        row1, self.slider_detection, _ = _make_confidence_row(
            "Detection Confidence",
            "lower = finds hands more easily\nhigher = fewer false positives",
            0.7,
        )
        layout.addRow(row1)

        row2, self.slider_presence, _ = _make_confidence_row(
            "Presence Confidence",
            "lower = hand stays tracked longer\nhigher = drops out faster",
            0.5,
        )
        layout.addRow(row2)

        row3, self.slider_tracking, _ = _make_confidence_row(
            "Tracking Confidence",
            "lower = re-detects hand more often\nhigher = trusts existing track",
            0.5,
        )
        layout.addRow(row3)

        self.tabs.addTab(tab, "Camera && Detection")

    def _build_gesture_tab(self):
        tab = QWidget()
        layout = QFormLayout(tab)
        layout.setVerticalSpacing(8)

        self.spin_palm_ext = QDoubleSpinBox()
        self.spin_palm_ext.setRange(0.5, 2.0)
        self.spin_palm_ext.setSingleStep(0.05)
        self.spin_palm_ext.setDecimals(2)
        layout.addRow(
            _stacked_label("Palm Extension", "lower = any open hand triggers\nhigher = needs wide finger spread"),
            self.spin_palm_ext,
        )

        self.spin_palm_fingers = QSpinBox()
        self.spin_palm_fingers.setRange(1, 4)
        layout.addRow(
            _stacked_label("Min Fingers for Palm", "lower = fewer fingers needed to open palm\nhigher = stricter"),
            self.spin_palm_fingers,
        )

        self.spin_move_activate = QDoubleSpinBox()
        self.spin_move_activate.setRange(0.01, 0.50)
        self.spin_move_activate.setSingleStep(0.01)
        self.spin_move_activate.setDecimals(2)
        layout.addRow(
            _stacked_label("Move Activate", "lower = keys fire with small movement\nhigher = needs bigger move"),
            self.spin_move_activate,
        )

        self.spin_move_release = QDoubleSpinBox()
        self.spin_move_release.setRange(0.01, 0.50)
        self.spin_move_release.setSingleStep(0.01)
        self.spin_move_release.setDecimals(2)
        layout.addRow(
            _stacked_label("Move Release", "lower = keys drop as soon as hand returns\nhigher = keys hold longer"),
            self.spin_move_release,
        )

        self.spin_grace = QDoubleSpinBox()
        self.spin_grace.setRange(0.0, 10.0)
        self.spin_grace.setSingleStep(0.5)
        self.spin_grace.setDecimals(1)
        self.spin_grace.setSuffix(" sec")
        layout.addRow(
            _stacked_label("Hand Loss Grace Period", "lower = controls drop immediately\nhigher = tolerates brief occlusion"),
            self.spin_grace,
        )

        self.spin_proximity = QDoubleSpinBox()
        self.spin_proximity.setRange(0.01, 1.0)
        self.spin_proximity.setSingleStep(0.01)
        self.spin_proximity.setDecimals(2)
        layout.addRow(
            _stacked_label("Hand Proximity", "lower = strict same-hand matching\nhigher = more tolerant of movement"),
            self.spin_proximity,
        )

        self.spin_pinch_dist = QDoubleSpinBox()
        self.spin_pinch_dist.setRange(0.01, 0.20)
        self.spin_pinch_dist.setSingleStep(0.005)
        self.spin_pinch_dist.setDecimals(3)
        layout.addRow(
            _stacked_label("Pinch Distance", "lower = fingers must nearly touch\nhigher = loose pinch is enough"),
            self.spin_pinch_dist,
        )

        self.spin_curl_ratio = QDoubleSpinBox()
        self.spin_curl_ratio.setRange(0.5, 1.0)
        self.spin_curl_ratio.setSingleStep(0.05)
        self.spin_curl_ratio.setDecimals(2)
        layout.addRow(
            _stacked_label("Finger Curl Ratio", "lower = finger must curl fully\nhigher = partial curl counts"),
            self.spin_curl_ratio,
        )

        self.tabs.addTab(tab, "Gesture Thresholds")

    def _build_preprocess_tab(self):
        tab = QWidget()
        layout = QFormLayout(tab)

        self.chk_clahe = QCheckBox("Enable CLAHE")
        layout.addRow(self.chk_clahe)

        self.spin_clahe_clip = QDoubleSpinBox()
        self.spin_clahe_clip.setRange(0.5, 10.0)
        self.spin_clahe_clip.setSingleStep(0.5)
        self.spin_clahe_clip.setDecimals(1)
        layout.addRow("CLAHE Clip Limit:", self.spin_clahe_clip)

        self.spin_clahe_tile = QSpinBox()
        self.spin_clahe_tile.setRange(2, 16)
        layout.addRow("CLAHE Tile Size:", self.spin_clahe_tile)

        self.chk_gamma = QCheckBox("Enable Gamma Correction")
        layout.addRow(self.chk_gamma)

        self.spin_gamma = QDoubleSpinBox()
        self.spin_gamma.setRange(0.3, 3.0)
        self.spin_gamma.setSingleStep(0.05)
        self.spin_gamma.setDecimals(2)
        layout.addRow("Gamma Value (1.0 = off):", self.spin_gamma)

        self.chk_auto = QCheckBox("Enable Auto-Brightness (adaptive)")
        layout.addRow(self.chk_auto)

        self.spin_auto_target = QSpinBox()
        self.spin_auto_target.setRange(40, 220)
        layout.addRow("Auto Target Brightness:", self.spin_auto_target)

        self.spin_auto_low = QSpinBox()
        self.spin_auto_low.setRange(0, 200)
        layout.addRow("Auto Dark Threshold:", self.spin_auto_low)

        self.spin_auto_high = QSpinBox()
        self.spin_auto_high.setRange(50, 255)
        layout.addRow("Auto Bright Threshold:", self.spin_auto_high)

        self.spin_auto_smooth = QDoubleSpinBox()
        self.spin_auto_smooth.setRange(0.05, 1.0)
        self.spin_auto_smooth.setSingleStep(0.05)
        self.spin_auto_smooth.setDecimals(2)
        layout.addRow("Auto EMA Smoothing:", self.spin_auto_smooth)

        self.tabs.addTab(tab, "Preprocessing")

    def _build_keys_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.chk_keypresses = QCheckBox("Enable Actual Keypresses")
        self.chk_debug = QCheckBox("Enable Debug Output")
        layout.addWidget(self.chk_keypresses)
        layout.addWidget(self.chk_debug)

        group = QGroupBox("Key Mapping")
        form = QFormLayout(group)

        self.key_forward = KeyCaptureButton("w")
        self.key_backward = KeyCaptureButton("s")
        self.key_left = KeyCaptureButton("a")
        self.key_right = KeyCaptureButton("d")

        form.addRow("Forward:", self.key_forward)
        form.addRow("Backward:", self.key_backward)
        form.addRow("Left:", self.key_left)
        form.addRow("Right:", self.key_right)

        layout.addWidget(group)

        gesture_group = QGroupBox("Right-Hand Gesture Keys")
        gesture_form = QFormLayout(gesture_group)

        self.chk_right_gestures = QCheckBox("Enable Right-Hand Gestures")
        gesture_form.addRow(self.chk_right_gestures)

        self.key_pinch = KeyCaptureButton("e")
        self.key_thumbsup = KeyCaptureButton("space")
        self.key_palm = KeyCaptureButton("f")
        self.key_point = KeyCaptureButton("q")

        gesture_form.addRow("Pinch:", self.key_pinch)
        gesture_form.addRow("Thumbs Up:", self.key_thumbsup)
        gesture_form.addRow("Flat Palm:", self.key_palm)
        gesture_form.addRow("Point:", self.key_point)

        layout.addWidget(gesture_group)
        layout.addStretch()
        self.tabs.addTab(tab, "Keys && Controls")

    def _build_display_tab(self):
        tab = QWidget()
        layout = QFormLayout(tab)
        layout.setVerticalSpacing(8)

        self.spin_pip_scale = QDoubleSpinBox()
        self.spin_pip_scale.setRange(0.1, 1.0)
        self.spin_pip_scale.setSingleStep(0.05)
        self.spin_pip_scale.setDecimals(2)
        layout.addRow(
            _stacked_label("PiP Window Size", "lower = smaller camera overlay\nhigher = larger"),
            self.spin_pip_scale,
        )

        self.chk_wasd_overlay = QCheckBox("WASD Overlay Enabled")
        layout.addRow(self.chk_wasd_overlay)

        self.spin_key_size = QSpinBox()
        self.spin_key_size.setRange(20, 100)
        self.spin_key_size.setSingleStep(5)
        layout.addRow(
            _stacked_label("Key Size (px)", "lower = smaller key boxes\nhigher = bigger key boxes"),
            self.spin_key_size,
        )

        self.spin_key_spacing = QSpinBox()
        self.spin_key_spacing.setRange(0, 30)
        self.spin_key_spacing.setSingleStep(2)
        layout.addRow(
            _stacked_label("Key Spacing (px)", "lower = keys closer together\nhigher = more spread apart"),
            self.spin_key_spacing,
        )

        self.spin_overlay_x = QSpinBox()
        self.spin_overlay_x.setRange(0, 500)
        self.spin_overlay_x.setSingleStep(10)
        layout.addRow(
            _stacked_label("Overlay X Position (px)", "horizontal offset from the left edge of the frame"),
            self.spin_overlay_x,
        )

        self.spin_overlay_y = QSpinBox()
        self.spin_overlay_y.setRange(50, 500)
        self.spin_overlay_y.setSingleStep(10)
        layout.addRow(
            _stacked_label("Overlay Y Offset (px)", "vertical offset upward from the bottom edge of the frame"),
            self.spin_overlay_y,
        )

        self.tabs.addTab(tab, "Display && Overlay")

    def _build_colors_tab(self):
        tab = QWidget()
        layout = QFormLayout(tab)
        layout.setVerticalSpacing(8)

        self.color_left_hand = ColorPickerButton([255, 0, 0])
        self.color_right_hand = ColorPickerButton([0, 0, 255])
        self.color_key_inactive = ColorPickerButton([60, 60, 60])
        self.color_key_active = ColorPickerButton([0, 255, 0])
        self.color_text_inactive = ColorPickerButton([180, 180, 180])
        self.color_text_active = ColorPickerButton([0, 0, 0])

        layout.addRow("Left Hand Color:", self.color_left_hand)
        layout.addRow("Right Hand Color:", self.color_right_hand)
        layout.addRow("Key Inactive Color:", self.color_key_inactive)
        layout.addRow("Key Active Color:", self.color_key_active)
        layout.addRow("Text Inactive Color:", self.color_text_inactive)
        layout.addRow("Text Active Color:", self.color_text_active)

        self.tabs.addTab(tab, "Colors")

    # ------------------------------------------------------------------
    # Config <-> widgets
    # ------------------------------------------------------------------

    def _populate(self, cfg: GestureConfig):
        self.spin_camera_index.setValue(cfg.camera_index)
        self.spin_width.setValue(cfg.desired_width)
        self.spin_height.setValue(cfg.desired_height)
        self.slider_detection.setValue(int(cfg.min_hand_detection_confidence * 100))
        self.slider_presence.setValue(int(cfg.min_hand_presence_confidence * 100))
        self.slider_tracking.setValue(int(cfg.min_tracking_confidence * 100))

        self.spin_palm_ext.setValue(cfg.palm_extension_threshold)
        self.spin_palm_fingers.setValue(cfg.palm_min_fingers)
        self.spin_move_activate.setValue(cfg.movement_threshold_activate)
        self.spin_move_release.setValue(cfg.movement_threshold_release)
        self.spin_grace.setValue(cfg.hand_loss_grace_period)
        self.spin_proximity.setValue(cfg.hand_proximity_threshold)
        self.spin_pinch_dist.setValue(cfg.pinch_distance_threshold)
        self.spin_curl_ratio.setValue(cfg.finger_curl_max_ratio)

        self.chk_clahe.setChecked(cfg.preprocess_clahe_enabled)
        self.spin_clahe_clip.setValue(cfg.preprocess_clahe_clip_limit)
        self.spin_clahe_tile.setValue(cfg.preprocess_clahe_tile_size)
        self.chk_gamma.setChecked(cfg.preprocess_gamma_enabled)
        self.spin_gamma.setValue(cfg.preprocess_gamma_value)
        self.chk_auto.setChecked(cfg.preprocess_auto_enabled)
        self.spin_auto_target.setValue(cfg.preprocess_auto_target)
        self.spin_auto_low.setValue(cfg.preprocess_auto_low)
        self.spin_auto_high.setValue(cfg.preprocess_auto_high)
        self.spin_auto_smooth.setValue(cfg.preprocess_auto_smoothing)

        self.chk_keypresses.setChecked(cfg.enable_actual_keypresses)
        self.chk_debug.setChecked(cfg.enable_debug_output)
        self.key_forward.set_key(cfg.key_forward)
        self.key_backward.set_key(cfg.key_backward)
        self.key_left.set_key(cfg.key_left)
        self.key_right.set_key(cfg.key_right)

        self.chk_right_gestures.setChecked(cfg.enable_right_hand_gestures)
        self.key_pinch.set_key(cfg.gesture_pinch_key)
        self.key_thumbsup.set_key(cfg.gesture_thumbsup_key)
        self.key_palm.set_key(cfg.gesture_palm_key)
        self.key_point.set_key(cfg.gesture_point_key)

        self.spin_pip_scale.setValue(cfg.pip_scale)
        self.chk_wasd_overlay.setChecked(cfg.wasd_overlay_enabled)
        self.spin_key_size.setValue(cfg.wasd_key_size)
        self.spin_key_spacing.setValue(cfg.wasd_key_spacing)
        self.spin_overlay_x.setValue(cfg.wasd_overlay_x)
        self.spin_overlay_y.setValue(cfg.wasd_overlay_y_offset)

        self.color_left_hand.set_bgr(cfg.color_left_hand)
        self.color_right_hand.set_bgr(cfg.color_right_hand)
        self.color_key_inactive.set_bgr(cfg.wasd_key_color_inactive)
        self.color_key_active.set_bgr(cfg.wasd_key_color_active)
        self.color_text_inactive.set_bgr(cfg.wasd_text_color_inactive)
        self.color_text_active.set_bgr(cfg.wasd_text_color_active)

    def _collect(self) -> GestureConfig:
        return GestureConfig(
            camera_index=self.spin_camera_index.value(),
            desired_width=self.spin_width.value(),
            desired_height=self.spin_height.value(),
            min_hand_detection_confidence=self.slider_detection.value() / 100,
            min_hand_presence_confidence=self.slider_presence.value() / 100,
            min_tracking_confidence=self.slider_tracking.value() / 100,
            palm_extension_threshold=self.spin_palm_ext.value(),
            palm_min_fingers=self.spin_palm_fingers.value(),
            movement_threshold_activate=self.spin_move_activate.value(),
            movement_threshold_release=self.spin_move_release.value(),
            hand_loss_grace_period=self.spin_grace.value(),
            hand_proximity_threshold=self.spin_proximity.value(),
            pinch_distance_threshold=self.spin_pinch_dist.value(),
            finger_curl_max_ratio=self.spin_curl_ratio.value(),
            preprocess_clahe_enabled=self.chk_clahe.isChecked(),
            preprocess_clahe_clip_limit=self.spin_clahe_clip.value(),
            preprocess_clahe_tile_size=self.spin_clahe_tile.value(),
            preprocess_gamma_enabled=self.chk_gamma.isChecked(),
            preprocess_gamma_value=self.spin_gamma.value(),
            preprocess_auto_enabled=self.chk_auto.isChecked(),
            preprocess_auto_target=self.spin_auto_target.value(),
            preprocess_auto_low=self.spin_auto_low.value(),
            preprocess_auto_high=self.spin_auto_high.value(),
            preprocess_auto_smoothing=self.spin_auto_smooth.value(),
            enable_actual_keypresses=self.chk_keypresses.isChecked(),
            enable_debug_output=self.chk_debug.isChecked(),
            key_forward=self.key_forward.get_key(),
            key_backward=self.key_backward.get_key(),
            key_left=self.key_left.get_key(),
            key_right=self.key_right.get_key(),
            enable_right_hand_gestures=self.chk_right_gestures.isChecked(),
            gesture_pinch_key=self.key_pinch.get_key(),
            gesture_thumbsup_key=self.key_thumbsup.get_key(),
            gesture_palm_key=self.key_palm.get_key(),
            gesture_point_key=self.key_point.get_key(),
            pip_scale=self.spin_pip_scale.value(),
            color_left_hand=self.color_left_hand.get_bgr(),
            color_right_hand=self.color_right_hand.get_bgr(),
            wasd_overlay_enabled=self.chk_wasd_overlay.isChecked(),
            wasd_key_size=self.spin_key_size.value(),
            wasd_key_spacing=self.spin_key_spacing.value(),
            wasd_overlay_x=self.spin_overlay_x.value(),
            wasd_overlay_y_offset=self.spin_overlay_y.value(),
            wasd_key_color_inactive=self.color_key_inactive.get_bgr(),
            wasd_key_color_active=self.color_key_active.get_bgr(),
            wasd_text_color_inactive=self.color_text_inactive.get_bgr(),
            wasd_text_color_active=self.color_text_active.get_bgr(),
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _restore_defaults(self):
        self._populate(GestureConfig())

    def _save_settings(self):
        """Persist current widget state to config.json."""
        self._collect().to_json("config.json")
        self.status_label.setText("Settings saved.")

    def _toggle_start_stop(self):
        if self.worker is not None and self.worker.isRunning():
            self._stop()
        else:
            self._start()

    def _start(self):
        cfg = self._collect()

        self.pip_overlay = PipOverlay()

        self.worker = CameraWorker(cfg)
        self.worker.frame_ready.connect(self.pip_overlay.update_frame)
        self.worker.error.connect(self._on_worker_error)
        self.worker.stopped.connect(self._on_worker_stopped)

        self.worker.start()
        self.pip_overlay.show()

        self.btn_start.setText("Stop")
        self.status_label.setText("Running...")

    def _stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(5000)
        if self.pip_overlay:
            self.pip_overlay.hide()
            self.pip_overlay = None
        self.worker = None
        self.btn_start.setText("Start")
        self.status_label.setText("Stopped")

    def _on_worker_error(self, message: str):
        self.status_label.setText(f"Error: {message}")
        self._stop()

    def _on_worker_stopped(self):
        if self.pip_overlay:
            self.pip_overlay.hide()
            self.pip_overlay = None
        self.worker = None
        self.btn_start.setText("Start")
        self.status_label.setText("Stopped")

    def closeEvent(self, event):
        self._stop()
        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())