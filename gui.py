import sys

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
    Qt.Key.Key_Space:     "space",
    Qt.Key.Key_Return:    "enter",
    Qt.Key.Key_Enter:     "enter",
    Qt.Key.Key_Tab:       "tab",
    Qt.Key.Key_Shift:     "shift",
    Qt.Key.Key_Control:   "ctrl",
    Qt.Key.Key_Alt:       "alt",
    Qt.Key.Key_Backspace: "backspace",
}


class KeyCaptureButton(QPushButton):
    """Button that captures a single keypress for key binding."""

    def __init__(self, key: str, parent=None):
        super().__init__(key.upper(), parent)
        self._key = key
        self._listening = False
        self.setMinimumWidth(60)
        self.clicked.connect(self._start_listening)

    def _start_listening(self):
        self._listening = True
        self.setText("...")
        self.grabKeyboard()

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
        self.setText(self._key.upper())

    def get_key(self) -> str:
        return self._key

    def set_key(self, key: str):
        self._key = key
        self.setText(key.upper())


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
# Helper: labelled slider for float values 0.0 - 1.0
# ---------------------------------------------------------------------------

def _make_confidence_row(label_text: str, value: float):
    """Return (layout, slider, value_label) for a 0-100 int slider mapped to 0.0-1.0."""
    row = QHBoxLayout()
    lbl = QLabel(label_text)
    lbl.setFixedWidth(200)
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
        self._build_keys_tab()
        self._build_display_tab()
        self._build_colors_tab()

        bottom = QHBoxLayout()
        self.btn_defaults = QPushButton("Restore Defaults")
        self.btn_defaults.clicked.connect(self._restore_defaults)
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.btn_start = QPushButton("Start")
        self.btn_start.setFixedWidth(100)
        self.btn_start.clicked.connect(self._toggle_start_stop)
        bottom.addWidget(self.btn_defaults)
        bottom.addStretch()
        bottom.addWidget(self.status_label)
        bottom.addStretch()
        bottom.addWidget(self.btn_start)
        root_layout.addLayout(bottom)

        self._populate(GestureConfig())

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------

    def _build_camera_tab(self):
        tab = QWidget()
        layout = QFormLayout(tab)

        self.spin_camera_index = QSpinBox()
        self.spin_camera_index.setRange(0, 10)
        layout.addRow("Camera Index:", self.spin_camera_index)

        self.spin_width = QSpinBox()
        self.spin_width.setRange(320, 3840)
        self.spin_width.setSingleStep(160)
        layout.addRow("Resolution Width:", self.spin_width)

        self.spin_height = QSpinBox()
        self.spin_height.setRange(240, 2160)
        self.spin_height.setSingleStep(120)
        layout.addRow("Resolution Height:", self.spin_height)

        row1, self.slider_detection, _ = _make_confidence_row("Hand Detection Confidence:", 0.7)
        layout.addRow(row1)
        row2, self.slider_presence, _ = _make_confidence_row("Hand Presence Confidence:", 0.5)
        layout.addRow(row2)
        row3, self.slider_tracking, _ = _make_confidence_row("Tracking Confidence:", 0.5)
        layout.addRow(row3)

        self.tabs.addTab(tab, "Camera && Detection")

    def _build_gesture_tab(self):
        tab = QWidget()
        layout = QFormLayout(tab)

        self.spin_palm_ext = QDoubleSpinBox()
        self.spin_palm_ext.setRange(0.5, 2.0)
        self.spin_palm_ext.setSingleStep(0.05)
        self.spin_palm_ext.setDecimals(2)
        layout.addRow("Palm Extension Threshold:", self.spin_palm_ext)

        self.spin_palm_fingers = QSpinBox()
        self.spin_palm_fingers.setRange(1, 4)
        layout.addRow("Palm Min Fingers:", self.spin_palm_fingers)

        self.spin_move_activate = QDoubleSpinBox()
        self.spin_move_activate.setRange(0.01, 0.50)
        self.spin_move_activate.setSingleStep(0.01)
        self.spin_move_activate.setDecimals(2)
        layout.addRow("Movement Activate Threshold:", self.spin_move_activate)

        self.spin_move_release = QDoubleSpinBox()
        self.spin_move_release.setRange(0.01, 0.50)
        self.spin_move_release.setSingleStep(0.01)
        self.spin_move_release.setDecimals(2)
        layout.addRow("Movement Release Threshold:", self.spin_move_release)

        self.spin_grace = QDoubleSpinBox()
        self.spin_grace.setRange(0.0, 10.0)
        self.spin_grace.setSingleStep(0.5)
        self.spin_grace.setDecimals(1)
        self.spin_grace.setSuffix(" sec")
        layout.addRow("Hand Loss Grace Period:", self.spin_grace)

        self.spin_proximity = QDoubleSpinBox()
        self.spin_proximity.setRange(0.01, 1.0)
        self.spin_proximity.setSingleStep(0.01)
        self.spin_proximity.setDecimals(2)
        layout.addRow("Hand Proximity Threshold:", self.spin_proximity)

        self.spin_pinch_dist = QDoubleSpinBox()
        self.spin_pinch_dist.setRange(0.01, 0.20)
        self.spin_pinch_dist.setSingleStep(0.005)
        self.spin_pinch_dist.setDecimals(3)
        layout.addRow("Pinch Distance Threshold:", self.spin_pinch_dist)

        self.spin_curl_ratio = QDoubleSpinBox()
        self.spin_curl_ratio.setRange(0.5, 1.0)
        self.spin_curl_ratio.setSingleStep(0.05)
        self.spin_curl_ratio.setDecimals(2)
        layout.addRow("Finger Curl Max Ratio:", self.spin_curl_ratio)

        self.tabs.addTab(tab, "Gesture Thresholds")

    def _build_keys_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.chk_keypresses = QCheckBox("Enable Actual Keypresses")
        self.chk_debug = QCheckBox("Enable Debug Output")
        layout.addWidget(self.chk_keypresses)
        layout.addWidget(self.chk_debug)

        group = QGroupBox("Key Mapping")
        form = QFormLayout(group)

        self.key_forward  = KeyCaptureButton("w")
        self.key_backward = KeyCaptureButton("s")
        self.key_left     = KeyCaptureButton("a")
        self.key_right    = KeyCaptureButton("d")

        form.addRow("Forward:",  self.key_forward)
        form.addRow("Backward:", self.key_backward)
        form.addRow("Left:",     self.key_left)
        form.addRow("Right:",    self.key_right)

        layout.addWidget(group)

        gesture_group = QGroupBox("Right-Hand Gesture Keys")
        gesture_form = QFormLayout(gesture_group)

        self.chk_right_gestures = QCheckBox("Enable Right-Hand Gestures")
        gesture_form.addRow(self.chk_right_gestures)

        self.key_pinch    = KeyCaptureButton("e")
        self.key_thumbsup = KeyCaptureButton("space")
        self.key_palm     = KeyCaptureButton("f")
        self.key_point    = KeyCaptureButton("q")

        gesture_form.addRow("Pinch:",     self.key_pinch)
        gesture_form.addRow("Thumbs Up:", self.key_thumbsup)
        gesture_form.addRow("Flat Palm:", self.key_palm)
        gesture_form.addRow("Point:",     self.key_point)

        layout.addWidget(gesture_group)
        layout.addStretch()
        self.tabs.addTab(tab, "Keys && Controls")

    def _build_display_tab(self):
        tab = QWidget()
        layout = QFormLayout(tab)

        self.spin_pip_scale = QDoubleSpinBox()
        self.spin_pip_scale.setRange(0.1, 1.0)
        self.spin_pip_scale.setSingleStep(0.05)
        self.spin_pip_scale.setDecimals(2)
        layout.addRow("PiP Scale:", self.spin_pip_scale)

        self.chk_wasd_overlay = QCheckBox("WASD Overlay Enabled")
        layout.addRow(self.chk_wasd_overlay)

        self.spin_key_size = QSpinBox()
        self.spin_key_size.setRange(20, 100)
        self.spin_key_size.setSingleStep(5)
        layout.addRow("WASD Key Size:", self.spin_key_size)

        self.spin_key_spacing = QSpinBox()
        self.spin_key_spacing.setRange(0, 30)
        self.spin_key_spacing.setSingleStep(2)
        layout.addRow("WASD Key Spacing:", self.spin_key_spacing)

        self.spin_overlay_x = QSpinBox()
        self.spin_overlay_x.setRange(0, 500)
        self.spin_overlay_x.setSingleStep(10)
        layout.addRow("WASD Overlay X:", self.spin_overlay_x)

        self.spin_overlay_y = QSpinBox()
        self.spin_overlay_y.setRange(50, 500)
        self.spin_overlay_y.setSingleStep(10)
        layout.addRow("WASD Overlay Y Offset:", self.spin_overlay_y)

        self.tabs.addTab(tab, "Display && Overlay")

    def _build_colors_tab(self):
        tab = QWidget()
        layout = QFormLayout(tab)

        self.color_left_hand    = ColorPickerButton([255, 0, 0])
        self.color_right_hand   = ColorPickerButton([0, 0, 255])
        self.color_key_inactive = ColorPickerButton([60, 60, 60])
        self.color_key_active   = ColorPickerButton([0, 255, 0])
        self.color_text_inactive = ColorPickerButton([180, 180, 180])
        self.color_text_active  = ColorPickerButton([0, 0, 0])

        layout.addRow("Left Hand Color:",    self.color_left_hand)
        layout.addRow("Right Hand Color:",   self.color_right_hand)
        layout.addRow("Key Inactive Color:", self.color_key_inactive)
        layout.addRow("Key Active Color:",   self.color_key_active)
        layout.addRow("Text Inactive Color:", self.color_text_inactive)
        layout.addRow("Text Active Color:",  self.color_text_active)

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