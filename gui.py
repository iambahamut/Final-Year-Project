import sys
import time

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QSpinBox, QDoubleSpinBox, QSlider, QCheckBox, QPushButton,
    QLabel, QColorDialog, QGroupBox, QScrollArea,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import (
    QColor, QKeyEvent, QIcon, QPixmap, QPainter,
    QShortcut, QKeySequence,
)

from config import GestureConfig
from pip_overlay import CameraWorker, PipOverlay

# ---------------------------------------------------------------------------
# Light theme
# ---------------------------------------------------------------------------

LIGHT_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #f5f5f5;
    color: #1a1a1a;
    font-family: "Segoe UI", sans-serif;
}
QTabWidget::pane {
    border: 1px solid #d0d0d5;
    background: #f5f5f5;
    top: -1px;
}
QTabBar::tab {
    background: #ffffff;
    color: #6b6b6b;
    padding: 12px 14px;
    border: none;
    border-bottom: 2px solid transparent;
}
QTabBar::tab:selected {
    color: #1a1a1a;
    border-bottom: 2px solid #0a8f80;
    background: #f5f5f5;
}
QTabBar::tab:hover:!selected {
    color: #1a1a1a;
    background: #efefef;
}
QGroupBox {
    border: 1px solid #d0d0d5;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: normal;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 4px;
    color: #6b6b6b;
}
QPushButton {
    background: #efefef;
    color: #1a1a1a;
    border: 1px solid #d0d0d5;
    border-radius: 4px;
    padding: 6px 12px;
}
QPushButton:hover {
    background: #d0d0d5;
}
QPushButton:pressed {
    background: #f5f5f5;
}
QSpinBox, QDoubleSpinBox {
    background: #ffffff;
    color: #1a1a1a;
    border: 1px solid #d0d0d5;
    border-radius: 4px;
    padding: 4px;
    min-height: 20px;
}
QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #0a8f80;
}
QSpinBox::up-button, QDoubleSpinBox::up-button {
    background: #efefef;
    border-left: 1px solid #d0d0d5;
    border-bottom: 1px solid #d0d0d5;
    width: 20px;
    subcontrol-position: top right;
    subcontrol-origin: border;
    border-top-right-radius: 4px;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
    background: #0a8f80;
}
QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed {
    background: #0e9f8e;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background: #efefef;
    border-left: 1px solid #d0d0d5;
    border-top: 1px solid #d0d0d5;
    width: 20px;
    subcontrol-position: bottom right;
    subcontrol-origin: border;
    border-bottom-right-radius: 4px;
}
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background: #0a8f80;
}
QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
    background: #0e9f8e;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4IiBoZWlnaHQ9IjgiPjxwb2x5Z29uIHBvaW50cz0iNCwxIDcsNyAxLDciIGZpbGw9IiMzMzMzMzMiLz48L3N2Zz4=");
    width: 8px;
    height: 8px;
}
QSpinBox::up-arrow:hover, QDoubleSpinBox::up-arrow:hover {
    image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4IiBoZWlnaHQ9IjgiPjxwb2x5Z29uIHBvaW50cz0iNCwxIDcsNyAxLDciIGZpbGw9IiNmZmZmZmYiLz48L3N2Zz4=");
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4IiBoZWlnaHQ9IjgiPjxwb2x5Z29uIHBvaW50cz0iMSwxIDcsMSA0LDciIGZpbGw9IiMzMzMzMzMiLz48L3N2Zz4=");
    width: 8px;
    height: 8px;
}
QSpinBox::down-arrow:hover, QDoubleSpinBox::down-arrow:hover {
    image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4IiBoZWlnaHQ9IjgiPjxwb2x5Z29uIHBvaW50cz0iMSwxIDcsMSA0LDciIGZpbGw9IiNmZmZmZmYiLz48L3N2Zz4=");
}
QSlider::groove:horizontal {
    height: 4px;
    background: #d0d0d5;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 10px;
    height: 10px;
    margin: -3px 0;
    background: #0a8f80;
    border-radius: 5px;
}
QSlider::sub-page:horizontal {
    background: #0a8f80;
    border-radius: 2px;
    height: 4px;
}
QCheckBox {
    color: #1a1a1a;
    spacing: 6px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #d0d0d5;
    border-radius: 3px;
    background: #ffffff;
}
QCheckBox::indicator:checked {
    background: #0a8f80;
    border-color: #0a8f80;
}
QCheckBox::indicator:hover {
    border-color: #0a8f80;
}
QScrollArea {
    border: none;
    background: transparent;
}
QScrollArea > QWidget > QWidget {
    background: transparent;
}
QScrollBar:vertical {
    background: #f5f5f5;
    width: 8px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #d0d0d5;
    border-radius: 4px;
    min-height: 20px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}
QToolTip {
    background: #ffffff;
    color: #1a1a1a;
    border: 1px solid #d0d0d5;
    padding: 6px 8px;
    font-size: 8pt;
}
QLabel {
    background: transparent;
}
"""

_BTN_PRIMARY = (
    "background: #0a8f80; color: white; font-weight: bold;"
    "border: 1px solid #0e9f8e; border-radius: 4px; padding: 6px 12px;"
)

_BTN_DANGER = (
    "background: #e04535; color: white; font-weight: bold;"
    "border: 1px solid #c0392b; border-radius: 4px; padding: 6px 12px;"
)

_COLLAPSIBLE_STYLE = (
    "QPushButton { text-align: left; background: #ffffff;"
    "border: 1px solid #d0d0d5; border-radius: 6px;"
    "padding: 8px 12px; color: #6b6b6b; }"
    "QPushButton:hover { background: #efefef; color: #1a1a1a; }"
    "QPushButton:checked { color: #1a1a1a; border-color: #0a8f80; }"
)

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
    """Label with name on the left and a hoverable info icon on the right."""
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 4, 0)
    layout.setSpacing(4)

    name_lbl = QLabel(name)
    if hint:
        name_lbl.setToolTip(hint)
    layout.addWidget(name_lbl)

    if hint:
        icon_lbl = QLabel("ⓘ")
        icon_lbl.setStyleSheet("color: #8888a0; font-size: 10px;")
        icon_lbl.setToolTip(hint)
        icon_lbl.setCursor(Qt.CursorShape.WhatsThisCursor)
        icon_lbl.setProperty("is_hint", True)
        layout.addWidget(icon_lbl)

    layout.addStretch()
    return container


class CollapsibleSection(QWidget):
    """Clickable header that expands/collapses a content area."""

    def __init__(self, title: str, collapsed: bool = True, parent=None):
        super().__init__(parent)
        self._title = title

        self._toggle = QPushButton()
        self._toggle.setCheckable(True)
        self._toggle.setChecked(not collapsed)
        self._toggle.setStyleSheet(_COLLAPSIBLE_STYLE)
        self._toggle.toggled.connect(self._on_toggle)

        self._content = QWidget()
        self._content.setVisible(not collapsed)

        self._update_arrow(not collapsed)

        main = QVBoxLayout(self)
        main.setContentsMargins(0, 4, 0, 4)
        main.setSpacing(0)
        main.addWidget(self._toggle)
        main.addWidget(self._content)

    def _update_arrow(self, expanded: bool):
        arrow = "▼" if expanded else "▶"
        self._toggle.setText(f"{arrow} {self._title}")

    def _on_toggle(self, checked: bool):
        self._content.setVisible(checked)
        self._update_arrow(checked)

    def content_widget(self) -> QWidget:
        return self._content

    def form_layout(self) -> QFormLayout:
        """Get or create a QFormLayout on the content area."""
        if self._content.layout() is None:
            form = QFormLayout(self._content)
            form.setVerticalSpacing(8)
            form.setContentsMargins(8, 8, 8, 8)
        return self._content.layout()


class KeyCaptureButton(QPushButton):
    """Button that captures a single keypress or mouse click for binding."""

    def __init__(self, key: str, parent=None):
        super().__init__(self._display_text(key), parent)
        self._key = key
        self._listening = False
        self._listen_start_time = 0.0
        self.setMinimumWidth(80)
        self.setToolTip("Click, then press a key or click a mouse button")
        self.clicked.connect(self._start_listening)

    @staticmethod
    def _display_text(key: str) -> str:
        return key.upper().replace("_", " ")

    def _start_listening(self):
        self._listening = True
        self._listen_start_time = time.time()
        self.setText("Press a key…")
        self.setStyleSheet(
            "background-color: #0a8f80; color: white;"
            "border: 1px solid #077568; border-radius: 4px; padding: 6px 12px;"
        )
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
        self.setStyleSheet("")

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
        self.setFixedSize(80, 30)
        self._update_swatch()
        self.clicked.connect(self._pick_color)

    def _update_swatch(self):
        b, g, r = self._bgr
        hex_color = f"#{r:02x}{g:02x}{b:02x}".upper()
        self.setStyleSheet(
            f"QPushButton {{ background-color: rgb({r},{g},{b});"
            f" border: 1px solid #888; border-radius: 4px; }}"
            f"QPushButton:hover {{ border: 1px solid #0a8f80; }}"
        )
        self.setToolTip(hex_color)

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
# Helper: confidence slider row
# ---------------------------------------------------------------------------

def _make_confidence_row(name: str, hint: str, value: float):
    """Return (layout, slider, value_label) for a 0-100 int slider mapped to 0.0-1.0."""
    row = QHBoxLayout()
    lbl = _stacked_label(name, hint)
    lbl.setFixedWidth(180)
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(0, 100)
    slider.setValue(int(value * 100))
    slider.setTickPosition(QSlider.TickPosition.TicksBelow)
    slider.setTickInterval(10)
    val_label = QLabel(f"{int(value * 100)}%")
    val_label.setFixedWidth(45)
    val_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    slider.valueChanged.connect(lambda v: val_label.setText(f"{v}%"))
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
        self.setWindowTitle("Gesture Recognition — Settings")
        self.setMinimumWidth(520)
        self.setMaximumWidth(680)
        self._set_window_icon()
        self.worker = None
        self.pip_overlay = None

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        # Save button row
        save_row = QHBoxLayout()
        self.chk_show_hints = QCheckBox("Show hints")
        self.chk_show_hints.setChecked(True)
        self.chk_show_hints.setStyleSheet("color: #6b6b6b; font-size: 8pt;")
        self.chk_show_hints.toggled.connect(self._toggle_hints)
        save_row.addWidget(self.chk_show_hints)
        save_row.addStretch()
        self.btn_save = QPushButton("💾 Save Settings")
        self.btn_save.setMinimumWidth(120)
        self.btn_save.setToolTip("Save settings (Ctrl+S)")
        self.btn_save.clicked.connect(self._save_settings)
        save_row.addWidget(self.btn_save)
        root_layout.addLayout(save_row)

        save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        save_shortcut.activated.connect(self._save_settings)

        # Tabs
        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs)

        self._build_detection_tab()
        self._build_gestures_tab()
        self._build_bindings_tab()
        self._build_display_tab()

        # Bottom bar
        bottom = QHBoxLayout()
        self.btn_defaults = QPushButton("🔄 Restore Defaults")
        self.btn_defaults.clicked.connect(self._restore_defaults)
        bottom.addWidget(self.btn_defaults)
        bottom.addStretch()

        status_row = QHBoxLayout()
        status_row.setSpacing(6)
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet("color: #6b6b6b; font-size: 14px;")
        self.status_label = QLabel("Ready")
        status_row.addWidget(self.status_dot)
        status_row.addWidget(self.status_label)
        bottom.addLayout(status_row)

        bottom.addStretch()

        self.btn_start = QPushButton("▶ Start")
        self.btn_start.setFixedWidth(110)
        self.btn_start.setStyleSheet(_BTN_PRIMARY)
        self.btn_start.clicked.connect(self._toggle_start_stop)
        bottom.addWidget(self.btn_start)
        root_layout.addLayout(bottom)

        # Pulse timer
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(800)
        self._pulse_timer.timeout.connect(self._pulse_dot)
        self._dot_visible = True

        # Load config
        try:
            self._populate(GestureConfig.from_json("config.json"))
        except Exception:
            self._populate(GestureConfig())

    def _set_window_icon(self):
        pixmap = QPixmap(32, 32)
        pixmap.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor("#0a8f80"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(2, 2, 28, 28)
        painter.end()
        self.setWindowIcon(QIcon(pixmap))

    # ------------------------------------------------------------------
    # Scrollable tab helper
    # ------------------------------------------------------------------

    def _make_scrollable_tab(self, tab_name: str) -> QVBoxLayout:
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        scroll.setWidget(inner)
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self.tabs.addTab(tab, tab_name)
        return layout

    # ------------------------------------------------------------------
    # Tab 1: Detection
    # ------------------------------------------------------------------

    def _build_detection_tab(self):
        layout = self._make_scrollable_tab("🎥 Detection")

        # -- Camera Input (always visible) --
        cam_group = QGroupBox("📷 Camera Input")
        cam_form = QFormLayout(cam_group)
        cam_form.setVerticalSpacing(8)

        self.spin_camera_index = QSpinBox()
        self.spin_camera_index.setRange(0, 10)
        cam_form.addRow(
            _stacked_label("Camera Index", "0 = built-in webcam, 1+ = external camera"),
            self.spin_camera_index,
        )

        self.spin_width = QSpinBox()
        self.spin_width.setRange(320, 3840)
        self.spin_width.setSingleStep(160)
        cam_form.addRow(
            _stacked_label("Resolution Width", "higher = sharper image, uses more CPU"),
            self.spin_width,
        )

        self.spin_height = QSpinBox()
        self.spin_height.setRange(240, 2160)
        self.spin_height.setSingleStep(120)
        cam_form.addRow(
            _stacked_label("Resolution Height", "higher = sharper image, uses more CPU"),
            self.spin_height,
        )

        layout.addWidget(cam_group)

        # -- MediaPipe Confidence (collapsed) --
        self._sec_confidence = CollapsibleSection("🧠 MediaPipe Confidence")
        conf_inner = self._sec_confidence.content_widget()
        conf_layout = QVBoxLayout(conf_inner)
        conf_layout.setContentsMargins(8, 8, 8, 8)

        row1, self.slider_detection, _ = _make_confidence_row(
            "Detection Confidence",
            "lower = finds hands more easily\nhigher = fewer false positives",
            0.7,
        )
        conf_layout.addLayout(row1)

        row2, self.slider_presence, _ = _make_confidence_row(
            "Presence Confidence",
            "lower = hand stays tracked longer\nhigher = drops out faster",
            0.5,
        )
        conf_layout.addLayout(row2)

        row3, self.slider_tracking, _ = _make_confidence_row(
            "Tracking Confidence",
            "lower = re-detects hand more often\nhigher = trusts existing track",
            0.5,
        )
        conf_layout.addLayout(row3)

        layout.addWidget(self._sec_confidence)

        # -- Frame Preprocessing (collapsed) --
        self._sec_preprocess = CollapsibleSection("🌗 Frame Preprocessing")
        pp_form = self._sec_preprocess.form_layout()

        self.chk_clahe = QCheckBox("Enable CLAHE")
        pp_form.addRow(self.chk_clahe)

        self.spin_clahe_clip = QDoubleSpinBox()
        self.spin_clahe_clip.setRange(0.5, 10.0)
        self.spin_clahe_clip.setSingleStep(0.5)
        self.spin_clahe_clip.setDecimals(1)
        pp_form.addRow(
            _stacked_label("CLAHE Clip Limit", "higher = stronger local contrast"),
            self.spin_clahe_clip,
        )

        self.spin_clahe_tile = QSpinBox()
        self.spin_clahe_tile.setRange(2, 16)
        pp_form.addRow(
            _stacked_label("CLAHE Tile Size", "NxN grid for adaptive processing"),
            self.spin_clahe_tile,
        )

        self.chk_gamma = QCheckBox("Enable Gamma Correction")
        pp_form.addRow(self.chk_gamma)

        self.spin_gamma = QDoubleSpinBox()
        self.spin_gamma.setRange(0.3, 3.0)
        self.spin_gamma.setSingleStep(0.05)
        self.spin_gamma.setDecimals(2)
        pp_form.addRow(
            _stacked_label("Gamma Value", "<1 brightens, >1 darkens\n1.0 = no change"),
            self.spin_gamma,
        )

        self.chk_auto = QCheckBox("Enable Auto-Brightness (adaptive)")
        pp_form.addRow(self.chk_auto)

        self.spin_auto_target = QSpinBox()
        self.spin_auto_target.setRange(40, 220)
        pp_form.addRow(
            _stacked_label("Auto Target Brightness", "target mean luminance (0-255)"),
            self.spin_auto_target,
        )

        self.spin_auto_low = QSpinBox()
        self.spin_auto_low.setRange(0, 200)
        pp_form.addRow(
            _stacked_label("Auto Dark Threshold", "below this, brighten +\ndenoise triggers"),
            self.spin_auto_low,
        )

        self.spin_auto_high = QSpinBox()
        self.spin_auto_high.setRange(50, 255)
        pp_form.addRow(
            _stacked_label("Auto Bright Threshold", "above this, darkening triggers"),
            self.spin_auto_high,
        )

        self.spin_auto_smooth = QDoubleSpinBox()
        self.spin_auto_smooth.setRange(0.05, 1.0)
        self.spin_auto_smooth.setSingleStep(0.05)
        self.spin_auto_smooth.setDecimals(2)
        pp_form.addRow(
            _stacked_label("Auto EMA Smoothing", "higher = more reactive to changes"),
            self.spin_auto_smooth,
        )

        layout.addWidget(self._sec_preprocess)
        layout.addStretch()

    # ------------------------------------------------------------------
    # Tab 2: Gestures
    # ------------------------------------------------------------------

    def _build_gestures_tab(self):
        layout = self._make_scrollable_tab("🎯 Gestures")

        hint = QLabel("Fine-tune gesture detection. Default values work well for most setups.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #6b6b6b; padding: 4px 0;")
        layout.addWidget(hint)

        # -- Left Hand — Movement (collapsed) --
        self._sec_left_hand = CollapsibleSection("🖐️ Left Hand — Movement")
        lh_form = self._sec_left_hand.form_layout()

        self.spin_palm_ext = QDoubleSpinBox()
        self.spin_palm_ext.setRange(0.5, 2.0)
        self.spin_palm_ext.setSingleStep(0.05)
        self.spin_palm_ext.setDecimals(2)
        lh_form.addRow(
            _stacked_label("Palm Extension", "lower = any open hand triggers\nhigher = needs wide spread"),
            self.spin_palm_ext,
        )

        self.spin_palm_fingers = QSpinBox()
        self.spin_palm_fingers.setRange(1, 4)
        lh_form.addRow(
            _stacked_label("Min Fingers for Palm", "lower = fewer fingers to open palm\nhigher = stricter"),
            self.spin_palm_fingers,
        )

        self.spin_move_activate = QDoubleSpinBox()
        self.spin_move_activate.setRange(0.01, 0.50)
        self.spin_move_activate.setSingleStep(0.01)
        self.spin_move_activate.setDecimals(2)
        lh_form.addRow(
            _stacked_label("Move Activate", "lower = keys fire with small movement\nhigher = needs bigger move"),
            self.spin_move_activate,
        )

        self.spin_move_release = QDoubleSpinBox()
        self.spin_move_release.setRange(0.01, 0.50)
        self.spin_move_release.setSingleStep(0.01)
        self.spin_move_release.setDecimals(2)
        lh_form.addRow(
            _stacked_label("Move Release", "lower = keys drop as hand returns\nhigher = keys hold longer"),
            self.spin_move_release,
        )

        self.spin_grace = QDoubleSpinBox()
        self.spin_grace.setRange(0.0, 10.0)
        self.spin_grace.setSingleStep(0.5)
        self.spin_grace.setDecimals(1)
        self.spin_grace.setSuffix(" sec")
        lh_form.addRow(
            _stacked_label("Hand Loss Grace Period", "lower = controls drop immediately\nhigher = tolerates brief occlusion"),
            self.spin_grace,
        )

        self.spin_proximity = QDoubleSpinBox()
        self.spin_proximity.setRange(0.01, 1.0)
        self.spin_proximity.setSingleStep(0.01)
        self.spin_proximity.setDecimals(2)
        lh_form.addRow(
            _stacked_label("Hand Proximity", "lower = strict same-hand matching\nhigher = more tolerant"),
            self.spin_proximity,
        )

        layout.addWidget(self._sec_left_hand)

        # -- Right Hand — Gestures (collapsed) --
        self._sec_right_hand = CollapsibleSection("✋ Right Hand — Gestures")
        rh_form = self._sec_right_hand.form_layout()

        self.spin_pinch_dist = QDoubleSpinBox()
        self.spin_pinch_dist.setRange(0.01, 0.20)
        self.spin_pinch_dist.setSingleStep(0.005)
        self.spin_pinch_dist.setDecimals(3)
        rh_form.addRow(
            _stacked_label("Pinch Distance", "lower = fingers must nearly touch\nhigher = loose pinch"),
            self.spin_pinch_dist,
        )

        self.spin_curl_ratio = QDoubleSpinBox()
        self.spin_curl_ratio.setRange(0.5, 1.0)
        self.spin_curl_ratio.setSingleStep(0.05)
        self.spin_curl_ratio.setDecimals(2)
        rh_form.addRow(
            _stacked_label("Finger Curl Ratio", "lower = finger must curl fully\nhigher = partial curl counts"),
            self.spin_curl_ratio,
        )

        layout.addWidget(self._sec_right_hand)

        # -- Debounce Timing (collapsed) --
        self._sec_debounce = CollapsibleSection("⏱️ Debounce Timing")
        db_form = self._sec_debounce.form_layout()

        self.spin_confirm_frames = QSpinBox()
        self.spin_confirm_frames.setRange(1, 10)
        db_form.addRow(
            _stacked_label("Confirm Frames", "frames to confirm new gesture"),
            self.spin_confirm_frames,
        )

        self.spin_release_frames = QSpinBox()
        self.spin_release_frames.setRange(1, 15)
        db_form.addRow(
            _stacked_label("Release Frames", "frames to confirm release"),
            self.spin_release_frames,
        )

        self.chk_auto_denoise = QCheckBox("Auto Denoise")
        db_form.addRow(
            _stacked_label("", "bilateral filter on dim frames"),
            self.chk_auto_denoise,
        )

        layout.addWidget(self._sec_debounce)
        layout.addStretch()

    # ------------------------------------------------------------------
    # Tab 3: Bindings
    # ------------------------------------------------------------------

    def _build_bindings_tab(self):
        layout = self._make_scrollable_tab("⌨️ Bindings")

        info = QLabel("Click a button then press any key or mouse button to remap it.")
        info.setWordWrap(True)
        info.setStyleSheet(
            "background: #e8f5f3; border-left: 2px solid #0a8f80;"
            "padding: 6px 8px; color: #6b6b6b;"
        )
        layout.addWidget(info)

        self.chk_keypresses = QCheckBox("Enable Actual Keypresses")
        self.chk_debug = QCheckBox("Enable Debug Output")
        layout.addWidget(self.chk_keypresses)
        layout.addWidget(self.chk_debug)

        # WASD keys
        group = QGroupBox("⌨️ Key Mapping")
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

        # Right-hand gesture keys
        gesture_group = QGroupBox("🤏 Right-Hand Gesture Keys")
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

    # ------------------------------------------------------------------
    # Tab 4: Display
    # ------------------------------------------------------------------

    def _build_display_tab(self):
        layout = self._make_scrollable_tab("🖥️ Display")

        # -- PiP (always visible) --
        pip_group = QGroupBox("🖼️ Picture-in-Picture")
        pip_form = QFormLayout(pip_group)
        pip_form.setVerticalSpacing(8)

        self.spin_pip_scale = QDoubleSpinBox()
        self.spin_pip_scale.setRange(0.1, 1.0)
        self.spin_pip_scale.setSingleStep(0.05)
        self.spin_pip_scale.setDecimals(2)
        pip_form.addRow(
            _stacked_label("PiP Window Size", "lower = smaller camera overlay\nhigher = larger"),
            self.spin_pip_scale,
        )

        layout.addWidget(pip_group)

        # -- WASD overlay toggle (always visible) --
        self.chk_wasd_overlay = QCheckBox("🎮 WASD Overlay Enabled")
        layout.addWidget(self.chk_wasd_overlay)

        # -- WASD overlay layout (collapsed) --
        self._sec_overlay = CollapsibleSection("🎮 WASD Overlay Layout")
        ov_form = self._sec_overlay.form_layout()

        self.spin_key_size = QSpinBox()
        self.spin_key_size.setRange(20, 100)
        self.spin_key_size.setSingleStep(5)
        ov_form.addRow(
            _stacked_label("Key Size (px)", "lower = smaller key boxes, higher = bigger"),
            self.spin_key_size,
        )

        self.spin_key_spacing = QSpinBox()
        self.spin_key_spacing.setRange(0, 30)
        self.spin_key_spacing.setSingleStep(2)
        ov_form.addRow(
            _stacked_label("Key Spacing (px)", "lower = keys closer\nhigher = more spread apart"),
            self.spin_key_spacing,
        )

        self.spin_overlay_x = QSpinBox()
        self.spin_overlay_x.setRange(0, 500)
        self.spin_overlay_x.setSingleStep(10)
        ov_form.addRow(
            _stacked_label("Overlay X Position (px)", "horizontal offset from left edge"),
            self.spin_overlay_x,
        )

        self.spin_overlay_y = QSpinBox()
        self.spin_overlay_y.setRange(50, 500)
        self.spin_overlay_y.setSingleStep(10)
        ov_form.addRow(
            _stacked_label("Overlay Y Offset (px)", "vertical offset from bottom edge"),
            self.spin_overlay_y,
        )

        layout.addWidget(self._sec_overlay)

        # -- Colors (collapsed) --
        self._sec_colors = CollapsibleSection("🎨 Colors")
        color_form = self._sec_colors.form_layout()

        self.color_left_hand = ColorPickerButton([255, 0, 0])
        self.color_right_hand = ColorPickerButton([0, 0, 255])
        self.color_key_inactive = ColorPickerButton([60, 60, 60])
        self.color_key_active = ColorPickerButton([0, 255, 0])
        self.color_text_inactive = ColorPickerButton([180, 180, 180])
        self.color_text_active = ColorPickerButton([0, 0, 0])

        color_form.addRow("Left Hand Color:", self.color_left_hand)
        color_form.addRow("Right Hand Color:", self.color_right_hand)
        color_form.addRow("Key Inactive Color:", self.color_key_inactive)
        color_form.addRow("Key Active Color:", self.color_key_active)
        color_form.addRow("Text Inactive Color:", self.color_text_inactive)
        color_form.addRow("Text Active Color:", self.color_text_active)

        layout.addWidget(self._sec_colors)
        layout.addStretch()

    # ------------------------------------------------------------------
    # Config <-> widgets
    # ------------------------------------------------------------------

    def _populate(self, cfg: GestureConfig):
        # Detection
        self.spin_camera_index.setValue(cfg.camera_index)
        self.spin_width.setValue(cfg.desired_width)
        self.spin_height.setValue(cfg.desired_height)
        self.slider_detection.setValue(int(cfg.min_hand_detection_confidence * 100))
        self.slider_presence.setValue(int(cfg.min_hand_presence_confidence * 100))
        self.slider_tracking.setValue(int(cfg.min_tracking_confidence * 100))

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

        # Gestures
        self.spin_palm_ext.setValue(cfg.palm_extension_threshold)
        self.spin_palm_fingers.setValue(cfg.palm_min_fingers)
        self.spin_move_activate.setValue(cfg.movement_threshold_activate)
        self.spin_move_release.setValue(cfg.movement_threshold_release)
        self.spin_grace.setValue(cfg.hand_loss_grace_period)
        self.spin_proximity.setValue(cfg.hand_proximity_threshold)
        self.spin_pinch_dist.setValue(cfg.pinch_distance_threshold)
        self.spin_curl_ratio.setValue(cfg.finger_curl_max_ratio)
        self.spin_confirm_frames.setValue(cfg.gesture_confirm_frames)
        self.spin_release_frames.setValue(cfg.gesture_release_frames)
        self.chk_auto_denoise.setChecked(cfg.preprocess_auto_denoise)

        # Bindings
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

        # Display
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
            preprocess_auto_denoise=self.chk_auto_denoise.isChecked(),
            gesture_confirm_frames=self.spin_confirm_frames.value(),
            gesture_release_frames=self.spin_release_frames.value(),
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

    def _toggle_hints(self, visible: bool):
        for widget in self.findChildren(QLabel):
            if widget.property("is_hint"):
                widget.setVisible(visible)

    def _restore_defaults(self):
        self._populate(GestureConfig())

    def _save_settings(self):
        self._collect().to_json("config.json")
        self.status_label.setText("Settings saved")

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

        self.btn_start.setText("⏹ Stop")
        self.btn_start.setStyleSheet(_BTN_DANGER)
        self.status_label.setText("Running…")
        self.status_dot.setStyleSheet("color: #0a8f80; font-size: 14px;")
        self._dot_visible = True
        self.status_dot.setVisible(True)
        self._pulse_timer.start()

    def _stop(self):
        self._pulse_timer.stop()
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(5000)
        if self.pip_overlay:
            self.pip_overlay.hide()
            self.pip_overlay = None
        self.worker = None
        self.btn_start.setText("▶ Start")
        self.btn_start.setStyleSheet(_BTN_PRIMARY)
        self.status_label.setText("Stopped")
        self.status_dot.setStyleSheet("color: #6b6b6b; font-size: 14px;")
        self.status_dot.setVisible(True)

    def _on_worker_error(self, message: str):
        self._pulse_timer.stop()
        self.status_dot.setStyleSheet("color: #d93025; font-size: 14px;")
        self.status_dot.setVisible(True)
        self.status_label.setText(f"Error: {message}")
        self._stop()

    def _on_worker_stopped(self):
        self._pulse_timer.stop()
        if self.pip_overlay:
            self.pip_overlay.hide()
            self.pip_overlay = None
        self.worker = None
        self.btn_start.setText("▶ Start")
        self.btn_start.setStyleSheet(_BTN_PRIMARY)
        self.status_label.setText("Stopped")
        self.status_dot.setStyleSheet("color: #6b6b6b; font-size: 14px;")
        self.status_dot.setVisible(True)

    def _pulse_dot(self):
        self._dot_visible = not self._dot_visible
        self.status_dot.setVisible(self._dot_visible)

    def closeEvent(self, event):
        self._stop()
        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(LIGHT_STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())