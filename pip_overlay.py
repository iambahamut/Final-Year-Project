import time

import cv2
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

from config import GestureConfig
from processor import GestureProcessor


class PipOverlay(QWidget):
    """Frameless, always-on-top, click-through overlay that displays camera frames."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
            | Qt.WindowType.Tool
        )
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self._drag_pos = None

        self._label = QLabel(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

    def update_frame(self, qimage: QImage):
        """Slot: receives a QImage and displays it as a pixmap."""
        pixmap = QPixmap.fromImage(qimage)
        self._label.setPixmap(pixmap)
        self.resize(pixmap.size())

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = None
            event.accept()


class CameraWorker(QThread):
    """Worker thread: captures camera frames, runs gesture processing, emits annotated frames."""

    frame_ready = pyqtSignal(QImage)
    error       = pyqtSignal(str)
    stopped     = pyqtSignal()

    # FIX (camera crash): only abort after this many *consecutive* cap.read()
    # failures.  Cameras can drop individual frames on USB bandwidth spikes or
    # driver hiccups — one failure should never kill the session.
    _MAX_CONSECUTIVE_FAILURES = 5

    # FIX (camera crash): cap the loop at 30 fps.  Without this the loop spins
    # at full CPU speed, which overwhelms the camera driver over time
    # (especially at 1920x1080) and eventually causes sustained read failures.
    _TARGET_FPS      = 30
    _TARGET_INTERVAL = 1.0 / _TARGET_FPS

    def __init__(self, config: GestureConfig, parent=None):
        super().__init__(parent)
        self._config  = config
        self._running = False

    def run(self):
        proc = GestureProcessor(self._config)

        try:
            cap, width, height = proc.init_camera()
        except Exception as e:
            self.error.emit(f"Camera error: {e}")
            return

        proc.init_keyboard()

        try:
            proc.init_landmarker()
        except Exception as e:
            cap.release()
            self.error.emit(f"MediaPipe error: {e}")
            return

        self._running        = True
        prev_time            = time.time()
        consecutive_failures = 0

        pip_w = int(width  * self._config.pip_scale)
        pip_h = int(height * self._config.pip_scale)

        try:
            while self._running:
                frame_start = time.time()

                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= self._MAX_CONSECUTIVE_FAILURES:
                        self.error.emit(
                            f"Camera read failed {consecutive_failures} times in a row"
                        )
                        break
                    # Brief pause before retrying the dropped frame
                    time.sleep(0.01)
                    continue

                consecutive_failures = 0  # reset on successful read

                annotated = proc.process_frame(frame)

                # FPS counter
                now      = time.time()
                fps      = 1 / max(now - prev_time, 1e-6)
                prev_time = now
                proc.draw_fps(annotated, fps)

                # Scale for PiP display
                display = cv2.resize(annotated, (pip_w, pip_h), interpolation=cv2.INTER_AREA)

                # BGR -> RGB for QImage
                rgb            = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                h, w, ch       = rgb.shape
                bytes_per_line = ch * w
                qimg = QImage(
                    rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
                ).copy()

                self.frame_ready.emit(qimg)

                # Rate cap: sleep the remaining time to hit TARGET_FPS.
                # Prevents the loop from spinning faster than the camera can
                # deliver frames and stressing the driver.
                elapsed   = time.time() - frame_start
                remaining = self._TARGET_INTERVAL - elapsed
                if remaining > 0:
                    time.sleep(remaining)

        finally:
            proc.cleanup()
            cap.release()
            self.stopped.emit()

    def stop(self):
        """Call from main thread to request stop."""
        self._running = False