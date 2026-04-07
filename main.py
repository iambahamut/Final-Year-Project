import time
import signal
import sys

import cv2

from config import GestureConfig
from processor import GestureProcessor

_MAX_CONSECUTIVE_FAILURES = 5
_TARGET_FPS               = 30
_TARGET_INTERVAL          = 1.0 / _TARGET_FPS


def main():
    # Load config from file if it exists, otherwise use defaults
    try:
        cfg = GestureConfig.from_json("config.json")
        print("Loaded config from config.json")
    except FileNotFoundError:
        cfg = GestureConfig()
        print("Using default configuration")

    proc = GestureProcessor(cfg)

    try:
        cap, width, height = proc.init_camera()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    proc.init_keyboard()

    try:
        proc.init_landmarker()
    except Exception as e:
        print(f"Error initializing MediaPipe: {e}")
        cap.release()
        sys.exit(1)

    # Graceful shutdown on Ctrl+C
    def _signal_handler(sig, frame):
        print("\nShutting down...")
        proc.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)

    print("\nGesture controller running.")
    print("  Left hand  - open palm to enable WASD control")
    print("  Right hand - pinch / thumbs-up / flat palm / point")
    print("  Press Ctrl+C to quit\n")

    prev_time            = time.time()
    consecutive_failures = 0

    while True:
        frame_start = time.time()

        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                print(
                    f"Error: Failed to capture frame {consecutive_failures} times in a row."
                )
                break
            print(
                f"Warning: Dropped frame ({consecutive_failures}/{_MAX_CONSECUTIVE_FAILURES}), retrying..."
            )
            time.sleep(0.01)
            continue

        consecutive_failures = 0  # reset on successful read

        annotated = proc.process_frame(frame)

        now       = time.time()
        fps       = 1 / max(now - prev_time, 1e-6)
        prev_time = now
        proc.draw_fps(annotated, fps)

        cv2.imshow("Gesture Controller", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\nQuitting...")
            break
        elif key == ord("r"):
            proc.reset_control()

        # Rate cap — keep loop at TARGET_FPS, don't overwhelm the driver
        elapsed   = time.time() - frame_start
        remaining = _TARGET_INTERVAL - elapsed
        if remaining > 0:
            time.sleep(remaining)

    proc.cleanup()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()