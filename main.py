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
    print("  - Show your hands to the camera for tracking")
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
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Cleanup complete. Goodbye!")


if __name__ == "__main__":
    main()
