import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

CIRCLE_PATH = "circle.png"
MODEL_PATH = "hand_landmarker.task"
CIRCLE_SIZE = 60          # pixel diameter the overlay is resized to
INDEX_TIP = 8              # landmark index for tip of index finger


def create_default_circle_png(path, size=200, color=(0, 255, 255)):
    """Generate a simple filled circle with a dark ring, saved as BGRA PNG."""
    img = np.zeros((size, size, 4), dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = size // 2 - 8
    cv2.circle(img, center, radius, (*color, 220), -1)          # filled, semi-transparent
    cv2.circle(img, center, radius, (0, 0, 0, 255), 4)           # dark border
    cv2.imwrite(path, img)


def overlay_transparent(background, overlay, cx, cy):
    """Alpha-blend a BGRA overlay image onto background, centered at (cx, cy)."""
    h, w = overlay.shape[:2]
    x, y = cx - w // 2, cy - h // 2
    bg_h, bg_w = background.shape[:2]

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, bg_w), min(y + h, bg_h)
    if x1 >= x2 or y1 >= y2:
        return background  # fully off-screen, nothing to draw

    crop = overlay[y1 - y:y2 - y, x1 - x:x2 - x]
    alpha = crop[:, :, 3:4] / 255.0
    background[y1:y2, x1:x2] = (
        alpha * crop[:, :, :3] + (1 - alpha) * background[y1:y2, x1:x2]
    )
    return background


def main():
    if not os.path.exists(CIRCLE_PATH):
        create_default_circle_png(CIRCLE_PATH)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Missing {MODEL_PATH}. Download it using:\n"
            "wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )

    circle_img = cv2.imread(CIRCLE_PATH, cv2.IMREAD_UNCHANGED)
    if circle_img.shape[2] != 4:
        raise ValueError("circle.png must have an alpha (transparency) channel.")
    circle_img = cv2.resize(circle_img, (CIRCLE_SIZE, CIRCLE_SIZE))

    # Configure MediaPipe Hand Landmarker Task
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Convert OpenCV BGR frame to MediaPipe Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Process hand landmarks
        results = landmarker.detect(mp_image)

        if results.hand_landmarks:
            landmark = results.hand_landmarks[0][INDEX_TIP]
            x, y = int(landmark.x * w), int(landmark.y * h)
            overlay_transparent(frame, circle_img, x, y)

        cv2.imshow("Hand Circle Overlay - press 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()