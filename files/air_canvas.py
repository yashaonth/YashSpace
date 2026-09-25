"""
Color-Tracking Air Canvas
--------------------------
RED marker  -> draws
BLUE marker -> erases
BOTH red & blue visible at once -> pauses
Neither visible -> pauses

Controls:
  c - clear canvas
  q - quit
"""

import cv2
import numpy as np

# ---------- 1. Tune these HSV ranges for your marker colors / lighting ----------
# Run hsv_tuner.py first and paste the values it prints here.
# Red wraps around hue 0, so it needs two ranges.
RED_LOWER1 = np.array([0, 120, 70])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([170, 120, 70])
RED_UPPER2 = np.array([180, 255, 255])

BLUE_LOWER = np.array([94, 80, 2])
BLUE_UPPER = np.array([126, 255, 255])

MIN_CONTOUR_AREA = 800   # ignore small noise blobs
DRAW_THICKNESS = 6
ERASE_RADIUS = 30


def get_largest_blob_center(mask):
    """Return (x, y, radius) of the largest contour in mask, or None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
        return None
    (x, y), radius = cv2.minEnclosingCircle(largest)
    return int(x), int(y), int(radius)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Could not read from webcam.")
        return

    h, w = frame.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)   # black drawing layer
    prev_point = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)   # mirror for natural interaction
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ---- build masks ----
        red_mask = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1) | cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
        blue_mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)

        # clean up noise
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

        red_blob = get_largest_blob_center(red_mask)
        blue_blob = get_largest_blob_center(blue_mask)

        red_present = red_blob is not None
        blue_present = blue_blob is not None

        # ---- gesture logic ----
        if red_present and blue_present:
            prev_point = None
            status = "PAUSED (both colors detected)"

        elif red_present:
            x, y, r = red_blob
            cv2.circle(frame, (x, y), r, (0, 0, 255), 2)
            if prev_point is not None:
                cv2.line(canvas, prev_point, (x, y), (0, 0, 255), DRAW_THICKNESS)
            prev_point = (x, y)
            status = "DRAWING"

        elif blue_present:
            x, y, r = blue_blob
            cv2.circle(frame, (x, y), r, (255, 0, 0), 2)
            cv2.circle(canvas, (x, y), ERASE_RADIUS, (0, 0, 0), -1)
            prev_point = None
            status = "ERASING"

        else:
            prev_point = None
            status = "PAUSED (no marker detected)"

        # ---- merge canvas onto live feed for display ----
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, inv_mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY_INV)
        inv_mask = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)
        frame_bg = cv2.bitwise_and(frame, inv_mask)
        combined = cv2.add(frame_bg, canvas)

        cv2.putText(combined, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
        cv2.imshow("Air Canvas - press 'c' to clear, 'q' to quit", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas[:] = 0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
