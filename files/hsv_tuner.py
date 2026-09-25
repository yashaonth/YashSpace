"""
HSV Range Tuner
----------------
Run this BEFORE air_canvas.py to find the right HSV lower/upper bounds
for your specific red and blue markers under your room's lighting.

Hold your marker in front of the webcam and drag the sliders until only
the marker shows up white in the "Mask" window (everything else black).
Then press 'q' and copy the printed values into air_canvas.py.

Run it once for red, note the values, then run it again for blue.
"""

import cv2
import numpy as np


def nothing(x):
    pass


cv2.namedWindow("Trackbars")
cv2.createTrackbar("L-H", "Trackbars", 0, 180, nothing)
cv2.createTrackbar("L-S", "Trackbars", 120, 255, nothing)
cv2.createTrackbar("L-V", "Trackbars", 70, 255, nothing)
cv2.createTrackbar("U-H", "Trackbars", 10, 180, nothing)
cv2.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U-V", "Trackbars", 255, 255, nothing)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L-H", "Trackbars")
    l_s = cv2.getTrackbarPos("L-S", "Trackbars")
    l_v = cv2.getTrackbarPos("L-V", "Trackbars")
    u_h = cv2.getTrackbarPos("U-H", "Trackbars")
    u_s = cv2.getTrackbarPos("U-S", "Trackbars")
    u_v = cv2.getTrackbarPos("U-V", "Trackbars")

    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"Lower: [{l_h}, {l_s}, {l_v}]")
        print(f"Upper: [{u_h}, {u_s}, {u_v}]")
        break

cap.release()
cv2.destroyAllWindows()
