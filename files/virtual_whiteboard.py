#!/usr/bin/env python3
"""
Gesture-Driven Virtual Whiteboard
==================================

A real-time, two-window (webcam + whiteboard) application that lets a user
draw, erase, zoom/pan and clear a persistent virtual canvas using nothing
but their hands, tracked with MediaPipe Hands. Gestures can optionally be
"color-gated": actions only fire when a red or blue marker is detected near
the relevant hand region, which drastically cuts down on false positives
caused by ambient hand motion.

Dependencies
------------
    pip install opencv-python mediapipe numpy

Controls
--------
    'c'    Toggle color gating ON/OFF
    'q'    Quit
    ESC    Quit

High-level architecture
------------------------
- HandAnalyzer      : pure geometry -> per-hand semantic features
                       (finger extension, fist/pinch, palm-facing-camera).
- ColorGate         : HSV segmentation + blob bookkeeping for the red/blue
                       marker system.
- PointSmoother     : temporal smoothing (moving average blended with an
                       exponential moving average) for jitter-free strokes.
- ShapeOptimizer    : classifies a finished freehand stroke as a line,
                       circle, rectangle, triangle, or leaves it freeform,
                       and rasterizes the result.
- WhiteboardCanvas  : the persistent 1920x1080 drawing surface, plus the
                       zoom/pan viewport transform.
- GestureStateMachine: the deterministic, mutually-exclusive FSM described
                       in the spec (CLEAR > ZOOM_PAN > STOP > DRAW/ERASE > IDLE).
- WhiteboardController: translates FSM decisions into concrete canvas
                       operations, and owns all of the zoom/pan/drawing
                       temporal state.
"""

import math
import time
from collections import deque
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

# ==========================================================================
# 1. Global configuration & tunable constants
# ==========================================================================

CAM_WIDTH, CAM_HEIGHT = 640, 480          # MediaPipe processing resolution
BOARD_WIDTH, BOARD_HEIGHT = 1920, 1080    # Persistent whiteboard resolution

MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

# --- Temporal smoothing -----------------------------------------------
SMOOTHING_WINDOW = 5     # N: moving-average window size for stroke points
EMA_ALPHA = 0.5          # exponential-smoothing factor blended with the MA

# --- Fist / pinch ("CLEAR ALL") detection ------------------------------
# tau_pinch: mean(fingertip -> palm-center distance) / hand_scale. Below
# this ratio, a hand is considered curled into a fist/pinch.
FIST_DISTANCE_RATIO = 0.55
CLEAR_HOLD_FRAMES = 8    # frames the double-fist pose must be sustained
                          # before the board is wiped (debounce against a
                          # single noisy detection triggering a wipe).

# --- Finger extension -----------------------------------------------------
# A finger counts as "extended" when its tip is farther from the wrist than
# its PIP/IP joint by at least this ratio. Scale-invariant by construction
# (both distances shrink/grow together as the hand moves toward/away from
# the camera), which is why no absolute pixel threshold is needed.
FINGER_EXTENSION_RATIO = 1.05

# --- Dual-hand zoom / pan ------------------------------------------------
# epsilon: minimum per-frame change in (smoothed, camera-width-normalized)
# inter-wrist distance required to register as an intentional zoom, as
# opposed to natural hand tremor.
ZOOM_EPSILON = 0.010
ZOOM_SENSITIVITY = 1.5
MIN_ZOOM, MAX_ZOOM = 0.3, 3.0

# --- Eraser ---------------------------------------------------------------
ERASE_RADIUS = 40  # R_erase, in board pixels

# --- HSV color segmentation ------------------------------------------------
# Red wraps around hue 0/180 in HSV, so it needs two ranges OR'd together.
RED_LOWER_1 = np.array([0, 120, 70])
RED_UPPER_1 = np.array([10, 255, 255])
RED_LOWER_2 = np.array([170, 120, 70])
RED_UPPER_2 = np.array([180, 255, 255])
BLUE_LOWER = np.array([100, 150, 50])
BLUE_UPPER = np.array([140, 255, 255])

COLOR_BLOB_MIN_AREA = 150        # px^2, rejects noise speckles
COLOR_PROXIMITY_RADIUS = 60      # px, "near the fingertip/palm" test

# --- Stroke rendering -------------------------------------------------
STROKE_LINE_THICKNESS = 4
SHAPE_SNAP_MIN_POINTS = 8        # minimum points before shape classification
                                  # is attempted; short strokes stay freeform.

# --- Misc -------------------------------------------------------------
FLASH_MESSAGE_FRAMES = 20        # how long a HUD flash (e.g. "CANVAS CLEARED")
                                  # stays on screen, in frames.

# ==========================================================================
# MediaPipe Hands landmark indices (for readability, mirrors the official
# 21-point topology: https://google.github.io/mediapipe/solutions/hands)
# ==========================================================================
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

# Maps each finger name -> (TIP index, reference joint index) used by the
# extension test. The reference joint is the PIP for the four long fingers
# and the IP for the thumb (the thumb has no PIP).
FINGER_JOINTS = {
    "thumb": (THUMB_TIP, THUMB_IP),
    "index": (INDEX_TIP, INDEX_PIP),
    "middle": (MIDDLE_TIP, MIDDLE_PIP),
    "ring": (RING_TIP, RING_PIP),
    "pinky": (PINKY_TIP, PINKY_PIP),
}


class Mode(Enum):
    IDLE = auto()
    DRAW = auto()
    ERASE = auto()
    STOP = auto()
    ZOOM_PAN = auto()
    CLEAR = auto()


# ==========================================================================
# 2. Small geometry helpers
# ==========================================================================

def euclid_xy(a, b) -> float:
    """Euclidean distance between two MediaPipe landmarks in the xy-plane."""
    return math.hypot(a.x - b.x, a.y - b.y)


def map_to_board(pt: Tuple[int, int], cam_w: int, cam_h: int,
                  board_w: int, board_h: int) -> Tuple[int, int]:
    """Map a point in camera-resolution pixel space to board-resolution
    pixel space (simple linear scaling; both spaces share the same
    normalized [0,1] origin/orientation)."""
    x = int(pt[0] / cam_w * board_w)
    y = int(pt[1] / cam_h * board_h)
    x = max(0, min(board_w - 1, x))
    y = max(0, min(board_h - 1, y))
    return x, y


# ==========================================================================
# 3. HandAnalyzer — per-hand geometric feature extraction
# ==========================================================================

class HandAnalyzer:
    """Stateless geometry helpers that turn raw MediaPipe landmarks into the
    semantic features the gesture FSM needs: which fingers are extended,
    whether the hand is a fist/pinch, and whether the palm faces the camera.
    """

    @staticmethod
    def get_finger_states(lm) -> Dict[str, bool]:
        wrist = lm[WRIST]
        states = {}
        for name, (tip_idx, ref_idx) in FINGER_JOINTS.items():
            d_tip = euclid_xy(wrist, lm[tip_idx])
            d_ref = euclid_xy(wrist, lm[ref_idx])
            states[name] = d_tip > d_ref * FINGER_EXTENSION_RATIO
        return states

    @staticmethod
    def palm_center(lm) -> np.ndarray:
        """Approximate palm center as the centroid of the wrist and the four
        (non-thumb) MCP joints — a stable, roughly-planar reference point."""
        idxs = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
        xs = float(np.mean([lm[i].x for i in idxs]))
        ys = float(np.mean([lm[i].y for i in idxs]))
        return np.array([xs, ys])

    @staticmethod
    def hand_scale(lm) -> float:
        """A rough, orientation-independent measure of apparent hand size,
        used to make the fist/pinch test scale-invariant."""
        return euclid_xy(lm[WRIST], lm[MIDDLE_MCP]) + 1e-6

    @staticmethod
    def is_fist(lm) -> bool:
        center = HandAnalyzer.palm_center(lm)
        scale = HandAnalyzer.hand_scale(lm)
        dists = [math.hypot(center[0] - lm[t].x, center[1] - lm[t].y) for t in FINGER_TIPS]
        mean_ratio = (sum(dists) / len(dists)) / scale
        return mean_ratio < FIST_DISTANCE_RATIO

    @staticmethod
    def palm_facing_camera(lm, handedness_label: str) -> bool:
        """Estimate whether the palm faces the camera via the normal of the
        plane spanned by WRIST -> INDEX_MCP and WRIST -> PINKY_MCP.

        NOTE: the input frame is flipped (mirrored) *before* being handed to
        MediaPipe, so `handedness_label` ("Left"/"Right") already matches
        what the user sees on screen (selfie-view convention). A left and a
        right hand trace the wrist->index->pinky loop with opposite winding,
        so the sign test below is intentionally inverted between them. This
        is a lightweight heuristic, not a full 3D pose solve — it is
        reliable for a hand held roughly upright facing toward/away from the
        camera, and may need the sign flipped for a particular camera/lens.
        """
        wrist, i_mcp, p_mcp = lm[WRIST], lm[INDEX_MCP], lm[PINKY_MCP]
        v1 = np.array([i_mcp.x - wrist.x, i_mcp.y - wrist.y, i_mcp.z - wrist.z])
        v2 = np.array([p_mcp.x - wrist.x, p_mcp.y - wrist.y, p_mcp.z - wrist.z])
        normal_z = np.cross(v1, v2)[2]
        return normal_z < 0 if handedness_label == "Right" else normal_z > 0


# ==========================================================================
# 4. ColorGate — HSV segmentation for the red/blue marker system
# ==========================================================================

class ColorGate:
    """Detects red and blue color blobs (e.g. colored fingertip caps, rings
    or cards) and answers proximity queries used to spatially confirm
    draw/erase gestures."""

    def __init__(self):
        self._kernel = np.ones((5, 5), np.uint8)

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=2)
        return mask

    def _contours(self, mask: np.ndarray) -> List[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in contours if cv2.contourArea(c) >= COLOR_BLOB_MIN_AREA]

    def detect_red(self, hsv: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        m1 = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
        m2 = cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
        mask = self._clean_mask(cv2.bitwise_or(m1, m2))
        return mask, self._contours(mask)

    def detect_blue(self, hsv: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        mask = self._clean_mask(cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER))
        return mask, self._contours(mask)

    @staticmethod
    def blob_centroids(contours: List[np.ndarray]) -> List[Tuple[int, int]]:
        centroids = []
        for c in contours:
            m = cv2.moments(c)
            if m["m00"] > 0:
                centroids.append((int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])))
        return centroids

    @staticmethod
    def any_blob_near(centroids: List[Tuple[int, int]],
                       point: Optional[Tuple[int, int]],
                       radius: int = COLOR_PROXIMITY_RADIUS) -> bool:
        if point is None:
            return False
        return any(math.hypot(c[0] - point[0], c[1] - point[1]) <= radius for c in centroids)


# ==========================================================================
# 5. PointSmoother — temporal smoothing for stroke coordinates
# ==========================================================================

class PointSmoother:
    """Stabilizes a noisy fingertip trajectory with a moving-average filter
    (window N) blended 50/50 with an exponential moving average, so strokes
    stay responsive but jitter-free."""

    def __init__(self, window: int = SMOOTHING_WINDOW, alpha: float = EMA_ALPHA):
        self._window: deque = deque(maxlen=window)
        self._alpha = alpha
        self._ema: Optional[np.ndarray] = None

    def update(self, point: Tuple[int, int]) -> Tuple[int, int]:
        self._window.append(point)
        moving_avg = np.mean(np.array(self._window), axis=0)
        pt = np.array(point, dtype=np.float64)
        self._ema = pt if self._ema is None else self._alpha * pt + (1 - self._alpha) * self._ema
        blended = 0.5 * moving_avg + 0.5 * self._ema
        return int(blended[0]), int(blended[1])

    def reset(self):
        self._window.clear()
        self._ema = None


# ==========================================================================
# 6. ShapeOptimizer — freehand stroke -> perfect primitive classification
# ==========================================================================

class ShapeOptimizer:
    """Classifies a finished stroke's polyline as a line, circle, rectangle,
    triangle, or leaves it as smoothed freeform handwriting, and knows how
    to render whichever primitive was chosen."""

    @staticmethod
    def classify(points: List[Tuple[int, int]]) -> dict:
        if len(points) < SHAPE_SNAP_MIN_POINTS:
            return {"type": "freeform", "points": points}

        pts = np.array(points, dtype=np.int32)
        path_length = float(sum(
            np.linalg.norm(pts[i] - pts[i - 1]) for i in range(1, len(pts))
        ))
        endpoint_dist = float(np.linalg.norm(pts[-1] - pts[0]))

        # --- 1. Straight line: path length ~= straight-line endpoint distance.
        if endpoint_dist > 1e-3 and path_length / endpoint_dist < 1.15:
            return {"type": "line", "p1": tuple(int(v) for v in pts[0]),
                    "p2": tuple(int(v) for v in pts[-1])}

        # A stroke is treated as "closed" (candidate for circle/rect/triangle)
        # if it loops back near its own starting point.
        is_closed = endpoint_dist < 0.15 * path_length
        if not is_closed:
            return {"type": "freeform", "points": points}

        contour = pts.reshape(-1, 1, 2)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * math.pi * area / (perimeter ** 2)) if perimeter > 1e-3 else 0.0

        # --- 2. Circle / ellipse.
        if circularity > 0.80:
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            return {"type": "circle", "center": (int(cx), int(cy)), "radius": int(radius)}

        # --- 3 & 4. Rectangle / triangle via polygon approximation.
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4 and ShapeOptimizer._is_rectangle(approx):
            return {"type": "rectangle", "points": [tuple(int(v) for v in p[0]) for p in approx]}
        if len(approx) == 3:
            return {"type": "triangle", "points": [tuple(int(v) for v in p[0]) for p in approx]}

        # --- 5. Freeform fallback.
        return {"type": "freeform", "points": points}

    @staticmethod
    def _is_rectangle(approx: np.ndarray) -> bool:
        pts = approx.reshape(4, 2)
        angles = []
        for i in range(4):
            p_prev, p_curr, p_next = pts[i - 1], pts[i], pts[(i + 1) % 4]
            v1, v2 = p_prev - p_curr, p_next - p_curr
            denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
            cos_angle = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
            angles.append(math.degrees(math.acos(cos_angle)))
        return all(abs(a - 90.0) <= 15.0 for a in angles)

    @staticmethod
    def render(canvas: np.ndarray, shape: dict, color: Tuple[int, int, int], thickness: int):
        t = shape["type"]
        if t == "line":
            cv2.line(canvas, shape["p1"], shape["p2"], color, thickness, cv2.LINE_AA)
        elif t == "circle":
            cv2.circle(canvas, shape["center"], shape["radius"], color, thickness, cv2.LINE_AA)
        elif t in ("rectangle", "triangle"):
            pts = np.array(shape["points"], dtype=np.int32)
            cv2.polylines(canvas, [pts], True, color, thickness, cv2.LINE_AA)
        else:  # freeform
            pts = np.array(shape["points"], dtype=np.int32)
            if len(pts) == 1:
                cv2.circle(canvas, tuple(pts[0]), max(1, thickness // 2 + 1), color, -1)
            else:
                cv2.polylines(canvas, [pts], False, color, thickness, cv2.LINE_AA)


# ==========================================================================
# 7. WhiteboardCanvas — the persistent drawing surface + zoom/pan viewport
# ==========================================================================

class WhiteboardCanvas:
    def __init__(self, width: int = BOARD_WIDTH, height: int = BOARD_HEIGHT):
        self.width = width
        self.height = height
        self.base_canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        self.zoom = 1.0
        self.pan = np.array([0.0, 0.0])          # board-pixel-space offset
        self.active_stroke: List[Tuple[int, int]] = []
        self.pen_color = (0, 0, 200)             # BGR

    # --- editing -----------------------------------------------------
    def clear(self):
        self.base_canvas[:] = 255
        self.active_stroke = []

    def start_or_continue_stroke(self, board_pt: Tuple[int, int]):
        self.active_stroke.append(board_pt)

    def commit_stroke(self):
        """Finalize the in-progress stroke: classify it and rasterize the
        resulting primitive (or the freeform polyline) onto the permanent
        canvas. Called whenever a stroke ends (hand drops, gesture changes,
        or a mode transition interrupts drawing)."""
        if len(self.active_stroke) >= 2:
            shape = ShapeOptimizer.classify(self.active_stroke)
            ShapeOptimizer.render(self.base_canvas, shape, self.pen_color, STROKE_LINE_THICKNESS)
        elif len(self.active_stroke) == 1:
            cv2.circle(self.base_canvas, self.active_stroke[0],
                       STROKE_LINE_THICKNESS // 2 + 1, self.pen_color, -1)
        self.active_stroke = []

    def erase_at(self, board_pt: Tuple[int, int], radius: int = ERASE_RADIUS):
        cv2.circle(self.base_canvas, board_pt, radius, (255, 255, 255), -1)

    # --- rendering -----------------------------------------------------
    def get_composite(self) -> np.ndarray:
        """Permanent canvas + a live preview of the in-progress stroke,
        without mutating the permanent canvas."""
        composite = self.base_canvas.copy()
        if len(self.active_stroke) >= 2:
            pts = np.array(self.active_stroke, dtype=np.int32)
            cv2.polylines(composite, [pts], False, self.pen_color, STROKE_LINE_THICKNESS, cv2.LINE_AA)
        return composite

    def get_crop_rect(self) -> Tuple[int, int, int, int]:
        zoom = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom))
        crop_w = max(1, int(self.width / zoom))
        crop_h = max(1, int(self.height / zoom))
        cx = int(self.width / 2 + self.pan[0])
        cy = int(self.height / 2 + self.pan[1])
        x1 = max(0, min(self.width - crop_w, cx - crop_w // 2))
        y1 = max(0, min(self.height - crop_h, cy - crop_h // 2))
        return x1, y1, crop_w, crop_h

    def get_view(self) -> np.ndarray:
        """Apply the current zoom/pan as a crop + resize, producing the
        image that is actually shown in the whiteboard window."""
        composite = self.get_composite()
        x1, y1, crop_w, crop_h = self.get_crop_rect()
        cropped = composite[y1:y1 + crop_h, x1:x1 + crop_w]
        return cv2.resize(cropped, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

    def board_to_view_coords(self, pt: Tuple[int, int]) -> Tuple[int, int]:
        x1, y1, crop_w, crop_h = self.get_crop_rect()
        vx = int((pt[0] - x1) / crop_w * self.width)
        vy = int((pt[1] - y1) / crop_h * self.height)
        return max(0, min(self.width - 1, vx)), max(0, min(self.height - 1, vy))


# ==========================================================================
# 8. GestureStateMachine — deterministic, mutually-exclusive gesture arbitration
# ==========================================================================

class GestureStateMachine:
    """Implements the priority-ordered FSM:
       CLEAR (sustained double-fist) > ZOOM_PAN (both palms facing camera) >
       STOP (both colors visible) > DRAW / ERASE > IDLE.
    """

    def __init__(self):
        self._fist_sustain_counter = 0

    def evaluate(self, hands_info: List[dict], red_detected: bool, blue_detected: bool,
                 red_centroids: List[Tuple[int, int]], blue_centroids: List[Tuple[int, int]],
                 color_gate_enabled: bool) -> Mode:

        # --- 1. CLEAR ALL: both hands present and both are a sustained fist/pinch.
        if len(hands_info) == 2 and hands_info[0]["is_fist"] and hands_info[1]["is_fist"]:
            self._fist_sustain_counter += 1
            return Mode.CLEAR if self._fist_sustain_counter >= CLEAR_HOLD_FRAMES else Mode.IDLE
        self._fist_sustain_counter = 0

        # --- 2. DUAL-HAND ZOOM & PAN: both hands visible, palms facing camera.
        if len(hands_info) == 2 and hands_info[0]["facing_camera"] and hands_info[1]["facing_camera"]:
            return Mode.ZOOM_PAN

        # --- 3. STOP / STANDBY: both red and blue visible simultaneously.
        if color_gate_enabled and red_detected and blue_detected:
            return Mode.STOP

        if not hands_info:
            return Mode.IDLE

        # --- 4 & 5. DRAW / ERASE (single dominant hand = first detected hand).
        dominant = hands_info[0]
        if color_gate_enabled:
            # Gated mode: the color marker IS the gesture trigger. "Spatial
            # confirmation" means the marker must additionally be near the
            # relevant hand region (fingertip for draw, palm for erase).
            draw_cond = (red_detected and not blue_detected and
                         ColorGate.any_blob_near(red_centroids, dominant["index_tip_px"]))
            erase_cond = (blue_detected and not red_detected and
                          ColorGate.any_blob_near(blue_centroids, dominant["palm_px"]))
        else:
            # Ungated mode: rely purely on geometric hand posture.
            fs = dominant["finger_states"]
            draw_cond = fs["index"] and not fs["middle"] and not fs["ring"] and not fs["pinky"]
            erase_cond = sum(fs.values()) >= 4  # open-palm swipe

        if draw_cond:
            return Mode.DRAW
        if erase_cond:
            return Mode.ERASE
        return Mode.IDLE


# ==========================================================================
# 9. WhiteboardController — bridges FSM decisions to canvas operations
# ==========================================================================

class WhiteboardController:
    def __init__(self, board: WhiteboardCanvas):
        self.board = board
        self.index_smoother = PointSmoother()
        self.palm_smoother = PointSmoother()
        self._wrist_dist_history: deque = deque(maxlen=SMOOTHING_WINDOW)
        self._prev_smoothed_dist: Optional[float] = None
        self._prev_midpoint: Optional[Tuple[int, int]] = None
        self.current_mode = Mode.IDLE

    def _reset_zoom_tracking(self):
        self._prev_smoothed_dist = None
        self._prev_midpoint = None
        self._wrist_dist_history.clear()

    def update(self, new_mode: Mode, hands_info: List[dict]) -> Optional[str]:
        flash: Optional[str] = None

        # A stroke "finishes" the instant we leave DRAW mode for anything
        # else (hand dropped, gesture changed, color lost, etc.).
        if self.current_mode == Mode.DRAW and new_mode != Mode.DRAW:
            self.board.commit_stroke()
            self.index_smoother.reset()

        if new_mode != Mode.ZOOM_PAN:
            self._reset_zoom_tracking()

        if new_mode == Mode.CLEAR:
            if self.current_mode != Mode.CLEAR:
                self.board.clear()
                flash = "CANVAS CLEARED"

        elif new_mode == Mode.ZOOM_PAN:
            self._apply_zoom_pan(hands_info)

        elif new_mode == Mode.STOP:
            pass  # freeze: no drawing, erasing, or clearing while both colors are visible

        elif new_mode == Mode.DRAW:
            dominant = hands_info[0]
            smoothed_pt = self.index_smoother.update(dominant["index_tip_px"])
            board_pt = map_to_board(smoothed_pt, CAM_WIDTH, CAM_HEIGHT, self.board.width, self.board.height)
            self.board.start_or_continue_stroke(board_pt)

        elif new_mode == Mode.ERASE:
            dominant = hands_info[0]
            smoothed_pt = self.palm_smoother.update(dominant["palm_px"])
            board_pt = map_to_board(smoothed_pt, CAM_WIDTH, CAM_HEIGHT, self.board.width, self.board.height)
            self.board.erase_at(board_pt)

        self.current_mode = new_mode
        return flash

    def _apply_zoom_pan(self, hands_info: List[dict]):
        w0, w1 = hands_info[0]["wrist_px"], hands_info[1]["wrist_px"]

        # Distance, normalized by camera width so the metric is resolution-
        # independent, then smoothed to reject frame-to-frame jitter.
        dist = math.hypot(w0[0] - w1[0], w0[1] - w1[1]) / CAM_WIDTH
        self._wrist_dist_history.append(dist)
        smoothed = float(np.mean(self._wrist_dist_history))

        if self._prev_smoothed_dist is not None:
            delta = smoothed - self._prev_smoothed_dist
            # Hands moving apart (delta > 0) -> magnify ("zoom in").
            # Hands moving together (delta < 0) -> shrink ("zoom out").
            if abs(delta) > ZOOM_EPSILON:
                factor = 1.0 + delta * ZOOM_SENSITIVITY
                self.board.zoom = float(np.clip(self.board.zoom * factor, MIN_ZOOM, MAX_ZOOM))
        self._prev_smoothed_dist = smoothed

        # Pan follows the midpoint of the two wrists.
        midpoint = ((w0[0] + w1[0]) // 2, (w0[1] + w1[1]) // 2)
        if self._prev_midpoint is not None:
            dx = (midpoint[0] - self._prev_midpoint[0]) * (self.board.width / CAM_WIDTH)
            dy = (midpoint[1] - self._prev_midpoint[1]) * (self.board.height / CAM_HEIGHT)
            self.board.pan[0] -= dx / self.board.zoom
            self.board.pan[1] -= dy / self.board.zoom
        self._prev_midpoint = midpoint


# ==========================================================================
# 10. HUD rendering
# ==========================================================================

def draw_hud(frame: np.ndarray, mode: Mode, color_gate_enabled: bool,
             fps: float, flash_message: Optional[str] = None):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"Mode: {mode.name}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Color Gate: {'ENABLED' if color_gate_enabled else 'DISABLED'}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 140, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "'c' gate  'q'/ESC quit", (w - 260, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    if flash_message:
        text_size = cv2.getTextSize(flash_message, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        tx = (w - text_size[0]) // 2
        ty = h // 2
        cv2.putText(frame, flash_message, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)


# ==========================================================================
# 11. Per-hand feature extraction (bridges MediaPipe output -> HandAnalyzer)
# ==========================================================================

def build_hand_info(lm, label: str) -> dict:
    finger_states = HandAnalyzer.get_finger_states(lm)
    fist = HandAnalyzer.is_fist(lm)
    facing = HandAnalyzer.palm_facing_camera(lm, label)

    index_tip_px = (int(lm[INDEX_TIP].x * CAM_WIDTH), int(lm[INDEX_TIP].y * CAM_HEIGHT))
    wrist_px = (int(lm[WRIST].x * CAM_WIDTH), int(lm[WRIST].y * CAM_HEIGHT))
    center_norm = HandAnalyzer.palm_center(lm)
    palm_px = (int(center_norm[0] * CAM_WIDTH), int(center_norm[1] * CAM_HEIGHT))

    return {
        "label": label,
        "finger_states": finger_states,
        "is_fist": fist,
        "facing_camera": facing,
        "index_tip_px": index_tip_px,
        "wrist_px": wrist_px,
        "palm_px": palm_px,
    }


# ==========================================================================
# 12. main()
# ==========================================================================

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Hand connection pairs for drawing landmarks manually
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0). Check camera permissions/connection.")

    # Configure MediaPipe Tasks HandLandmarker
    base_options = mp_python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=MAX_NUM_HANDS,
        min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_TRACKING_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)

    board = WhiteboardCanvas()
    color_gate = ColorGate()
    fsm = GestureStateMachine()
    controller = WhiteboardController(board)

    color_gate_enabled = True
    flash_text = ""
    flash_timer = 0
    prev_time = time.time()
    frame_timestamp_ms = 0

    cam_window = "Gesture Control - Webcam"
    board_window = "Virtual Whiteboard"
    cv2.namedWindow(cam_window)
    cv2.namedWindow(board_window)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))
            display_frame = frame.copy()

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_timestamp_ms = int(time.time() * 1000)
            results = detector.detect_for_video(mp_image, frame_timestamp_ms)

            red_mask, red_contours = color_gate.detect_red(hsv)
            blue_mask, blue_contours = color_gate.detect_blue(hsv)
            red_centroids = ColorGate.blob_centroids(red_contours)
            blue_centroids = ColorGate.blob_centroids(blue_contours)
            red_detected = len(red_centroids) > 0
            blue_detected = len(blue_centroids) > 0

            for c in red_contours:
                x, y, bw, bh = cv2.boundingRect(c)
                cv2.rectangle(display_frame, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
            for c in blue_contours:
                x, y, bw, bh = cv2.boundingRect(c)
                cv2.rectangle(display_frame, (x, y), (x + bw, y + bh), (255, 0, 0), 2)

            hands_info: List[dict] = []
            if results.hand_landmarks and results.handedness:
                for hand_landmarks, handedness in zip(results.hand_landmarks, results.handedness):
                    label = handedness[0].category_name
                    info = build_hand_info(hand_landmarks, label)
                    hands_info.append(info)

                    # Draw skeleton connections
                    for p1_idx, p2_idx in HAND_CONNECTIONS:
                        pt1 = (int(hand_landmarks[p1_idx].x * CAM_WIDTH), int(hand_landmarks[p1_idx].y * CAM_HEIGHT))
                        pt2 = (int(hand_landmarks[p2_idx].x * CAM_WIDTH), int(hand_landmarks[p2_idx].y * CAM_HEIGHT))
                        cv2.line(display_frame, pt1, pt2, (0, 255, 0), 2)

                    for lm_pt in hand_landmarks:
                        cx, cy = int(lm_pt.x * CAM_WIDTH), int(lm_pt.y * CAM_HEIGHT)
                        cv2.circle(display_frame, (cx, cy), 3, (0, 0, 255), -1)

                    cv2.circle(display_frame, info["index_tip_px"], 8, (0, 255, 255), -1)
                    cv2.circle(display_frame, info["palm_px"], 6, (255, 0, 255), -1)

            new_mode = fsm.evaluate(hands_info, red_detected, blue_detected,
                                     red_centroids, blue_centroids, color_gate_enabled)
            flash = controller.update(new_mode, hands_info)
            current_mode = controller.current_mode
            if flash:
                flash_text, flash_timer = flash, FLASH_MESSAGE_FRAMES
            elif flash_timer > 0:
                flash_timer -= 1

            now = time.time()
            fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
            prev_time = now
            draw_hud(display_frame, current_mode, color_gate_enabled, fps,
                     flash_text if flash_timer > 0 else None)

            board_view = board.get_view()
            if hands_info:
                cursor_board_pt = map_to_board(hands_info[0]["index_tip_px"], CAM_WIDTH, CAM_HEIGHT,
                                                board.width, board.height)
                cursor_view_pt = board.board_to_view_coords(cursor_board_pt)
                cv2.drawMarker(board_view, cursor_view_pt, (0, 0, 0), cv2.MARKER_CROSS, 24, 2)

            cv2.imshow(cam_window, display_frame)
            cv2.imshow(board_window, board_view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                color_gate_enabled = not color_gate_enabled
            elif key == ord('q') or key == 27:
                break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
