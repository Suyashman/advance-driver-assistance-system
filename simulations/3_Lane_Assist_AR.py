"""
Lane Assist AR Annotation System
=================================
Real-time lane detection and steering guidance overlay.

Architecture:
  - Each feature is a self-contained module/function
  - A FeatureFlags dataclass controls what runs each frame
  - Main loop calls only enabled features → easy to extend later

Future integration points (search "# FUTURE HOOK"):
  - Obstacle detection
  - Road sign detection
  - Adaptive cruise control
  - etc.
"""

import cv2
import numpy as np
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

@dataclass
class Config:
    """Tune detection and display parameters here."""

    # --- ROI (Region of Interest) ---
    roi_top_ratio: float = 0.55        # fraction from top where ROI starts
    roi_bottom_ratio: float = 1.0      # fraction from top where ROI ends
    roi_left_ratio: float = 0.0        # fraction from left
    roi_right_ratio: float = 1.0       # fraction from right

    # --- Canny edge detection ---
    canny_low: int = 50
    canny_high: int = 150

    # --- Hough transform ---
    hough_rho: int = 1
    hough_theta: float = np.pi / 180
    hough_threshold: int = 40
    hough_min_line_len: int = 40
    hough_max_line_gap: int = 100

    # --- Lane classification ---
    left_slope_range: Tuple[float, float] = (-2.5, -0.3)
    right_slope_range: Tuple[float, float] = (0.3, 2.5)

    # --- Steering ---
    center_deadzone_px: int = 30       # pixels — no command inside this range
    max_offset_display: int = 200      # clamps the displayed offset label

    # --- AR path ---
    path_lookahead_ratio: float = 0.25  # how far ahead (fraction of height) the path extends
    path_steps: int = 12               # number of curve segments
    path_width_ratio: float = 0.25     # lane width as fraction of frame width (fallback)

    # --- Visual style ---
    lane_color_left: Tuple = (0, 220, 120)      # green-ish
    lane_color_right: Tuple = (0, 220, 120)
    lane_color_current: Tuple = (50, 200, 255)  # cyan — current lane fill
    path_color: Tuple = (255, 170, 0)            # amber path
    hud_bg_alpha: float = 0.55
    hud_text_color: Tuple = (255, 255, 255)
    hud_warn_color: Tuple = (30, 30, 255)
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.55
    font_thickness: int = 1


@dataclass
class FeatureFlags:
    """Toggle individual features on/off at runtime."""
    lane_detection: bool = True
    steering_guidance: bool = True
    ar_path: bool = True
    # FUTURE HOOK — add new flags here:
    # obstacle_detection: bool = False
    # sign_detection: bool = False
    # adaptive_cruise: bool = False


# ─────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class LaneState:
    """Holds per-frame lane analysis results."""
    left_line: Optional[np.ndarray] = None      # (x1,y1,x2,y2)
    right_line: Optional[np.ndarray] = None
    lane_center_x: Optional[float] = None
    frame_center_x: Optional[float] = None
    offset_px: float = 0.0                      # + = right of center, - = left
    offset_pct: float = 0.0
    steer_cmd: str = "CENTER"                   # "LEFT" | "RIGHT" | "CENTER"
    confidence: float = 0.0                     # 0–1
    current_lane_idx: int = 0                   # which lane (left=0, right=1, etc.)
    all_lines: List = field(default_factory=list)


# ─────────────────────────────────────────────
#  PREPROCESSING
# ─────────────────────────────────────────────

def preprocess_frame(frame: np.ndarray, cfg: Config) -> np.ndarray:
    """Grayscale → Gaussian blur → Canny edges → ROI mask."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, cfg.canny_low, cfg.canny_high)
    mask = _build_roi_mask(edges, frame.shape, cfg)
    return cv2.bitwise_and(edges, mask)


def _build_roi_mask(edges: np.ndarray, shape: tuple, cfg: Config) -> np.ndarray:
    h, w = shape[:2]
    top    = int(h * cfg.roi_top_ratio)
    bottom = int(h * cfg.roi_bottom_ratio)
    left   = int(w * cfg.roi_left_ratio)
    right  = int(w * cfg.roi_right_ratio)
    mask = np.zeros_like(edges)
    poly = np.array([[
        (left,  bottom),
        (left,  top),
        (right, top),
        (right, bottom),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, poly, 255)
    return mask


# ─────────────────────────────────────────────
#  LINE DETECTION & CLASSIFICATION
# ─────────────────────────────────────────────

def detect_lane_lines(edges: np.ndarray, cfg: Config) -> Tuple[List, List]:
    """Run Hough and split into left/right candidate lines."""
    raw = cv2.HoughLinesP(
        edges,
        cfg.hough_rho,
        cfg.hough_theta,
        cfg.hough_threshold,
        minLineLength=cfg.hough_min_line_len,
        maxLineGap=cfg.hough_max_line_gap,
    )
    left_lines, right_lines = [], []
    if raw is None:
        return left_lines, right_lines

    for line in raw:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if cfg.left_slope_range[0] <= slope <= cfg.left_slope_range[1]:
            left_lines.append(line[0])
        elif cfg.right_slope_range[0] <= slope <= cfg.right_slope_range[1]:
            right_lines.append(line[0])
    return left_lines, right_lines


def average_line(lines: List, height: int, cfg: Config) -> Optional[np.ndarray]:
    """Fit a single representative line through detected segments."""
    if not lines:
        return None
    pts = np.array([[x1, y1, x2, y2] for x1, y1, x2, y2 in lines], dtype=np.float32)
    # weighted by line length
    lengths = np.sqrt((pts[:, 2] - pts[:, 0])**2 + (pts[:, 3] - pts[:, 1])**2)
    weights = lengths / lengths.sum()
    avg = (pts * weights[:, None]).sum(axis=0)
    x1, y1, x2, y2 = avg
    # extrapolate to ROI boundaries
    slope = (y2 - y1) / (x2 - x1 + 1e-6)
    intercept = y1 - slope * x1
    y_bottom = int(height * cfg.roi_bottom_ratio)
    y_top    = int(height * cfg.roi_top_ratio)
    x_bottom = int((y_bottom - intercept) / (slope + 1e-6))
    x_top    = int((y_top    - intercept) / (slope + 1e-6))
    return np.array([x_bottom, y_bottom, x_top, y_top])


# ─────────────────────────────────────────────
#  LANE ANALYSIS
# ─────────────────────────────────────────────

def analyse_lane(
    frame: np.ndarray,
    left_line: Optional[np.ndarray],
    right_line: Optional[np.ndarray],
    cfg: Config,
) -> LaneState:
    """Compute lane center, vehicle offset, and steering command."""
    h, w = frame.shape[:2]
    state = LaneState()
    state.frame_center_x = w / 2.0

    # --- lane center ---
    if left_line is not None and right_line is not None:
        lx = left_line[0]    # bottom x of left line
        rx = right_line[0]   # bottom x of right line
        state.lane_center_x = (lx + rx) / 2.0
        state.left_line = left_line
        state.right_line = right_line
        state.confidence = 1.0
    elif left_line is not None:
        state.left_line = left_line
        # estimate right edge
        state.lane_center_x = left_line[0] + w * cfg.path_width_ratio
        state.confidence = 0.5
    elif right_line is not None:
        state.right_line = right_line
        state.lane_center_x = right_line[0] - w * cfg.path_width_ratio
        state.confidence = 0.5
    else:
        state.lane_center_x = w / 2.0
        state.confidence = 0.0

    # offset: positive = vehicle is RIGHT of lane center (needs left steer)
    state.offset_px = state.frame_center_x - state.lane_center_x
    state.offset_pct = (state.offset_px / (w / 2.0)) * 100.0

    # steering command
    dz = cfg.center_deadzone_px
    if state.offset_px > dz:
        state.steer_cmd = "STEER LEFT"
    elif state.offset_px < -dz:
        state.steer_cmd = "STEER RIGHT"
    else:
        state.steer_cmd = "ON CENTER"

    return state


# ─────────────────────────────────────────────
#  AR PATH CALCULATION
# ─────────────────────────────────────────────

def compute_ar_path(
    state: LaneState,
    frame_shape: tuple,
    cfg: Config,
) -> List[Tuple[int, int]]:
    """
    Generate a list of (x, y) points forming the corrective future path.
    The path curves from the current vehicle position toward the lane center.
    """
    h, w = frame_shape[:2]
    points = []
    steps = cfg.path_steps
    y_start = h - 1
    y_end = int(h * (1.0 - cfg.path_lookahead_ratio))

    x_start = int(w / 2.0)                            # vehicle bottom center
    x_end   = int(state.lane_center_x)                # target: lane center

    for i in range(steps + 1):
        t = i / steps
        y = int(y_start + (y_end - y_start) * t)
        # ease-in curve: slow correction near vehicle, sharpen ahead
        t_ease = t ** 1.5
        x = int(x_start + (x_end - x_start) * t_ease)
        points.append((x, y))
    return points


# ─────────────────────────────────────────────
#  DRAWING / HUD
# ─────────────────────────────────────────────

def draw_lane_overlay(frame: np.ndarray, state: LaneState, cfg: Config) -> np.ndarray:
    """Draw detected lane lines and filled lane polygon."""
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # fill current lane
    if state.left_line is not None and state.right_line is not None:
        pts = np.array([
            [state.left_line[0],  state.left_line[1]],
            [state.left_line[2],  state.left_line[3]],
            [state.right_line[2], state.right_line[3]],
            [state.right_line[0], state.right_line[1]],
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 180, 90))
        frame = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)

    # draw lane lines
    lw = 3
    if state.left_line is not None:
        x1, y1, x2, y2 = state.left_line
        cv2.line(frame, (x1, y1), (x2, y2), cfg.lane_color_left, lw, cv2.LINE_AA)
    if state.right_line is not None:
        x1, y1, x2, y2 = state.right_line
        cv2.line(frame, (x1, y1), (x2, y2), cfg.lane_color_right, lw, cv2.LINE_AA)

    # lane center tick at bottom
    if state.lane_center_x is not None:
        cx = int(state.lane_center_x)
        cv2.line(frame, (cx, h - 20), (cx, h - 60), cfg.lane_color_current, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, h - 20), 5, cfg.lane_color_current, -1, cv2.LINE_AA)

    return frame


def draw_ar_path(frame: np.ndarray, path_pts: List[Tuple[int, int]], cfg: Config) -> np.ndarray:
    """Draw a tapered, semi-transparent AR path ribbon."""
    if len(path_pts) < 2:
        return frame
    overlay = frame.copy()
    steps = len(path_pts)
    for i in range(steps - 1):
        alpha = 0.55 - 0.35 * (i / steps)      # fade toward horizon
        width = max(2, int(18 * (1 - i / steps)))
        color = cfg.path_color
        cv2.line(overlay, path_pts[i], path_pts[i + 1], color, width, cv2.LINE_AA)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    # arrow at lookahead tip
    if len(path_pts) >= 4:
        tip = path_pts[-1]
        tail = path_pts[-3]
        angle = math.atan2(tip[1] - tail[1], tip[0] - tail[0])
        arrow_len = 14
        for da in [-0.4, 0.4]:
            ax = int(tip[0] - arrow_len * math.cos(angle + da))
            ay = int(tip[1] - arrow_len * math.sin(angle + da))
            cv2.line(frame, tip, (ax, ay), cfg.path_color, 2, cv2.LINE_AA)
    return frame


def draw_steering_hud(frame: np.ndarray, state: LaneState, cfg: Config) -> np.ndarray:
    """Draw a compact bottom-center HUD with offset bar and steering command."""
    h, w = frame.shape[:2]
    if state.confidence == 0:
        return frame

    # --- positions ---
    bar_w, bar_h = 200, 10
    hud_x = w // 2 - bar_w // 2
    hud_y = h - 90
    pad = 8

    overlay = frame.copy()

    # background pill
    cv2.rectangle(
        overlay,
        (hud_x - pad, hud_y - pad),
        (hud_x + bar_w + pad, hud_y + bar_h + 50),
        (20, 20, 20),
        -1,
        cv2.LINE_AA,
    )
    frame = cv2.addWeighted(overlay, cfg.hud_bg_alpha, frame, 1 - cfg.hud_bg_alpha, 0)

    # offset bar background
    cv2.rectangle(frame, (hud_x, hud_y), (hud_x + bar_w, hud_y + bar_h), (60, 60, 60), -1)

    # offset bar fill
    mid = hud_x + bar_w // 2
    offset_clamped = max(-cfg.max_offset_display, min(cfg.max_offset_display, state.offset_px))
    fill_w = int(abs(offset_clamped) / cfg.max_offset_display * (bar_w // 2))
    bar_color = (0, 200, 80) if state.steer_cmd == "ON CENTER" else (30, 140, 255)
    if state.offset_px > 0:
        cv2.rectangle(frame, (mid - fill_w, hud_y), (mid, hud_y + bar_h), bar_color, -1)
    else:
        cv2.rectangle(frame, (mid, hud_y), (mid + fill_w, hud_y + bar_h), bar_color, -1)

    # center tick
    cv2.line(frame, (mid, hud_y - 2), (mid, hud_y + bar_h + 2), (200, 200, 200), 1)

    # steering command label
    cmd = state.steer_cmd
    text_color = (30, 30, 220) if "LEFT" in cmd else (220, 100, 30) if "RIGHT" in cmd else (80, 220, 80)
    _draw_text_center(frame, cmd, (w // 2, hud_y + bar_h + 22), cfg, text_color, scale=0.62, thickness=2)

    # offset value
    side = "R" if state.offset_px > 0 else "L"
    offset_label = f"{abs(int(state.offset_px))}px {side}"
    _draw_text_center(frame, offset_label, (w // 2, hud_y + bar_h + 40), cfg, (180, 180, 180), scale=0.42)

    return frame


def draw_confidence_badge(frame: np.ndarray, state: LaneState, cfg: Config) -> np.ndarray:
    """Small top-right confidence indicator."""
    h, w = frame.shape[:2]
    conf_pct = int(state.confidence * 100)
    color = (0, 200, 80) if conf_pct > 60 else (0, 180, 220) if conf_pct > 30 else (30, 30, 200)
    label = f"Lane {conf_pct}%"
    x, y = w - 120, 28
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 8, y - 18), (x + 110, y + 8), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    cv2.putText(frame, label, (x, y), cfg.font, 0.50, color, 1, cv2.LINE_AA)
    return frame


def draw_feature_legend(frame: np.ndarray, flags: FeatureFlags, cfg: Config) -> np.ndarray:
    """Top-left mini legend showing active features — press key to toggle."""
    items = [
        ("L  Lane detect",    flags.lane_detection),
        ("S  Steer guide",    flags.steering_guidance),
        ("P  AR path",        flags.ar_path),
        # FUTURE HOOK — add new features here
    ]
    x, y0 = 12, 22
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 6, y0 - 16), (x + 148, y0 + len(items) * 18), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    for i, (label, active) in enumerate(items):
        color = (80, 220, 80) if active else (100, 100, 100)
        cv2.putText(frame, label, (x, y0 + i * 18), cfg.font, 0.40, color, 1, cv2.LINE_AA)
    return frame


def _draw_text_center(frame, text, center, cfg, color, scale=None, thickness=None):
    s = scale or cfg.font_scale
    t = thickness or cfg.font_thickness
    (tw, th), _ = cv2.getTextSize(text, cfg.font, s, t)
    ox = center[0] - tw // 2
    oy = center[1] + th // 2
    cv2.putText(frame, text, (ox, oy), cfg.font, s, color, t, cv2.LINE_AA)


# ─────────────────────────────────────────────
#  SMOOTHING (temporal)
# ─────────────────────────────────────────────

class LineSmoother:
    """Exponential moving average over detected line endpoints."""
    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha
        self._left: Optional[np.ndarray] = None
        self._right: Optional[np.ndarray] = None

    def update(
        self,
        left: Optional[np.ndarray],
        right: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        self._left  = self._ema(self._left,  left)
        self._right = self._ema(self._right, right)
        return self._left, self._right

    def _ema(self, prev, cur):
        if cur is None:
            return prev          # keep last known
        if prev is None:
            return cur.astype(float)
        return prev * (1 - self.alpha) + cur.astype(float) * self.alpha


# ─────────────────────────────────────────────
#  KEYBOARD CONTROL
# ─────────────────────────────────────────────

KEYMAP = {
    ord('l'): 'lane_detection',
    ord('s'): 'steering_guidance',
    ord('p'): 'ar_path',
    # FUTURE HOOK — add new toggles here
}

def handle_keypress(key: int, flags: FeatureFlags) -> bool:
    """Returns False if user wants to quit."""
    if key == ord('q') or key == 27:   # q or ESC
        return False
    attr = KEYMAP.get(key)
    if attr and hasattr(flags, attr):
        setattr(flags, attr, not getattr(flags, attr))
    return True


# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────

def run(source=0, cfg: Config = None, flags: FeatureFlags = None, display_width: Optional[int] = None):
    """
    Main entry point.

    Args:
        source: Camera index (int) or video file path (str).
        cfg:    Config object — defaults to Config() if None.
        flags:  FeatureFlags object — defaults to FeatureFlags() if None.
    """
    cfg   = cfg   or Config()
    flags = flags or FeatureFlags()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    smoother = LineSmoother(alpha=0.25)

    print("Lane Assist running.")
    print("  Keys:  l=lane  s=steer  p=path  q/ESC=quit")

    # allow window resizing and optionally scale the displayed frame
    cv2.namedWindow("Lane Assist  [q=quit]", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            # end of file → loop, or camera lost
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        h, w = frame.shape[:2]
        annotated = frame.copy()
        state = LaneState()
        state.frame_center_x = w / 2.0
        state.lane_center_x  = w / 2.0

        # ── FEATURE: Lane Detection ──────────────────────────────
        if flags.lane_detection:
            edges              = preprocess_frame(frame, cfg)
            left_raw, right_raw = detect_lane_lines(edges, cfg)
            left_avg           = average_line(left_raw,  h, cfg)
            right_avg          = average_line(right_raw, h, cfg)
            left_smooth, right_smooth = smoother.update(left_avg, right_avg)

            # cast to int for drawing
            left_i  = left_smooth.astype(int)  if left_smooth  is not None else None
            right_i = right_smooth.astype(int) if right_smooth is not None else None

            state = analyse_lane(frame, left_i, right_i, cfg)
            annotated = draw_lane_overlay(annotated, state, cfg)
            annotated = draw_confidence_badge(annotated, state, cfg)

        # ── FEATURE: AR Path ─────────────────────────────────────
        if flags.ar_path and state.confidence > 0:
            path_pts  = compute_ar_path(state, frame.shape, cfg)
            annotated = draw_ar_path(annotated, path_pts, cfg)

        # ── FEATURE: Steering Guidance ───────────────────────────
        if flags.steering_guidance and state.confidence > 0:
            annotated = draw_steering_hud(annotated, state, cfg)

        # FUTURE HOOK — add feature blocks here, e.g.:
        # if flags.obstacle_detection:
        #     obstacles = detect_obstacles(frame, cfg)
        #     annotated = draw_obstacle_boxes(annotated, obstacles, cfg)

        # ── Always-on UI ─────────────────────────────────────────
        annotated = draw_feature_legend(annotated, flags, cfg)

        # resize for display if a target width was provided
        if display_width is not None and display_width > 0 and w > display_width:
            scale = float(display_width) / float(w)
            disp_w = int(w * scale)
            disp_h = int(h * scale)
            display_frame = cv2.resize(annotated, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        else:
            display_frame = annotated

        cv2.imshow("Lane Assist  [q=quit]", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if not handle_keypress(key, flags):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lane Assist AR System")
    parser.add_argument(
        "--source", default="0",
        help="Camera index (0,1,...) or path to a video file"
    )
    parser.add_argument("--roi-top",    type=float, default=0.55,  help="ROI top ratio (0–1)")
    parser.add_argument("--canny-low",  type=int,   default=50,    help="Canny low threshold")
    parser.add_argument("--canny-high", type=int,   default=150,   help="Canny high threshold")
    parser.add_argument("--no-path",    action="store_true",       help="Disable AR path on startup")
    parser.add_argument("--no-steer",   action="store_true",       help="Disable steering HUD on startup")
    parser.add_argument("--display-width", type=int, default=960, help="Target display width in pixels (keeps aspect ratio). Set 0 to disable scaling.")
    args = parser.parse_args()

    # parse source
    source = int(args.source) if args.source.isdigit() else args.source

    cfg = Config(
        roi_top_ratio=args.roi_top,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
    )
    flags = FeatureFlags(
        ar_path=not args.no_path,
        steering_guidance=not args.no_steer,
    )

    run(source=source, cfg=cfg, flags=flags, display_width=args.display_width)
