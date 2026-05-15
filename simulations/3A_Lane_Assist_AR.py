"""
Lane Assist AR System — v2.0  (Industry-grade rewrite)
=======================================================
Key improvements over v1:
  • Trapezoid ROI locked to lower 40% of frame  → eliminates sky/trees/guardrails
  • HSV colour isolation for WHITE + YELLOW lane markings before edge detection
  • Strict left/right line split by BOTH x-position AND slope — no crossing lines
  • 2nd-degree polynomial fit (x = f(y)) instead of single Hough line → handles curves
  • Temporal EMA smoothing on polynomial coefficients → stable on video
  • AR path ribbon anchored to vehicle hood, follows lane curvature
  • Steering offset derived from poly evaluation at bottom of frame
  • Resizes to 960×540 (configurable) — no full-screen takeover
  • Fully modular: each stage is an isolated function, flags toggle at runtime

Keyboard controls (live, no restart):
  l  — toggle lane detection
  s  — toggle steering HUD
  p  — toggle AR path
  d  — toggle debug edge inset
  q / ESC — quit

Usage:
  python lane_assist.py                         # webcam 0
  python lane_assist.py --source dashcam.mp4
  python lane_assist.py --source 1              # webcam index 1
  python lane_assist.py --source video.mp4 --display-width 1280

Tuning tips:
  --roi-top 0.65        raise ROI if trees/guardrails still detected
  --canny-low 30        lower if white lines not appearing in debug view
  --smooth-alpha 0.15   reduce jitter (lower = more inertia)
  --debug               show edge mask inset for live tuning
"""

import cv2
import numpy as np
import math
import argparse
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION  — every tunable in one place
# ═══════════════════════════════════════════════════════════════

@dataclass
class Config:
    # ── Display ──────────────────────────────────────────────
    display_width:    int   = 960
    display_height:   int   = 540

    # ── ROI trapezoid in NORMALISED coords (0–1) ─────────────
    # Vertices: bottom-left, top-left, top-right, bottom-right
    # Tight to road surface — does NOT include sky, trees, horizon
    roi_bottom_y:     float = 1.00
    roi_top_y:        float = 0.60   # raise this if non-road detected
    roi_top_left_x:   float = 0.38
    roi_top_right_x:  float = 0.62
    roi_bot_left_x:   float = 0.08
    roi_bot_right_x:  float = 0.92

    # ── HSV colour masks ─────────────────────────────────────
    white_hsv_lo:  Tuple = (0,   0,  160)
    white_hsv_hi:  Tuple = (180, 60, 255)
    yellow_hsv_lo: Tuple = (15,  60, 100)
    yellow_hsv_hi: Tuple = (40, 255, 255)

    # ── Canny edges ───────────────────────────────────────────
    canny_low:   int = 40
    canny_high:  int = 120
    blur_ksize:  int = 5

    # ── Hough probabilistic ───────────────────────────────────
    hough_threshold:    int = 30
    hough_min_line_len: int = 30
    hough_max_line_gap: int = 80

    # ── Slope gate (keeps only plausible lane slopes) ─────────
    slope_abs_min: float = 0.25   # reject near-horizontal (noise, road texture)
    slope_abs_max: float = 3.50   # reject near-vertical

    # ── Polynomial fit ────────────────────────────────────────
    poly_degree: int = 2           # quadratic handles most road curves

    # ── Temporal smoothing ────────────────────────────────────
    smooth_alpha:  float = 0.20    # EMA weight for new frame (0.1=very smooth)
    smooth_hold:   int   = 8       # frames to retain last-known before dropping

    # ── Steering ──────────────────────────────────────────────
    steer_deadzone_px: int   = 25
    steer_max_px:      int   = 300  # clamp for display bar

    # ── AR path ribbon ────────────────────────────────────────
    path_steps:      int   = 22
    path_lookahead:  float = 0.42   # fraction of frame height projected ahead
    path_ribbon_w:   int   = 14     # max half-width at bottom (tapers up)

    # ── Visual colours ────────────────────────────────────────
    color_lane_L:  Tuple = (0, 230, 110)    # green
    color_lane_R:  Tuple = (0, 230, 110)    # green
    color_fill:    Tuple = (0, 180,  80)    # lane polygon fill
    color_path:    Tuple = (255, 165,  0)   # amber AR ribbon
    hud_alpha:     float = 0.58
    font = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class FeatureFlags:
    """One bool per feature.  Add new features by adding a field here."""
    lane_detection:    bool = True
    steering_guidance: bool = True
    ar_path:           bool = True
    debug_edges:       bool = False
    # FUTURE: obstacle_detection: bool = False
    # FUTURE: sign_detection:     bool = False
    # FUTURE: adaptive_cruise:    bool = False


KEYMAP = {
    ord('l'): 'lane_detection',
    ord('s'): 'steering_guidance',
    ord('p'): 'ar_path',
    ord('d'): 'debug_edges',
    # FUTURE: ord('o'): 'obstacle_detection',
}


# ═══════════════════════════════════════════════════════════════
#  DATA STRUCTURE
# ═══════════════════════════════════════════════════════════════

@dataclass
class LaneState:
    left_poly:      Optional[np.ndarray] = None   # coefficients [a,b,c] for x=f(y)
    right_poly:     Optional[np.ndarray] = None
    lane_center_x:  float = 0.0
    frame_center_x: float = 0.0
    offset_px:      float = 0.0   # positive = vehicle right of lane centre
    steer_cmd:      str   = "SEARCHING"
    confidence:     float = 0.0   # 0=no lanes, 0.55=one lane, 1.0=both lanes
    has_left:       bool  = False
    has_right:      bool  = False


# ═══════════════════════════════════════════════════════════════
#  ROI
# ═══════════════════════════════════════════════════════════════

def build_roi_vertices(h: int, w: int, cfg: Config) -> np.ndarray:
    """Return a trapezoid tightly bounding the road surface."""
    return np.array([
        [int(w * cfg.roi_bot_left_x),  int(h * cfg.roi_bottom_y)],
        [int(w * cfg.roi_top_left_x),  int(h * cfg.roi_top_y)],
        [int(w * cfg.roi_top_right_x), int(h * cfg.roi_top_y)],
        [int(w * cfg.roi_bot_right_x), int(h * cfg.roi_bottom_y)],
    ], dtype=np.int32)


def apply_roi(img: np.ndarray, verts: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(img)
    fill = 255 if img.ndim == 2 else (255, 255, 255)
    cv2.fillPoly(mask, [verts], fill)
    return cv2.bitwise_and(img, mask)


# ═══════════════════════════════════════════════════════════════
#  COLOUR ISOLATION  (HSV)
# ═══════════════════════════════════════════════════════════════

def isolate_lane_colours(frame: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Extract white AND yellow pixels.
    Trees, guardrails, asphalt are NOT white/yellow → filtered out here.
    Morphological ops remove speckle and reconnect dashed lines.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mw  = cv2.inRange(hsv, np.array(cfg.white_hsv_lo),  np.array(cfg.white_hsv_hi))
    my  = cv2.inRange(hsv, np.array(cfg.yellow_hsv_lo), np.array(cfg.yellow_hsv_hi))
    combined = cv2.bitwise_or(mw, my)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k, iterations=1)
    return combined


# ═══════════════════════════════════════════════════════════════
#  EDGE DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_edges(mask: np.ndarray, cfg: Config) -> np.ndarray:
    blur  = cv2.GaussianBlur(mask, (cfg.blur_ksize, cfg.blur_ksize), 0)
    return cv2.Canny(blur, cfg.canny_low, cfg.canny_high)


# ═══════════════════════════════════════════════════════════════
#  HOUGH + LINE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════

def run_hough(edges: np.ndarray, cfg: Config) -> Optional[np.ndarray]:
    return cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=cfg.hough_threshold,
        minLineLength=cfg.hough_min_line_len,
        maxLineGap=cfg.hough_max_line_gap,
    )


def classify_lines(
    raw: Optional[np.ndarray],
    w:   int,
    cfg: Config,
) -> Tuple[List[Tuple[int,int]], List[Tuple[int,int]]]:
    """
    Separate Hough segments into left/right point clouds.

    Dual-gate rule (both conditions required):
      Left  line: negative slope  AND  midpoint x < 55% of width
      Right line: positive slope  AND  midpoint x > 45% of width

    This prevents crossing artefacts — a common failure mode of slope-only gating.
    """
    left_pts:  List[Tuple[int,int]] = []
    right_pts: List[Tuple[int,int]] = []
    if raw is None:
        return left_pts, right_pts

    mid_w = w * 0.50
    for seg in raw:
        x1, y1, x2, y2 = seg[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if not (cfg.slope_abs_min <= abs(slope) <= cfg.slope_abs_max):
            continue
        mx = (x1 + x2) / 2.0
        if slope < 0 and mx < mid_w * 1.10:
            left_pts.extend([(x1, y1), (x2, y2)])
        elif slope > 0 and mx > mid_w * 0.90:
            right_pts.extend([(x1, y1), (x2, y2)])

    return left_pts, right_pts


# ═══════════════════════════════════════════════════════════════
#  POLYNOMIAL FITTING  (x = f(y))
# ═══════════════════════════════════════════════════════════════

def fit_poly(pts: List[Tuple[int,int]], degree: int) -> Optional[np.ndarray]:
    """
    Fit  x = poly(y)  through a point cloud.
    Using the y→x convention prevents degenerate fits on steep lines.
    Requires at least degree+2 distinct points.
    """
    if len(pts) < degree + 2:
        return None
    ys = np.array([p[1] for p in pts], dtype=np.float32)
    xs = np.array([p[0] for p in pts], dtype=np.float32)
    # Check for enough y-spread (avoid degenerate horizontal patches)
    if np.ptp(ys) < 10:
        return None
    try:
        return np.polyfit(ys, xs, degree)
    except (np.linalg.LinAlgError, ValueError):
        return None


def poly_x(c: np.ndarray, y: float) -> float:
    """Evaluate x = f(y)."""
    return float(np.polyval(c, y))


# ═══════════════════════════════════════════════════════════════
#  TEMPORAL SMOOTHING  (EMA on poly coefficients)
# ═══════════════════════════════════════════════════════════════

class PolySmoother:
    """
    Exponential moving average over polynomial coefficients.
    Retains last-known values for up to `hold_frames` before dropping.
    """
    def __init__(self, alpha: float, hold_frames: int):
        self.alpha       = alpha
        self.hold_frames = hold_frames
        self._lc: Optional[np.ndarray] = None
        self._rc: Optional[np.ndarray] = None
        self._la = 0
        self._ra = 0

    def update(
        self,
        lc: Optional[np.ndarray],
        rc: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        self._lc, self._la = self._step(self._lc, lc, self._la)
        self._rc, self._ra = self._step(self._rc, rc, self._ra)
        return self._lc, self._rc

    def _step(self, prev, cur, age):
        if cur is not None:
            smoothed = (prev * (1 - self.alpha) + cur.astype(float) * self.alpha
                        if prev is not None else cur.astype(float))
            return smoothed, 0
        age += 1
        return (None, age) if age > self.hold_frames else (prev, age)


# ═══════════════════════════════════════════════════════════════
#  LANE STATE ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyse_lane(
    lp: Optional[np.ndarray],
    rp: Optional[np.ndarray],
    w:  int,
    h:  int,
    cfg: Config,
) -> LaneState:
    state = LaneState(frame_center_x=w / 2.0)
    y_ref = int(h * cfg.roi_bottom_y) - 5   # evaluate at near-bottom of frame

    hl = lp is not None
    hr = rp is not None
    state.has_left  = hl
    state.has_right = hr
    state.left_poly  = lp
    state.right_poly = rp

    if hl and hr:
        state.lane_center_x = (poly_x(lp, y_ref) + poly_x(rp, y_ref)) / 2.0
        state.confidence     = 1.0
    elif hl:
        state.lane_center_x = poly_x(lp, y_ref) + w * 0.22
        state.confidence     = 0.55
    elif hr:
        state.lane_center_x = poly_x(rp, y_ref) - w * 0.22
        state.confidence     = 0.55
    else:
        state.lane_center_x = w / 2.0
        state.confidence     = 0.0

    # positive offset = vehicle is RIGHT of lane centre (needs LEFT steer)
    state.offset_px = state.frame_center_x - state.lane_center_x
    dz = cfg.steer_deadzone_px
    if state.confidence == 0:
        state.steer_cmd = "SEARCHING"
    elif state.offset_px > dz:
        state.steer_cmd = "◀  STEER LEFT"
    elif state.offset_px < -dz:
        state.steer_cmd = "STEER RIGHT  ▶"
    else:
        state.steer_cmd = "●  ON CENTER"

    return state


# ═══════════════════════════════════════════════════════════════
#  AR PATH
# ═══════════════════════════════════════════════════════════════

def compute_ar_path(
    state: LaneState,
    h:     int,
    w:     int,
    cfg:   Config,
) -> Tuple[List, List, List]:
    """
    Build (centre_pts, left_edge_pts, right_edge_pts) for the ribbon.
    Projects from vehicle hood (bottom) to lookahead point (up the frame).
    When both lane polys are available the ribbon naturally follows curvature.
    """
    y_bot = int(h * cfg.roi_bottom_y)
    y_tip = int(h * (cfg.roi_top_y + (1.0 - cfg.roi_top_y) * (1.0 - cfg.path_lookahead)))
    ys    = np.linspace(y_bot, y_tip, cfg.path_steps)

    centre, left_e, right_e = [], [], []
    lp = state.left_poly
    rp = state.right_poly

    for y in ys:
        t = (y_bot - y) / max(y_bot - y_tip, 1)   # 0=bottom, 1=tip
        ribbon = max(2, int(cfg.path_ribbon_w * (1 - 0.65 * t)))

        if state.has_left and state.has_right:
            cx = int((poly_x(lp, y) + poly_x(rp, y)) / 2.0)
        elif state.has_left:
            cx = int(poly_x(lp, y) + w * 0.22)
        elif state.has_right:
            cx = int(poly_x(rp, y) - w * 0.22)
        else:
            # gentle correction arc toward estimated lane centre
            t_ease = t ** 1.6
            cx = int(w / 2.0 + (state.lane_center_x - w / 2.0) * t_ease)

        iy = int(y)
        centre.append((cx,            iy))
        left_e.append( (cx - ribbon,  iy))
        right_e.append((cx + ribbon,  iy))

    return centre, left_e, right_e


# ═══════════════════════════════════════════════════════════════
#  DRAWING
# ═══════════════════════════════════════════════════════════════

def draw_lane_overlay(
    frame:    np.ndarray,
    state:    LaneState,
    cfg:      Config,
) -> np.ndarray:
    """Filled lane polygon + fitted polynomial curves."""
    h, w = frame.shape[:2]
    y_top = int(h * cfg.roi_top_y)
    y_bot = int(h * cfg.roi_bottom_y)
    ys    = np.linspace(y_top, y_bot, 50).astype(int)

    lp = state.left_poly
    rp = state.right_poly

    # ── filled polygon ─────────────────────────────────────
    if lp is not None and rp is not None:
        l_pts = [(int(poly_x(lp, y)), y) for y in ys]
        r_pts = [(int(poly_x(rp, y)), y) for y in reversed(ys)]
        poly_arr = np.array(l_pts + r_pts, dtype=np.int32)
        ov = frame.copy()
        cv2.fillPoly(ov, [poly_arr], cfg.color_fill)
        frame = cv2.addWeighted(ov, 0.20, frame, 0.80, 0)

    # ── lane boundary curves ────────────────────────────────
    def draw_curve(poly, color):
        pts = np.array([(int(poly_x(poly, y)), y) for y in ys], dtype=np.int32)
        for i in range(len(pts) - 1):
            cv2.line(frame, tuple(pts[i]), tuple(pts[i+1]), color, 4, cv2.LINE_AA)

    if lp is not None: draw_curve(lp, cfg.color_lane_L)
    if rp is not None: draw_curve(rp, cfg.color_lane_R)

    return frame


def draw_ar_ribbon(
    frame:   np.ndarray,
    centre:  List,
    left_e:  List,
    right_e: List,
    cfg:     Config,
) -> np.ndarray:
    """Tapered amber ribbon path toward lane centre."""
    steps = len(centre)
    if steps < 2:
        return frame

    ov = frame.copy()
    for i in range(steps - 1):
        quad = np.array([left_e[i], left_e[i+1], right_e[i+1], right_e[i]], dtype=np.int32)
        cv2.fillPoly(ov, [quad], cfg.color_path)
    frame = cv2.addWeighted(ov, 0.55, frame, 0.45, 0)

    # spine
    for i in range(steps - 1):
        t = i / steps
        lw = max(1, int(3 * (1 - t)))
        cv2.line(frame, centre[i], centre[i+1], (255, 255, 255), lw, cv2.LINE_AA)

    # arrowhead at lookahead tip
    tip  = centre[-1]
    tail = centre[max(0, len(centre) - 5)]
    ang  = math.atan2(tip[1] - tail[1], tip[0] - tail[0])
    for da in (-0.45, 0.45):
        ax = int(tip[0] - 16 * math.cos(ang + da))
        ay = int(tip[1] - 16 * math.sin(ang + da))
        cv2.line(frame, tip, (ax, ay), cfg.color_path, 2, cv2.LINE_AA)

    return frame


def draw_steering_hud(
    frame: np.ndarray,
    state: LaneState,
    cfg:   Config,
) -> np.ndarray:
    """Compact bottom-centre bar + steering label."""
    if state.confidence == 0:
        return frame
    h, w = frame.shape[:2]
    bw, bh = 220, 10
    hx   = w // 2 - bw // 2
    hy   = h - 75
    pad  = 10
    mid  = hx + bw // 2

    ov = frame.copy()
    cv2.rectangle(ov, (hx-pad, hy-pad-4), (hx+bw+pad, hy+bh+54), (18, 18, 18), -1)
    frame = cv2.addWeighted(ov, cfg.hud_alpha, frame, 1 - cfg.hud_alpha, 0)

    # track
    cv2.rectangle(frame, (hx, hy), (hx+bw, hy+bh), (55, 55, 55), -1)
    cv2.circle(frame, (hx,      hy+bh//2), bh//2, (55,55,55), -1)
    cv2.circle(frame, (hx+bw,   hy+bh//2), bh//2, (55,55,55), -1)

    # fill
    off = max(-cfg.steer_max_px, min(cfg.steer_max_px, state.offset_px))
    fw  = int(abs(off) / cfg.steer_max_px * (bw // 2))
    col = (60,220,60) if "CENTER" in state.steer_cmd or "●" in state.steer_cmd else (30,140,255)
    if state.offset_px > 0:
        cv2.rectangle(frame, (mid-fw, hy), (mid,    hy+bh), col, -1)
    else:
        cv2.rectangle(frame, (mid,    hy), (mid+fw, hy+bh), col, -1)
    cv2.line(frame, (mid, hy-3), (mid, hy+bh+3), (200,200,200), 1)

    # command label
    if "LEFT"   in state.steer_cmd: cc = (100,100,255)
    elif "RIGHT" in state.steer_cmd: cc = (60,180,255)
    elif "●"     in state.steer_cmd: cc = (60,220,60)
    else:                             cc = (150,150,150)
    _tc(frame, state.steer_cmd, (w//2, hy+bh+22), cfg, cc, scale=0.62, thick=2)

    side  = "R" if state.offset_px > 0 else "L"
    info  = f"{abs(int(state.offset_px))}px {side}   conf {int(state.confidence*100)}%"
    _tc(frame, info, (w//2, hy+bh+42), cfg, (155,155,155), scale=0.38)

    return frame


def draw_confidence_badge(frame: np.ndarray, state: LaneState, cfg: Config) -> np.ndarray:
    h, w = frame.shape[:2]
    conf = int(state.confidence * 100)
    col  = (60,220,60) if conf > 70 else (60,180,255) if conf > 40 else (60,60,200)
    label = f"LANE  {conf}%"
    x, y  = w - 130, 30
    ov = frame.copy()
    cv2.rectangle(ov, (x-10, y-20), (x+120, y+8), (18,18,18), -1)
    frame = cv2.addWeighted(ov, 0.55, frame, 0.45, 0)
    cv2.putText(frame, label, (x, y), cfg.font, 0.50, col, 1, cv2.LINE_AA)
    return frame


def draw_legend(frame: np.ndarray, flags: FeatureFlags, cfg: Config) -> np.ndarray:
    entries = [
        ("[L] Lane",     flags.lane_detection),
        ("[S] Steer",    flags.steering_guidance),
        ("[P] AR path",  flags.ar_path),
        ("[D] Debug",    flags.debug_edges),
    ]
    x, y0 = 12, 24
    rh    = 18
    ov = frame.copy()
    cv2.rectangle(ov, (x-6, y0-18), (x+120, y0+len(entries)*rh+2), (18,18,18), -1)
    frame = cv2.addWeighted(ov, 0.52, frame, 0.48, 0)
    for i, (lbl, on) in enumerate(entries):
        col = (60,220,60) if on else (90,90,90)
        cv2.putText(frame, lbl, (x, y0+i*rh), cfg.font, 0.40, col, 1, cv2.LINE_AA)
    return frame


def draw_debug_inset(
    frame:     np.ndarray,
    edges:     np.ndarray,
    roi_verts: np.ndarray,
    cfg:       Config,
) -> np.ndarray:
    h, w = frame.shape[:2]
    sw, sh = w // 4, h // 4
    small  = cv2.resize(edges, (sw, sh))
    bgr    = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    scale  = np.array([[sw/w, sh/h]])
    rs     = (roi_verts * scale[0]).astype(np.int32)
    cv2.polylines(bgr, [rs], True, (0,200,255), 1)
    x0, y0 = w - sw - 10, 10
    frame[y0:y0+sh, x0:x0+sw] = bgr
    cv2.rectangle(frame, (x0-1,y0-1), (x0+sw,y0+sh), (0,200,255), 1)
    return frame


def _tc(frame, text, cen, cfg, color, scale=0.55, thick=1):
    """Draw text centred at cen."""
    (tw, th), _ = cv2.getTextSize(text, cfg.font, scale, thick)
    cv2.putText(frame, text, (cen[0]-tw//2, cen[1]+th//2),
                cfg.font, scale, color, thick, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════
#  SINGLE-FRAME PIPELINE
# ═══════════════════════════════════════════════════════════════

def process_frame(
    frame:    np.ndarray,
    smoother: PolySmoother,
    flags:    FeatureFlags,
    cfg:      Config,
) -> Tuple[np.ndarray, LaneState]:
    h, w      = frame.shape[:2]
    annotated = frame.copy()
    roi_verts = build_roi_vertices(h, w, cfg)
    state     = LaneState(frame_center_x=w/2.0, lane_center_x=w/2.0)
    edges_img = None

    # ── Lane Detection ──────────────────────────────────────────
    if flags.lane_detection:
        colour   = isolate_lane_colours(frame, cfg)
        roi_mask = apply_roi(colour, roi_verts)
        edges    = detect_edges(roi_mask, cfg)
        edges_img = edges

        raw              = run_hough(edges, cfg)
        left_pts, rpt    = classify_lines(raw, w, cfg)
        lp               = fit_poly(left_pts, cfg.poly_degree)
        rp               = fit_poly(rpt,      cfg.poly_degree)
        lp_s, rp_s       = smoother.update(lp, rp)

        state            = analyse_lane(lp_s, rp_s, w, h, cfg)
        annotated        = draw_lane_overlay(annotated, state, cfg)
        annotated        = draw_confidence_badge(annotated, state, cfg)

    # ── AR Path ─────────────────────────────────────────────────
    if flags.ar_path and flags.lane_detection and state.confidence > 0:
        cpts, le, re = compute_ar_path(state, h, w, cfg)
        annotated    = draw_ar_ribbon(annotated, cpts, le, re, cfg)

    # ── Steering HUD ────────────────────────────────────────────
    if flags.steering_guidance:
        annotated = draw_steering_hud(annotated, state, cfg)

    # FUTURE HOOK ─ add feature blocks here, e.g.:
    # if flags.obstacle_detection:
    #     obs = detect_obstacles(frame, cfg)
    #     annotated = draw_obstacle_boxes(annotated, obs, cfg)

    # ── Always-on chrome ────────────────────────────────────────
    annotated = draw_legend(annotated, flags, cfg)

    if flags.debug_edges and edges_img is not None:
        annotated = draw_debug_inset(annotated, edges_img, roi_verts, cfg)

    return annotated, state


# ═══════════════════════════════════════════════════════════════
#  KEY HANDLING
# ═══════════════════════════════════════════════════════════════

def handle_key(key: int, flags: FeatureFlags) -> bool:
    if key in (ord('q'), 27):
        return False
    attr = KEYMAP.get(key)
    if attr and hasattr(flags, attr):
        setattr(flags, attr, not getattr(flags, attr))
        print(f"  Toggle [{attr}] → {getattr(flags, attr)}")
    return True


# ═══════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════

def run(source, cfg: Config, flags: FeatureFlags):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}", file=sys.stderr)
        sys.exit(1)

    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay_ms = max(1, int(1000 / fps))

    smoother = PolySmoother(cfg.smooth_alpha, cfg.smooth_hold)

    win = "Lane Assist v2  |  q=quit  l/s/p/d=toggle"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, cfg.display_width, cfg.display_height)

    print("=" * 58)
    print("  Lane Assist AR System  v2.0")
    print(f"  Source  : {source}")
    print(f"  Display : {cfg.display_width} × {cfg.display_height}")
    print("  Keys    :  l=lane  s=steer  p=path  d=debug  q=quit")
    print("=" * 58)

    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str):
                # loop video file
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                smoother = PolySmoother(cfg.smooth_alpha, cfg.smooth_hold)
                continue
            break

        n += 1
        frame_rs          = cv2.resize(frame, (cfg.display_width, cfg.display_height))
        annotated, state  = process_frame(frame_rs, smoother, flags, cfg)

        cv2.imshow(win, annotated)
        key = cv2.waitKey(delay_ms) & 0xFF
        if not handle_key(key, flags):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"  Processed {n} frames.  Goodbye.")


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Lane Assist AR System v2.0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source",          default="0",
                   help="Camera index (int) or path to video file")
    p.add_argument("--display-width",   type=int,   default=960)
    p.add_argument("--display-height",  type=int,   default=540)
    # ROI
    p.add_argument("--roi-top",         type=float, default=0.60,
                   help="ROI top y fraction. Raise (e.g. 0.65) to cut trees/horizon")
    p.add_argument("--roi-top-lx",      type=float, default=0.38)
    p.add_argument("--roi-top-rx",      type=float, default=0.62)
    p.add_argument("--roi-bot-lx",      type=float, default=0.08)
    p.add_argument("--roi-bot-rx",      type=float, default=0.92)
    # Edge detection
    p.add_argument("--canny-low",       type=int,   default=40)
    p.add_argument("--canny-high",      type=int,   default=120)
    # Smoothing
    p.add_argument("--smooth-alpha",    type=float, default=0.20,
                   help="EMA weight: 0.1=very smooth/laggy, 0.4=responsive/jittery")
    # Feature startup state
    p.add_argument("--no-path",         action="store_true")
    p.add_argument("--no-steer",        action="store_true")
    p.add_argument("--debug",           action="store_true",
                   help="Start with debug edge inset enabled")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    src  = int(args.source) if args.source.isdigit() else args.source

    cfg = Config(
        display_width   = args.display_width,
        display_height  = args.display_height,
        roi_top_y       = args.roi_top,
        roi_top_left_x  = args.roi_top_lx,
        roi_top_right_x = args.roi_top_rx,
        roi_bot_left_x  = args.roi_bot_lx,
        roi_bot_right_x = args.roi_bot_rx,
        canny_low       = args.canny_low,
        canny_high      = args.canny_high,
        smooth_alpha    = args.smooth_alpha,
    )
    flags = FeatureFlags(
        ar_path           = not args.no_path,
        steering_guidance = not args.no_steer,
        debug_edges       = args.debug,
    )

    run(src, cfg, flags)
