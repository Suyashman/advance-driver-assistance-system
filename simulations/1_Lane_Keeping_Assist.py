"""
2D Car Simulation — Kinematic Bicycle Model
============================================
Controls: W/S = throttle/brake, A/D = steer, Q = quit, R = reset
Requires: pip install pygame numpy

Architecture is designed for ADAS extension:
  - CarState     : all vehicle state (position, heading, speed, steer angle)
  - CarParams    : tunable physical parameters
  - PhysicsEngine: pure kinematics — no pygame dependency
  - ADASModule   : base class for plugging in ADAS features (lane assist, etc.)
  - Renderer     : pygame visualisation
  - Simulation   : top-level loop
"""

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

# ── Optional pygame import ──────────────────────────────────────────────────
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("pygame not found. Running in headless/text mode. Install with: pip install pygame")


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CarState:
    """Full vehicle state at a single timestep."""
    x: float = 0.0          # world X position (m)
    y: float = 0.0          # world Y position (m)
    heading: float = 0.0    # orientation (radians, 0 = east)
    speed: float = 0.0      # m/s (positive = forward)
    steer: float = 0.0      # current front wheel steer angle (rad)
    dist_traveled: float = 0.0  # odometer (m)

    def to_dict(self) -> dict:
        return {
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "heading_deg": round(math.degrees(self.heading), 2),
            "speed_mps": round(self.speed, 3),
            "steer_deg": round(math.degrees(self.steer), 2),
            "dist_m": round(self.dist_traveled, 2),
        }


@dataclass
class CarParams:
    """
    Tunable physical parameters.

    Kinematic bicycle model (see: Rajamani, 'Vehicle Dynamics and Control'):
      - Wheelbase L separates front/rear axle
      - Front wheel steers at angle delta (steer)
      - Heading changes at rate:  dθ/dt = (v / L) * tan(delta)
      - Position integrates:      dx/dt = v*cos(θ),  dy/dt = v*sin(θ)
    """
    wheelbase: float = 4.5         # L (m) — distance between axles
    car_width: float = 2.0         # m (visualisation only)

    max_accel: float = 5.0         # m/s²  — wide open throttle
    max_brake: float = 8.0         # m/s²  — hard braking
    friction_decel: float = 1.2    # m/s²  — rolling resistance (engine off)
    aero_drag: float = 0.08        # dimensionless drag coeff (F_drag ∝ v²)

    max_steer: float = math.radians(30)   # max front wheel angle (rad)
    steer_rate: float = math.radians(90)  # how fast the wheel turns (rad/s)

    max_speed_fwd: float = 30.0    # m/s  ~108 km/h
    max_speed_rev: float = 8.0     # m/s  ~29 km/h reverse


@dataclass
class DriverInput:
    """Normalised driver commands, range [-1, 1]."""
    throttle: float = 0.0   # +1 = full throttle, -1 = full brake
    steer: float = 0.0      # +1 = full right, -1 = full left


# ══════════════════════════════════════════════════════════════════════════════
# Physics engine  (pure Python, no rendering dependency)
# ══════════════════════════════════════════════════════════════════════════════

class PhysicsEngine:
    """
    Kinematic bicycle model integration.

    Suitable for ADAS simulation at low/medium speeds.
    For high-speed or tyre-slip scenarios, replace with a dynamic model.
    """

    def __init__(self, params: CarParams):
        self.p = params

    def step(self, state: CarState, inp: DriverInput, dt: float) -> CarState:
        """Integrate state forward by dt seconds. Returns a NEW state object."""
        p = self.p
        s = CarState(
            x=state.x, y=state.y,
            heading=state.heading,
            speed=state.speed,
            steer=state.steer,
            dist_traveled=state.dist_traveled,
        )

        # ── 1. Steering update ─────────────────────────────────────────────
        steer_target = inp.steer * p.max_steer
        steer_diff = steer_target - s.steer
        max_delta = p.steer_rate * dt
        s.steer += math.copysign(min(abs(steer_diff), max_delta), steer_diff)

        # ── 2. Longitudinal dynamics ───────────────────────────────────────
        if inp.throttle > 0:
            accel = inp.throttle * p.max_accel
        elif inp.throttle < 0:
            accel = inp.throttle * p.max_brake    # negative = braking
        else:
            # Friction / rolling resistance
            accel = -math.copysign(p.friction_decel, s.speed) if abs(s.speed) > 0.01 else 0.0

        # Aerodynamic drag (opposes motion, scales with v²)
        drag = p.aero_drag * s.speed * abs(s.speed)
        accel -= drag

        s.speed += accel * dt
        s.speed = max(-p.max_speed_rev, min(p.max_speed_fwd, s.speed))
        if abs(s.speed) < 0.01 and inp.throttle == 0:
            s.speed = 0.0

        # ── 3. Kinematic bicycle model ─────────────────────────────────────
        # Heading rate:  dθ/dt = (v / L) * tan(δ)
        if abs(s.speed) > 0.001:
            heading_rate = (s.speed / p.wheelbase) * math.tan(s.steer)
            s.heading += heading_rate * dt

            prev_x, prev_y = s.x, s.y
            s.x += s.speed * math.cos(s.heading) * dt
            s.y += s.speed * math.sin(s.heading) * dt
            s.dist_traveled += math.hypot(s.x - prev_x, s.y - prev_y)

        # Keep heading in (-π, π]
        s.heading = (s.heading + math.pi) % (2 * math.pi) - math.pi

        return s


# ══════════════════════════════════════════════════════════════════════════════
# ADAS Module base class — extend this for lane assist, alerts, parking etc.
# ══════════════════════════════════════════════════════════════════════════════

class ADASModule:
    """
    Base class for all ADAS features.
    Override `process()` to read state, optionally override driver input.

    Example subclasses to add later:
      - LaneKeepAssist   : reads lane geometry, adds steer correction
      - LaneChangeAlert  : detects rapid heading change, emits warning
      - AdaptiveCruise   : adjusts throttle to maintain gap to lead car
      - ParkingAssist    : guides car into a target pose
      - CollisionWarning : checks proximity to obstacles
    """

    name: str = "base"

    def process(
        self,
        state: CarState,
        inp: DriverInput,
        dt: float,
    ) -> Tuple[DriverInput, List[str]]:
        """
        Args:
            state: current vehicle state
            inp: raw driver input (may be overridden)
            dt: timestep

        Returns:
            (modified_input, list_of_alerts)
        """
        return inp, []


class LaneKeepAssistDemo(ADASModule):
    """
    Demo ADAS: keep the car near y=0 (a straight virtual lane).
    Uses PD (proportional-derivative) control for quick, non-oscillatory centering.
    A real implementation would use camera lane detection.
    """
    name = "LKA"

    def __init__(
        self,
        lane_y: float = 0.0,
        kp: float = 0.5,          # proportional gain (reduced to prevent aggressive oversteer)
        kd: float = 1.8,          # derivative gain (high damping to prevent oscillations)
        lane_center_threshold: float = 0.5,  # within this range, steering is normal
        lane_width: float = 4.0,   # total lane width (±2m from center)
        lane_departure_threshold: float = 2.5,  # disable if exceeded
    ):
        self.lane_y = lane_y
        self.kp = kp  # proportional control gain
        self.kd = kd  # derivative (damping) gain
        self.lane_center_threshold = lane_center_threshold
        self.lane_width = lane_width
        self.lane_departure_threshold = lane_departure_threshold
        self._prev_error = 0.0
        self._is_disabled = False
        self._disabled_since = 0.0

    def process(self, state: CarState, inp: DriverInput, dt: float):
        alerts = []
        lateral_error = state.y - self.lane_y
        
        # Lane departure detection — disable if exceeded
        if abs(lateral_error) > self.lane_departure_threshold:
            self._is_disabled = True
            self._disabled_since = 0.0
            alerts.append(f"LKA: LANE DETECTION DISABLED — vehicle left lane ({lateral_error:+.1f} m)")
            return inp, alerts
        
        # Auto-recovery: if within lane center threshold for 1 second, re-enable
        if self._is_disabled:
            self._disabled_since += dt
            if abs(lateral_error) < self.lane_center_threshold and self._disabled_since > 1.0:
                self._is_disabled = False
                alerts.append("LKA: Lane detection re-enabled")
            else:
                return inp, alerts  # Keep disabled, pass through input
        
        # Normal operation: PD control to keep car centered
        if abs(lateral_error) > self.lane_center_threshold:
            # Calculate PD correction
            # P: strong proportional term based on position error
            p_term = -self.kp * lateral_error
            
            # D: derivative term (damping) based on rate of error change
            error_rate = (lateral_error - self._prev_error) / max(dt, 0.001)
            d_term = -self.kd * error_rate
            
            # Combined correction
            correction = p_term + d_term
            correction = max(-1.0, min(1.0, correction))
            
            # Smooth steering correction - blend with driver input
            # The high damping prevents oscillations while maintaining responsiveness
            steer_command = inp.steer + correction
            
            steer_command = max(-1.0, min(1.0, steer_command))
            inp = DriverInput(throttle=inp.throttle, steer=steer_command)
            
            if abs(lateral_error) > 1.5:
                alerts.append(f"LKA: recentering... offset {lateral_error:+.2f} m")
        
        self._prev_error = lateral_error
        return inp, alerts


class LaneChangeAlertDemo(ADASModule):
    """Warn if heading changes sharply without a turn signal (simulated)."""
    name = "LCA"

    def __init__(self, heading_rate_threshold: float = math.radians(15)):
        self.threshold = heading_rate_threshold
        self._prev_heading: Optional[float] = None

    def process(self, state: CarState, inp: DriverInput, dt: float):
        alerts = []
        if self._prev_heading is not None and dt > 0:
            rate = (state.heading - self._prev_heading) / dt
            if abs(rate) > self.threshold:
                direction = "right" if rate > 0 else "left"
                alerts.append(f"LCA: rapid lane change {direction} ({math.degrees(rate):.1f}°/s)")
        self._prev_heading = state.heading
        return inp, alerts


# ══════════════════════════════════════════════════════════════════════════════
# Renderer (pygame)
# ══════════════════════════════════════════════════════════════════════════════

if PYGAME_AVAILABLE:
    class Renderer:
        SCALE = 20          # pixels per metre
        BG    = (26, 42, 26)
        GRID  = (50, 90, 50)
        TRAIL = (120, 220, 120, 60)
        CAR_BODY  = (58, 159, 255)
        CAR_WHEEL = (20, 20, 40)
        HUD_GREEN = (100, 220, 130)
        HUD_BLUE  = (100, 170, 255)
        WHITE     = (240, 240, 240)
        AMBER     = (255, 180, 60)

        def __init__(self, width=1000, height=700):
            pygame.init()
            self.w, self.h = width, height
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("2D Car Simulation — ADAS Ready")
            self.clock = pygame.time.Clock()
            self.font_sm = pygame.font.SysFont("monospace", 13)
            self.font_md = pygame.font.SysFont("monospace", 15, bold=True)
            self.trail_surf = pygame.Surface((width, height), pygame.SRCALPHA)
            self._trail_pts = []

        def world_to_screen(self, wx, wy, state: CarState):
            """Convert world coordinates to screen pixels (centred on car)."""
            sx = self.w // 2 + (wx - state.x) * self.SCALE
            sy = self.h // 2 - (wy - state.y) * self.SCALE
            return int(sx), int(sy)

        def draw_grid(self, state: CarState):
            step = 5  # 5m grid
            left  = state.x - self.w / (2 * self.SCALE)
            right = state.x + self.w / (2 * self.SCALE)
            bot   = state.y - self.h / (2 * self.SCALE)
            top   = state.y + self.h / (2 * self.SCALE)
            x0 = math.floor(left / step) * step
            y0 = math.floor(bot  / step) * step
            for gx in range(int(x0), int(right) + step + 1, step):
                sx, _ = self.world_to_screen(gx, 0, state)
                pygame.draw.line(self.screen, self.GRID, (sx, 0), (sx, self.h), 1)
            for gy in range(int(y0), int(top) + step + 1, step):
                _, sy = self.world_to_screen(0, gy, state)
                pygame.draw.line(self.screen, self.GRID, (0, sy), (self.w, sy), 1)
            # Origin cross
            ox, oy = self.world_to_screen(0, 0, state)
            pygame.draw.line(self.screen, self.AMBER, (ox-10, oy), (ox+10, oy), 1)
            pygame.draw.line(self.screen, self.AMBER, (ox, oy-10), (ox, oy+10), 1)

        def draw_trail(self, state: CarState):
            if len(self._trail_pts) < 2:
                return
            pts = [self.world_to_screen(p[0], p[1], state) for p in self._trail_pts]
            if len(pts) >= 2:
                pygame.draw.lines(self.screen, (100, 200, 100, 80), False, pts, 2)

        def draw_car(self, state: CarState, params: CarParams):
            W = params.car_width * self.SCALE
            H = params.wheelbase * self.SCALE
            cx, cy = self.w // 2, self.h // 2

            # Build car surface
            car_surf = pygame.Surface((int(W + 16), int(H + 16)), pygame.SRCALPHA)
            cw, ch = car_surf.get_size()
            # Body
            pygame.draw.rect(car_surf, self.CAR_BODY, (8, 8, cw-16, ch-16), border_radius=4)
            # Windshield
            ws_h = int((ch - 16) * 0.28)
            pygame.draw.rect(car_surf, (160, 220, 240, 180), (10, 10, cw-20, ws_h), border_radius=2)

            # Wheels
            ww, wh = 8, 14

            def draw_wheel(surf, wx, wy, angle_rad):
                ws = pygame.Surface((ww, wh), pygame.SRCALPHA)
                ws.fill(self.CAR_WHEEL)
                ws = pygame.transform.rotate(ws, math.degrees(angle_rad))
                surf.blit(ws, (wx - ws.get_width()//2, wy - ws.get_height()//2))

            draw_wheel(car_surf, 4,        int(ch*0.7), 0)
            draw_wheel(car_surf, cw - 4,   int(ch*0.7), 0)
            draw_wheel(car_surf, 4,        int(ch*0.3), state.steer)
            draw_wheel(car_surf, cw - 4,   int(ch*0.3), state.steer)

            # Rotate whole car surface
            rotated = pygame.transform.rotate(car_surf, math.degrees(state.heading) - 90)
            rect = rotated.get_rect(center=(cx, cy))
            self.screen.blit(rotated, rect.topleft)

        def draw_hud(self, state: CarState, alerts: List[str]):
            lines = [
                (f"Speed   {abs(state.speed):6.2f} m/s ({abs(state.speed)*3.6:.1f} km/h)", self.HUD_GREEN),
                (f"Steer   {math.degrees(state.steer):+6.1f}°", self.HUD_BLUE),
                (f"Heading {math.degrees(state.heading) % 360:6.1f}°", self.WHITE),
                (f"Pos     ({state.x:.1f}, {state.y:.1f}) m", self.WHITE),
                (f"Dist    {state.dist_traveled:.1f} m", self.WHITE),
            ]
            for i, (txt, col) in enumerate(lines):
                surf = self.font_sm.render(txt, True, col)
                self.screen.blit(surf, (12, 10 + i * 18))

            for j, alert in enumerate(alerts):
                surf = self.font_md.render(f"⚠ {alert}", True, self.AMBER)
                self.screen.blit(surf, (12, self.h - 30 - j * 22))

        def add_trail_point(self, state: CarState):
            self._trail_pts.append((state.x, state.y))
            if len(self._trail_pts) > 3000:
                self._trail_pts = self._trail_pts[-3000:]

        def begin_frame(self):
            self.screen.fill(self.BG)

        def end_frame(self):
            pygame.display.flip()
            return self.clock.tick(60)


# ══════════════════════════════════════════════════════════════════════════════
# Simulation controller
# ══════════════════════════════════════════════════════════════════════════════

class Simulation:
    """
    Main loop. Connects:
      keyboard → DriverInput → ADAS modules → PhysicsEngine → Renderer
    """

    def __init__(
        self,
        params: Optional[CarParams] = None,
        adas_modules: Optional[List[ADASModule]] = None,
    ):
        self.params  = params or CarParams()
        self.physics = PhysicsEngine(self.params)
        self.state   = CarState()
        self.adas    = adas_modules or []

    def get_keyboard_input(self) -> DriverInput:
        keys = pygame.key.get_pressed()
        throttle = 0.0
        steer    = 0.0
        if keys[pygame.K_w] or keys[pygame.K_UP]:    throttle =  1.0
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:  throttle = -1.0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:  steer    = 1.0
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: steer    = -1.0
        return DriverInput(throttle=throttle, steer=steer)

    def run_pygame(self):
        if not PYGAME_AVAILABLE:
            print("pygame not available. Use run_headless() instead.")
            return

        renderer = Renderer()
        print("Running. Controls: W/S = throttle/brake, A/D = steer, R = reset, Q = quit")

        last_time = time.time()
        running   = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        self.state = CarState()
                        renderer._trail_pts.clear()

            now = time.time()
            dt  = min(now - last_time, 0.05)
            last_time = now

            # Driver input
            inp = self.get_keyboard_input()

            # ADAS processing (each module may modify input and emit alerts)
            all_alerts: List[str] = []
            for module in self.adas:
                inp, alerts = module.process(self.state, inp, dt)
                all_alerts.extend(alerts)

            # Physics step
            self.state = self.physics.step(self.state, inp, dt)
            renderer.add_trail_point(self.state)

            # Render
            renderer.begin_frame()
            renderer.draw_grid(self.state)
            renderer.draw_trail(self.state)
            renderer.draw_car(self.state, self.params)
            renderer.draw_hud(self.state, all_alerts)
            renderer.end_frame()

        pygame.quit()

    def run_headless(self, steps: int = 200, dt: float = 0.05):
        """Run without a window — useful for testing ADAS logic."""
        print(f"{'Step':>5}  {'X':>8}  {'Y':>8}  {'Speed':>8}  {'Heading':>10}  {'Steer':>8}")
        print("-" * 60)
        # Demo: drive forward, turn right after 100 steps
        for i in range(steps):
            inp = DriverInput(
                throttle=1.0 if i < 150 else 0.0,
                steer=0.5 if i > 80 else 0.0,
            )
            for module in self.adas:
                inp, alerts = module.process(self.state, inp, dt)
                for a in alerts:
                    print(f"  [ALERT] {a}")
            self.state = self.physics.step(self.state, inp, dt)
            if i % 20 == 0:
                s = self.state
                print(f"{i:>5}  {s.x:>8.2f}  {s.y:>8.2f}  "
                      f"{s.speed:>8.2f}  {math.degrees(s.heading):>10.2f}  "
                      f"{math.degrees(s.steer):>8.2f}")
        print("\nFinal state:", self.state.to_dict())


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    params = CarParams(
        max_accel=5.0,
        max_brake=8.0,
        max_steer=math.radians(30),
        aero_drag=0.08,
    )

    # Add ADAS modules here — comment out to disable
    adas = [
        LaneChangeAlertDemo(),
        LaneKeepAssistDemo(lane_y=0.0, kp=0.5, kd=1.8),  # uncomment to try LKA
    ]

    sim = Simulation(params=params, adas_modules=adas)

    if PYGAME_AVAILABLE:
        sim.run_pygame()
    else:
        sim.run_headless(steps=300, dt=0.05)
