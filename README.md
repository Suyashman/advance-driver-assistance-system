# Project 7 ADAS

A Python-based Advanced Driver Assistance Systems (ADAS) simulation focused on lane behavior, steering dynamics, and modular ADAS experimentation.

This project currently includes a real-time 2D car simulator built on a kinematic bicycle model, plus demo ADAS modules such as lane-keeping and lane-change alerts.

## Demo Features

- Real-time vehicle simulation using a kinematic bicycle model
- Keyboard driving controls (throttle, brake, steer, reset)
- Lane Keep Assist (LKA) demo with damping-based recentering logic
- Lane Change Alert (LCA) demo for rapid heading-rate detection
- Live HUD for speed, heading, steering, distance, and alerts
- Grid-based world view and vehicle trail rendering

## Project Structure

```text
.
|-- README.md
|-- simulations/
|   `-- 1_car_simulation.py
`-- screenshots/
	`-- simulation_screenshots/
```

## Tech Stack

- Python 3.10+
- pygame
- numpy

## Quick Start

### 1. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
```

Windows (cmd):

```bash
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install pygame numpy
```

### 3. Run the simulator

```bash
python simulations/1_car_simulation.py
```

If pygame is not installed or available, the script can run in headless mode for text-based output.

## Controls

- W or Up Arrow: throttle
- S or Down Arrow: brake/reverse
- A or Left Arrow: steer
- D or Right Arrow: steer
- R: reset simulation
- Q: quit

## ADAS Architecture

The code is intentionally modular so additional ADAS behaviors can be added quickly.

- CarState: full vehicle state snapshot per timestep
- CarParams: configurable vehicle and dynamics parameters
- PhysicsEngine: pure kinematic update logic
- ADASModule: pluggable base class for assistance logic
- LaneKeepAssistDemo: steering intervention and lane departure handling
- LaneChangeAlertDemo: heading-rate based lane-change warning
- Renderer: pygame visualization and HUD
- Simulation: orchestrates input, ADAS pipeline, physics, and rendering

## Current Lane-Keeping Behavior

LaneKeepAssistDemo attempts to keep the car near the lane centerline and provides lane departure status messages when the vehicle exits the lane bounds.

Current behavior goals:

- Follow lane center in straight driving
- Resist steering that pushes farther away from lane center
- Recenter quickly while minimizing oscillation
- Disable lane detection when lane departure threshold is exceeded

## Future Enhancements

- Tune LKA controller gains for multiple speeds and road profiles
- Add curved-lane support instead of fixed centerline
- Add adaptive cruise control (ACC) module
- Add collision warning and emergency braking module
- Add multi-car cooperative ADAS scenario (HIVE MIND concept)
- Add real sensor/camera pipeline integration

## Why This Project

This repository is designed as a practical ADAS learning and prototyping environment. It demonstrates core ideas from vehicle dynamics, closed-loop control, and modular autonomy software design in a format that is easy to run and extend.

## License

No license file is included yet. Add a LICENSE file (for example MIT) before public reuse.