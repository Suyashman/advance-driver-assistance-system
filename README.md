# 🚗 Mini ADAS System with HIVE MIND — RC Car Implementation

> A physical prototype of an Advanced Driver Assistance System (ADAS) built on RC cars using Arduino R4 and Python OpenCV. Developed as a scaled demonstration of real-world automotive ADAS concepts, featuring a novel **HIVE MIND** V2V (Vehicle-to-Vehicle) communication layer.

> Built as a project submission for **Tata Technologies** internship consideration.

---

## Table of Contents

- [Project Overview](#project-overview)
- [HIVE MIND — The Core Innovation](#hive-mind--the-core-innovation)
- [System Architecture](#system-architecture)
- [ADAS Features Implemented](#adas-features-implemented)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
- [Wiring Guide](#wiring-guide)
- [Calibration](#calibration)
- [Running the System](#running-the-system)
- [How Each Module Works](#how-each-module-works)
- [HIVE MIND Protocol](#hive-mind-protocol)
- [Serial Command Reference](#serial-command-reference)
- [Demo Scenarios](#demo-scenarios)
- [Roadmap](#roadmap)
- [Real-World ADAS Mapping](#real-world-adas-mapping)
- [License](#license)

---

## Project Overview

This project implements a working ADAS system on a fleet of RC cars, demonstrating the core principles of modern automotive safety systems at a scale that can be physically built and demonstrated. A fixed **overhead (top-view) camera** feeds live video to a laptop running a Python + OpenCV pipeline. The vision system tracks every car in real time, computes ADAS decisions, and dispatches control commands to each car's **Arduino R4** over USB serial or Wi-Fi.

The key innovation beyond standard ADAS is the **HIVE MIND** — a cooperative awareness layer that lets cars share safety-critical events with each other in real time, allowing the entire fleet to react to a hazard before any individual car's sensors can detect it.

```
Overhead Camera
      │
      ▼
Python (OpenCV + ADAS Logic + HIVE MIND Broker)
      │
      ├──── USB Serial / Wi-Fi ────► Arduino R4 (Car A)
      ├──── USB Serial / Wi-Fi ────► Arduino R4 (Car B)
      └──── USB Serial / Wi-Fi ────► Arduino R4 (Car C)
```

---

## HIVE MIND — The Core Innovation

In standard ADAS, each vehicle only reacts to what its own sensors detect. The **HIVE MIND** layer enables **predictive propagation** — safety events are broadcast to all cars the moment they are detected, regardless of whether those cars can yet sense the hazard themselves.

### Use Cases

**Emergency Brake Cascade**
Car A detects an obstacle and triggers AEB. Instead of waiting for Car B and Car C to each independently detect the issue (by which time a collision may be inevitable), Car A instantly broadcasts an emergency event. Cars B and C pre-emptively reduce speed — replicating real-world cooperative braking that is being standardised under V2X protocols.

**Ambulance / Emergency Vehicle Detection**
The overhead vision system detects a flashing red-blue pattern on a prop emergency vehicle. A HIVE broadcast is sent to all cars in the fleet, triggering them to steer toward the lane edge and reduce speed — simulating the real-world requirement for vehicles to yield to emergency services.

**Road Hazard Coordinate Broadcast**
When Car A crosses a simulated hazard (e.g. a bump strip triggering its ultrasonic sensor), it logs the arena coordinates of the hazard. All other cars are warned to slow down when they approach the same coordinate — even if the hazard has moved out of camera view.

**Traffic Jam / Convoy Assist**
When multiple cars bunch up below a speed threshold, the HIVE switches the fleet into convoy mode — maintaining a safe following distance collectively rather than each car computing it independently.

### How It Works Technically

The laptop runs a **Mosquitto MQTT broker**. Each car's Arduino (via ESP32) subscribes to the fleet MQTT topic. The OpenCV pipeline publishes events as JSON payloads:

```json
{
  "event": "EMERGENCY_BRAKE",
  "source_car": "A",
  "position": { "x": 340, "y": 210 },
  "heading": 92.4,
  "timestamp": 1718023441.23
}
```

Receiving Arduinos parse the event and execute pre-programmed responses — pre-braking, path-clearing, or speed reduction — without waiting for their own sensors to trigger.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      PERCEPTION LAYER                           │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │  Overhead Camera │  │   RC Car Arena   │  │  HC-SR04     │  │
│  │  (USB / RTSP)    │  │  Lanes + Objects │  │  Ultrasonic  │  │
│  └────────┬─────────┘  └──────────────────┘  └──────┬───────┘  │
└───────────┼──────────────────────────────────────────┼──────────┘
            │                                          │
┌───────────▼──────────────────────────────────────────▼──────────┐
│                      PROCESSING LAYER (Laptop)                   │
│  ┌───────────────┐  ┌─────────────────┐  ┌──────────────────┐   │
│  │ OpenCV Module │  │  ADAS Logic     │  │  HIVE MIND       │   │
│  │ Detection +   │─►│  LDW, FCW,      │  │  Broker (MQTT)   │   │
│  │ Tracking      │  │  AEB, ACC, BSD  │  │  V2V Events      │   │
│  └───────────────┘  └────────┬────────┘  └──────────────────┘   │
│                               │                                   │
│                    ┌──────────▼──────────┐                        │
│                    │  Command Dispatcher  │                        │
│                    │  Serial / Wi-Fi      │                        │
│                    └──────────┬──────────┘                        │
└───────────────────────────────┼───────────────────────────────────┘
                                │
┌───────────────────────────────▼───────────────────────────────────┐
│                      ACTUATOR LAYER (Per Car)                      │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │
│   │  Arduino R4  │    │  Arduino R4  │    │  Arduino R4  │         │
│   │  Car A       │    │  Car B       │    │  Car C       │         │
│   │  L298N       │    │  L298N       │    │  L298N       │         │
│   │  HC-SR04     │    │  HC-SR04     │    │  HC-SR04     │         │
│   │  ESP32       │    │  ESP32       │    │  ESP32       │         │
│   └──────────────┘    └──────────────┘    └──────────────┘         │
└────────────────────────────────────────────────────────────────────┘
```

---

## ADAS Features Implemented

### Passive (Warning Only)

| Feature | Description | Implementation |
|---|---|---|
| **LDW** — Lane Departure Warning | Alerts when a car drifts out of its lane | HSV color masking + Hough line detection on lane tape |
| **FCW** — Forward Collision Warning | Warns when closing distance to a car ahead is dangerous | Overhead position tracking + heading vector cone check |
| **BSD** — Blind Spot Detection | Detects vehicles in the side-rear danger zone | Local-frame coordinate transform per car |
| **Parking Assist** | Warns of objects at close range | HC-SR04 ultrasonic on each car |

### Active (Vehicle Takes Action)

| Feature | Description | Implementation |
|---|---|---|
| **AEB** — Automatic Emergency Braking | Cuts speed instantly when collision is imminent | PWM override to zero, bypasses ramp |
| **LKA** — Lane Keeping Assist | Proportional steering correction to stay in lane | Differential motor speed via P-controller |
| **ACC** — Adaptive Cruise Control | Maintains a set following distance to the car ahead | PID loop on inter-car distance in pixels |
| **HIVE MIND** | Fleet-wide cooperative hazard response | MQTT publish/subscribe, JSON event payloads |

### Inertia Simulation

Rather than instant speed changes (unrealistic for demonstration), all speed transitions use a software ramp:

- **Acceleration rate:** gradual, simulating vehicle mass
- **Normal braking rate:** faster than acceleration, simulating friction
- **AEB rate:** instantaneous PWM cut — the only exception

---

## Hardware Requirements

### Per Car (×3 for full demo)

| Component | Model | Purpose |
|---|---|---|
| Microcontroller | Arduino R4 Minima or R4 WiFi | Main car brain |
| Motor Driver | L298N module | Drive DC motors |
| DC Motors + Chassis | Generic 2WD RC chassis kit | Car movement |
| Proximity Sensor | HC-SR04 ultrasonic | Local obstacle detection |
| Wi-Fi Module | ESP32 (if using R4 Minima) | MQTT / HIVE MIND |
| Buzzer | Active 5V buzzer | Audio ADAS alerts |
| Battery | 7.4V Li-Po or 9V | Car power supply |
| Jumper wires | — | Connections |

### Shared (×1)

| Component | Model | Purpose |
|---|---|---|
| Overhead Camera | 1080p USB webcam or phone (DroidCam) | Top-view perception |
| Tripod / mount | Any stable mount ~1.5–2m height | Camera positioning |
| Laptop | Any, running Python 3.11+ | Vision + ADAS processing |
| Colored tape | White or yellow | Lane markings on arena floor |
| Colored stickers / ArUco markers | 4×4 cm, distinct per car | Car identification |

---

## Software Requirements

### Python (Laptop)

```
Python 3.11+
opencv-python >= 4.8
numpy
pyserial
paho-mqtt
```

Install all at once:

```bash
pip install opencv-python numpy pyserial paho-mqtt
```

### Arduino

- Arduino IDE 2.x — [arduino.cc/en/software](https://www.arduino.cc/en/software)
- Board: Arduino UNO R4 (install via Arduino IDE Board Manager)
- No additional libraries required for basic operation

### MQTT Broker (for HIVE MIND)

```bash
# Ubuntu / Debian
sudo apt install mosquitto mosquitto-clients

# macOS
brew install mosquitto

# Windows
# Download installer from mosquitto.org
```

Start the broker:

```bash
mosquitto -v
```

---

## Repository Structure

```
mini-adas-hive-mind/
│
├── arduino/
│   ├── car_controller/
│   │   └── car_controller.ino       # Main Arduino sketch (upload to each car)
│   └── README.md                    # Arduino-specific wiring notes
│
├── python/
│   ├── main.py                      # Entry point — runs the full ADAS pipeline
│   ├── camera.py                    # Camera capture and frame management
│   ├── detection.py                 # Car detection (HSV / ArUco tracking)
│   ├── lane_detection.py            # Lane tape detection and LDW logic
│   ├── adas_logic.py                # FCW, AEB, LKA, ACC, BSD modules
│   ├── hive_mind.py                 # MQTT broker interface and V2V events
│   ├── dispatcher.py                # Serial command sender per car
│   ├── calibration.py               # HSV tuner + pixels-per-cm tool
│   └── config.py                    # All tunable constants in one place
│
├── docs/
│   ├── wiring_diagram.png           # L298N → Arduino R4 wiring
│   ├── arena_setup.png              # Recommended arena layout
│   └── system_architecture.png     # Full architecture diagram
│
├── demo/
│   └── scenarios.md                 # Step-by-step demo script for presentation
│
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/mini-adas-hive-mind.git
cd mini-adas-hive-mind
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Flash the Arduino

Open `arduino/car_controller/car_controller.ino` in Arduino IDE.

Select your board: **Tools → Board → Arduino UNO R4 Minima** (or WiFi).

Select your port: **Tools → Port → COMx** (Windows) or `/dev/ttyUSBx` (Linux/Mac).

Click **Upload**. Repeat for each car, noting the COM port of each.

### 4. Start the MQTT broker

```bash
mosquitto -v
```

### 5. Configure your setup

Open `python/config.py` and set:

```python
# Serial ports for each car
CAR_A_PORT = "COM3"
CAR_B_PORT = "COM4"
CAR_C_PORT = "COM5"

# Camera index (0 = first USB camera)
CAMERA_INDEX = 0

# MQTT broker
MQTT_BROKER = "localhost"
MQTT_PORT   = 1883

# Lane positions (pixels) — calibrate with calibration.py first
LEFT_LANE_X  = 150
RIGHT_LANE_X = 490

# Scale factor
PIXELS_PER_CM = 5.0

# ADAS thresholds (cm)
FCW_DISTANCE_CM = 40
AEB_DISTANCE_CM = 20
```

### 6. Run calibration

```bash
python python/calibration.py
```

This opens the HSV tuner window. Adjust sliders until only your car's color marker shows white in the mask. Note the values and paste them into `config.py`. Then place a ruler in the arena and measure `PIXELS_PER_CM`.

### 7. Run the system

```bash
python python/main.py
```

---

## Wiring Guide

### L298N → Arduino R4

| L298N Pin | Arduino R4 Pin | Notes |
|---|---|---|
| IN1 | D5 | Left motor direction |
| IN2 | D6 | Left motor direction |
| ENA | D9 | Left motor speed (PWM) |
| IN3 | D7 | Right motor direction |
| IN4 | D8 | Right motor direction |
| ENB | D10 | Right motor speed (PWM) |
| GND | GND | Common ground |
| VCC (12V) | External battery | Do NOT power from Arduino 5V |
| 5V out | Arduino VIN | (Optional) powers Arduino from L298N |

### HC-SR04 → Arduino R4

| HC-SR04 Pin | Arduino R4 Pin |
|---|---|
| VCC | 5V |
| GND | GND |
| TRIG | D11 |
| ECHO | D12 |

### Buzzer → Arduino R4

| Buzzer Pin | Arduino R4 Pin |
|---|---|
| + (positive) | D13 |
| − (negative) | GND |

> **Important:** Power the L298N from your battery pack directly. Never draw motor power from the Arduino's 5V pin — motors cause voltage spikes that will reset or damage the Arduino.

---

## Calibration

### HSV Color Calibration

Run:
```bash
python python/calibration.py --mode hsv
```

Place each car in camera view one at a time. Adjust the H/S/V sliders until the Mask window shows a clean white blob over just that car's marker. Record the Low/High values for each car into `config.py`.

**Tip:** Use distinct, saturated colors with no overlap — e.g. red for Car A, green for Car B, blue for Car C.

### Pixels-per-CM Calibration

Run:
```bash
python python/calibration.py --mode scale
```

Place a ruler or tape measure in the arena. Click two points in the camera window that correspond to a known distance (e.g. 30 cm). The tool outputs your `PIXELS_PER_CM` value.

### Lane Boundary Calibration

Run:
```bash
python python/calibration.py --mode lanes
```

Click the left and right lane edge positions in the camera view. Values are saved automatically to `config.py`.

---

## Running the System

### Full ADAS pipeline (all cars, all features)

```bash
python python/main.py
```

### Single car test (no HIVE MIND)

```bash
python python/main.py --cars A --no-hive
```

### HIVE MIND only (test V2V events without vision)

```bash
python python/main.py --hive-test
```

### Keyboard controls during runtime

| Key | Action |
|---|---|
| `q` | Quit |
| `p` | Pause / resume all cars |
| `a` | Trigger manual AEB on Car A |
| `h` | Trigger a manual HIVE broadcast (for demo) |
| `d` | Toggle debug overlay |
| `r` | Reset all cars to stopped state |

---

## How Each Module Works

### `detection.py` — Car Tracking

Captures each frame from the overhead camera. For each car, it applies an HSV color mask to isolate the car's marker sticker. The largest matching contour is taken as the car's position. `cv2.moments()` gives the centroid (cx, cy) and `cv2.minAreaRect()` gives the heading angle.

For more robust tracking, **ArUco markers** are supported as an alternative to color stickers — each car gets a printed ArUco ID tag, and `cv2.aruco.detectMarkers()` gives position and orientation directly.

### `lane_detection.py` — LDW and LKA

Applies a threshold or color mask to isolate lane tape. Hough line transform (`cv2.HoughLinesP`) finds the lane lines. Once the left and right boundaries are known, each car's centroid is checked against them. If a car drifts outside the threshold, an LDW warning is raised. For LKA, a proportional controller computes a steering correction based on how far off-center the car is, and sends a differential speed command to the motors.

### `adas_logic.py` — FCW, AEB, ACC, BSD

Using the known pixel positions and headings of all cars, this module computes:

- **FCW/AEB:** Whether a car is inside another car's forward detection cone, and how close. Threshold distances are in real-world cm (converted via `PIXELS_PER_CM`).
- **ACC:** A PID controller that adjusts following-car speed to maintain a target gap.
- **BSD:** A coordinate transform into each car's local frame. If another car falls within the side-rear zone rectangle, a blind spot alert is raised.

### `hive_mind.py` — V2V Communication

Wraps a `paho-mqtt` client. Publishes JSON event payloads to `/adas/hive/events`. Each Arduino's ESP32 subscribes to this topic and executes the appropriate pre-programmed response when an event arrives. The module also handles event de-duplication (so one emergency brake event doesn't trigger 10 redundant broadcasts) and expiry (events older than 2 seconds are discarded).

### `dispatcher.py` — Serial Command Sending

Maintains an open `pyserial` connection to each Arduino. Commands are sent as newline-terminated strings. Incoming acknowledgements from the Arduino (e.g. `"ACK:AEB"`) are read and logged for debugging.

---

## HIVE MIND Protocol

### Event Topics

| Topic | Direction | Description |
|---|---|---|
| `/adas/hive/events` | Laptop → All cars | General HIVE broadcast |
| `/adas/hive/car_A/status` | Car A → Laptop | Car A status updates |
| `/adas/hive/car_B/status` | Car B → Laptop | Car B status updates |
| `/adas/hive/car_C/status` | Car C → Laptop | Car C status updates |

### Event Payload Format

```json
{
  "event":      "EMERGENCY_BRAKE",
  "source":     "car_A",
  "position":   { "x": 340, "y": 210 },
  "heading":    92.4,
  "severity":   "HIGH",
  "timestamp":  1718023441.23,
  "ttl":        2.0
}
```

### Event Types

| Event | Severity | Triggered By | Fleet Response |
|---|---|---|---|
| `EMERGENCY_BRAKE` | HIGH | AEB on any car | All following cars pre-brake |
| `AMBULANCE_DETECTED` | HIGH | Vision: red/blue flash pattern | All cars steer to lane edge |
| `ROAD_HAZARD` | MEDIUM | Ultrasonic spike + coordinates | Cars slow at hazard coordinate |
| `TRAFFIC_JAM` | MEDIUM | Multiple cars below speed threshold | Fleet switches to convoy mode |
| `LANE_BLOCKED` | MEDIUM | Vision: obstacle in lane | Cars behind warned to expect stop |

---

## Serial Command Reference

Commands sent from Python to Arduino over serial. All commands are newline-terminated (`\n`).

| Command | Effect |
|---|---|
| `FORWARD` | Accelerate to cruise speed (ramped) |
| `BACKWARD` | Reverse at reduced speed (ramped) |
| `STOP` | Decelerate to zero (ramped) |
| `AEB` | Instant PWM cut to zero (bypasses ramp) |
| `LEFT` | Differential steer left (80/160 L/R ratio) |
| `RIGHT` | Differential steer right (160/80 L/R ratio) |
| `SPEED:150` | Set target speed to 150 (0–255) |
| `DIFF:120:160` | Set left motor to 120, right to 160 |
| `BUZZ:1` | Turn buzzer on (ADAS alert) |
| `BUZZ:0` | Turn buzzer off |

### Arduino Response Codes

The Arduino sends acknowledgements back over serial for debugging:

| Response | Meaning |
|---|---|
| `ACK:FORWARD` | Forward command received |
| `ACK:AEB` | AEB executed |
| `ULTRA:12` | Ultrasonic reading of 12 cm |
| `ERR:UNKNOWN` | Unrecognised command |

---

## Demo Scenarios

### Scenario 1 — Emergency Brake Cascade (HIVE MIND)

1. Place Car A at the front, Car B behind it, Car C behind Car B.
2. Start all cars moving forward.
3. Place an obstacle in front of Car A.
4. Car A triggers AEB. Watch Cars B and C slow down immediately — before they are close enough to detect the obstacle themselves.
5. Highlight in the OpenCV window the HIVE broadcast event and the cascade effect.

### Scenario 2 — Lane Departure Warning and Correction

1. Run Car A along the lane.
2. Manually override the steering to drift toward the lane edge.
3. LDW warning appears on screen. Lane Keeping Assist fires a correction command.
4. Car steers back toward center.

### Scenario 3 — Ambulance Detection

1. Place a small prop car with a red+blue flashing LED on one road.
2. Run the fleet on an adjacent road.
3. Vision detects the flashing pattern, HIVE broadcasts `AMBULANCE_DETECTED`.
4. All fleet cars steer toward the lane edge and reduce speed.

### Scenario 4 — Adaptive Cruise Control

1. Set Car A to cruise at a fixed speed.
2. Set Car B behind Car A with ACC enabled.
3. Slow Car A manually (via keyboard).
4. Watch Car B automatically slow to match the new gap — without any explicit command.

---

## Roadmap

- [x] Basic motor control with serial commands
- [x] Software inertia / braking ramp simulation
- [x] Overhead camera car detection (HSV color)
- [x] ArUco marker tracking support
- [x] Lane detection and LDW
- [x] Lane Keeping Assist (proportional control)
- [x] Forward Collision Warning
- [x] Automatic Emergency Braking
- [x] Blind Spot Detection
- [x] HIVE MIND MQTT broker
- [x] Emergency brake cascade
- [x] Ambulance detection
- [ ] Adaptive Cruise Control (PID, in progress)
- [ ] Self-parking demonstration
- [ ] Road hazard coordinate memory
- [ ] Web dashboard (Flask) for live arena view
- [ ] ArUco-only tracking (drop color dependency)
- [ ] Onboard camera variant (future Phase 2)

---

## Real-World ADAS Mapping

This project maps directly to production ADAS concepts and industry standards:

| This Project | Real-World Equivalent | Standard |
|---|---|---|
| Overhead camera + OpenCV tracking | Radar + camera sensor fusion | ISO 26262 |
| Forward Collision Warning | Production FCW systems | Euro NCAP AEB test |
| Automatic Emergency Braking | AEB — now mandatory in new EU vehicles | UNECE R152 |
| Lane Keeping Assist | LKA in production vehicles | UNECE R79 |
| HIVE MIND broker | V2X (Vehicle-to-Everything) communication | C-V2X / DSRC / IEEE 802.11p |
| HIVE emergency brake cascade | Cooperative Adaptive Cruise Control (CACC) | SAE J2945 |
| Ambulance detection + fleet response | Emergency Vehicle Preemption (EVP) | NTCIP 1202 |
| Software inertia ramp | Jerk-limited motion planning | ISO 15622 |

The HIVE MIND system in particular is a physical prototype of **V2X cooperative ADAS** — one of the most actively developed areas in the automotive industry, targeted at ADAS Level 3 and above.

---

## Acknowledgements

- Tata Technologies — for the ADAS technical brief and internship opportunity context
- OpenCV community — for comprehensive vision library documentation
- Arduino community — for L298N motor driver guides
- Eclipse Mosquitto — lightweight open-source MQTT broker

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with Arduino R4, Python, OpenCV, and a fleet of RC cars. Designed to demonstrate that ADAS and V2X concepts are implementable, testable, and demonstrable at any scale.*
