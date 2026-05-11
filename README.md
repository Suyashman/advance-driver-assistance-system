## Project overview: Mini ADAS with HIVE MIND

This is a genuinely strong internship project. Tata Technologies works in automotive engineering and ADAS is one of their core focus areas — building a physical, demonstrable implementation (even at RC scale) puts you miles ahead of someone who built a simulation.

---

## Top-view camera vs. onboard camera

Your instinct is correct — go with the **top-view (overhead) camera for the demo**. Here's the reasoning:

The top-view setup lets you see all cars, lanes, and obstacles simultaneously, which is exactly what you need to demonstrate multi-car HIVE MIND behaviour. With an onboard camera, you'd only see what one car sees, and demonstrating V2V (vehicle-to-vehicle) communication becomes nearly impossible to show visually. You can still talk about how real ADAS uses onboard cameras in your presentation — the overhead camera is purely a pragmatic demo choice that an interviewer will understand immediately.

Mount a webcam or phone camera (using DroidCam or similar) on a tripod about 1.5–2 metres above the arena looking straight down. Connect it to your laptop via USB or Wi-Fi.

---

## How the car drives (Arduino R4)

Each RC car has an Arduino R4 Minima (or WiFi variant) as its brain. The R4 receives commands from the laptop and translates them into physical motor actions.

**Hardware per car:**
- L298N motor driver (controls 2 DC motors — one for drive, one for steering, or use a servo for steering)
- HC-SR04 ultrasonic sensor (local proximity — functions as the onboard "backup camera" / parking sensor)
- Buzzer (audio ADAS alerts — LDW warning sound, FCW beep)
- NRF24L01 or ESP32 module (for HIVE MIND wireless communication between cars)
- Optional: small OLED display for showing current ADAS mode

**Communication from laptop to Arduino:** Use USB serial (pyserial in Python) for a wired demo, or an ESP32 attached to the Arduino for wireless. Wired is highly recommended for your first version — it eliminates Wi-Fi latency from the equation, and you can add wireless in V2 if time allows.

**Command format:** The laptop sends simple JSON packets over serial, like `{"car_id": 1, "speed": 120, "steer": 0, "brake": true, "mode": "AEB"}`. The Arduino parses this and drives the motors accordingly.

---

## How the OpenCV processing works

The Python application on your laptop does all the heavy lifting:

**Step 1 — Frame capture:** Grab frames from the overhead camera at 30fps using `cv2.VideoCapture`.

**Step 2 — Lane detection:** Tape colored lanes on your arena floor. Use HSV color masking and Hough line transforms to detect lane boundaries. This gives you LDW (Lane Departure Warning) — if a car's centroid drifts outside the lane mask, fire a warning.

**Step 3 — Car tracking:** Color-code each car with a distinct marker on top (a colored sticker or LED). Use color segmentation + `cv2.findContours` to locate each car's position and heading each frame. You can also use ArUco markers (a type of QR code OpenCV natively supports) for much more robust tracking — each car gets a unique ArUco ID and the library gives you position and orientation automatically.

**Step 4 — ADAS logic:** With every car's position and heading known, you compute:
- **FCW/AEB:** Is Car B within a dangerous distance of Car A in the direction Car A is heading? → trigger braking
- **LDW:** Is a car's centroid outside its lane polygon?
- **ACC (Adaptive Cruise Control):** Maintain a target gap to the car ahead by modulating speed
- **Blind spot detection:** Is another car within a defined zone beside a car?

**Step 5 — Command dispatch:** Results are packaged and sent to each Arduino over serial.

---

## HIVE MIND — how it works

This is the unique feature that will make the project memorable. The core idea is **predictive propagation** — events propagate to cars before they reach the event themselves.

**The broker:** Run a lightweight MQTT broker (Mosquitto) on your laptop. Each Arduino (via ESP32) subscribes to an MQTT topic. The OpenCV application acts as the publisher.

**Example flow — emergency braking:**
1. Car A's ultrasonic sensor detects an obstacle and triggers hard braking. It sends a signal to the laptop: `"car_A: EMERGENCY_BRAKE"`.
2. The laptop publishes to the MQTT topic `/hive/emergency` with Car A's position and heading.
3. Cars B and C (behind Car A) receive the event and pre-emptively reduce speed — even though their own sensors haven't detected anything yet.
4. The OpenCV display shows a visual cascade — a "shockwave" graphic propagating backwards through the fleet.

**Example flow — ambulance detection:**
1. OpenCV detects a red+blue flashing pattern (simulate this with a small LED on a prop car). It classifies this as an emergency vehicle.
2. The HIVE broadcasts to all cars: `"ambulance detected at position X, heading Y"`.
3. Cars in the projected path reduce speed or steer to the side.

**Other HIVE scenarios you can demo:**
- Road surface anomaly (simulated with a bump strip) — first car to cross it warns all others to slow down at that coordinate
- Traffic jam assist — if multiple cars bunch up, the fleet automatically switches to convoy mode with reduced speed
- Hazard light cascade — if one car "detects" a simulated pothole (ultrasonic spike), it flags coordinates and all cars in the convoy slow at that X,Y position

---

## ADAS features you can implement (priority order)

**Start with these (Phase 1):**
- Lane departure warning — color masking + Hough lines
- Forward collision warning — distance estimation from overhead positions
- Automatic emergency braking — command override to brake motors
- Adaptive cruise control — PID loop on following distance

**Add these next (Phase 2):**
- Blind spot detection — zone polygon checks per car
- Parking assistance — ultrasonic sensor data from the car itself
- HIVE MIND emergency brake cascade
- HIVE MIND ambulance detection

**Stretch goals (Phase 3):**
- Lane keeping assist — steer correction commands
- HIVE MIND hazard coordinate broadcasting
- Dashboard UI in Python (tkinter or a web interface) showing live car positions, active ADAS events, and HIVE broadcast logs

---

## Hardware shopping list

| Component | Qty | Purpose |
|---|---|---|
| Arduino R4 Minima or WiFi | 3 | One per car |
| L298N motor driver | 3 | Motor control |
| DC motors + chassis kit | 3 | RC car base |
| HC-SR04 ultrasonic | 3+ | Local proximity sensing |
| ESP32 module (if using R4 Minima) | 3 | Wi-Fi / MQTT |
| NRF24L01 (alternative) | 6 | Simpler V2V radio |
| Overhead webcam (1080p) | 1 | Top-view perception |
| Tripod or camera mount | 1 | Stable overhead view |
| Colored tape | — | Lane markings |
| ArUco marker printouts | 3 | Car identification |
| Servo motor | 3 | Steering (optional) |
| Buzzer | 3 | Audio ADAS alerts |
| Li-Po battery 7.4V | 3 | Car power |

**Software stack:** Python 3, OpenCV 4, pyserial, paho-mqtt, Mosquitto broker, NumPy, optionally Flask for a web dashboard.

---

## Feasibility and Tata Technologies fit

This project is very feasible within 4–8 weeks solo. The overhead camera approach removes the hardest part (real-time onboard computer vision) and lets you focus on the ADAS logic and multi-car coordination, which is actually the more impressive and differentiating work.

For Tata Technologies specifically: they work closely with Tata Motors on embedded systems, ADAS integration, and connected vehicle platforms. The HIVE MIND feature maps directly to real-world V2X (Vehicle-to-Everything) communication standards like C-V2X and DSRC that the industry is actively developing. Framing it that way in your presentation — "this is a physical prototype of V2X cooperative ADAS" — shows you understand where the industry is heading, not just what textbooks say ADAS is.

The project demonstrates computer vision, embedded systems, real-time communication protocols, and systems-level thinking in one package. That's a strong combination.