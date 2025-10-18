# Passive Localization

Estimate robot position using a particle filter, while driving around manually. 

![alt text](media/output.gif)

---

## TL;DR

* **Aim:** Passive localization to estimate robot position using a particle filter.
* **Robot:**
  * **Sensors:** 1D LiDAR with 100 rays (±45° fan, `ray_length=100`).
  * **Actuators:** Two wheels, differentially driven (forward/back, arc turns).
* **Run:** `python robot_control.py`
* **Control:** `W/S` or `↑/↓` to move, `A/D` or `←/→` to turn.
* **What you get:** Octagonal robot, LiDAR visualization, particle filter for position estimation.
* **Why it’s useful:** Simple base for localization experiments, extensible for SLAM or autonomous navigation.

---

## Setup

1. **Install** (Python 3.9+ recommended):
   ```bash
   pip install pygame numpy matplotlib
   ```
2. **Run:**
   ```bash
   python robot_control.py
   ```
3. **Drive:** Use WASD or arrow keys to navigate; particle filter updates position estimate.

> **Note:** Emscripten support for browser builds; desktop Python is default.

---

## How It Works

* **Robot:** Octagonal body (`ROBOT_SIZE=4`), pose as `(x, y, orientation°)`.
* **LiDAR:** 100 rays cast ±45° from heading, detects walls and circular obstacles.
* **Particle Filter:** Estimates position with `NUM_PARTICLES=100`, updated via LiDAR measurements and motion model.
* **Controls:**
  * `W`/`↑` or `S`/`↓`: Move forward/back.
  * `A`/`←` or `D`/`→`: Turn via differential wheel speeds.
* **Collision Detection:** Prevents robot from passing through walls or obstacles.

---

## File Layout

```
.
├── robot_control.py          # Main loop, controls, LiDAR, particle filter
└── utils/
    ├── lidar_utils.py        # Line and circle intersection math
    ├── plotting_utils.py     # Map and robot visualization
    ├── particle_filter.py    # Particle filter for localization
    └── __init__.py
```

---

## Customization

* **Speeds:** Adjust `linear_speed`, `angular_speed` in `robot_control.py`.
* **Robot:** Modify `ROBOT_SIZE`, `wheel_base`, `wheel_radius`.
* **LiDAR:** Tweak `num_rays`, `ray_length`.
* **Map:** Edit `get_map_objects()` in `plotting_utils.py` for walls/circles.

---

## Credits

Based on work by **Nathaniel Helgesen** (GitHub: `spriglithika`).

* [https://github.com/spriglithika](https://github.com/spriglithika)
* [https://github.com/spriglithika/DanielCramer](https://github.com/spriglithika/DanielCramer)

Enhanced with particle filter localization for passive position estimation.

---

## License

**MIT** — see `LICENSE` in this folder.