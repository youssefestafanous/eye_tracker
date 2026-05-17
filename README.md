# Eye Tracker

A real-time gaze tracking system built with Python and MediaPipe, using a webcam to estimate where a user is looking on screen. Designed and implemented from scratch as a personal project.

---

## Overview

This system tracks eye movement using facial landmark detection and maps iris position to screen coordinates. It includes a physically-grounded 3D eye model, multi-point calibration, and several accuracy improvements developed iteratively through testing.

The goal was to build something genuinely usable — not just a proof of concept — with features like profile saving, blink handling, and drift correction that make it practical for extended sessions.

---

## Features

### Calibration
- **9-point calibration** — center, four corners, and four edge midpoints
- **Auto-capture** — detects when gaze is stable and captures automatically; no button-pressing required after the first point
- **Averaged sampling** — collects ~2 seconds of data per point and averages for robustness
- **Profile system** — save and reload calibration profiles (1–10) so you don't have to recalibrate each session
- **Optional advanced mode** — captures face normal vectors at each calibration point for head-roll compensation

### Gaze Estimation
- **3D Eye Sphere Model** — models each eye as a 12mm sphere; calculates true 3D gaze rays and finds their intersection with the screen plane (binocular fusion)
- **2D interpolation fallback** — KD-tree nearest-neighbour lookup over a learned iris→screen position map, built from the eye movement paths between calibration points
- **Head pose compensation** — corrects for depth (distance from camera), XY translation, and IPD (interpupillary distance) changes
- **Roll compensation** — optional correction for head tilt using face normal vectors (advanced calibration mode)
- **Calibration point anchoring** — snaps gaze to exact calibration positions when iris matches, ensuring zero error at known points

### Signal Quality
- **Blink detection** — Eye Aspect Ratio (EAR) based, with a per-user baseline calibrated at session start; freezes gaze position during blinks and for a short stabilisation window after
- **Outlier rejection** — discards physiologically impossible eye movements (>5000 px/s)
- **Fixation detection** — distinguishes fixation from saccades and applies tighter smoothing during fixation
- **Anti-drift** — re-references head pose after 2+ seconds of centre fixation, adapting to posture shifts over long sessions

### Validation
- **9-point accuracy test** — measure mean, median, and per-target error in pixels and approximate visual degrees
- **Error vector storage** — correction vectors saved to profile for future use

---

## Technical Stack

| Component | Library |
|---|---|
| Face & iris landmarks | MediaPipe Face Mesh |
| Image processing | OpenCV |
| Numerical computation | NumPy |
| Spatial indexing | SciPy KDTree |
| Screen resolution | Tkinter |
| Profile persistence | JSON |

---

## Requirements

```
opencv-python
mediapipe
numpy
scipy
```

Install with:
```bash
pip install opencv-python mediapipe numpy scipy
```

Requires Python 3.8+ and a webcam.

---

## Usage

```bash
python eye_tracker.py
```

On launch, choose to load an existing profile or run a new calibration. After calibration, the tracker runs fullscreen. Calibration profiles are saved to a `profiles/` directory.

### Hotkeys

| Key | Action |
|---|---|
| `S` | Manual capture during calibration |
| `A` | Toggle advanced calibration (head tilt) / toggle adaptive calibration |
| `3` | Toggle 3D eye sphere model vs 2D interpolation |
| `V` | Toggle 3D visualisation overlay |
| `T` | Run accuracy validation |
| `R` | Recalibrate |
| `Q` / `ESC` | Quit |

---

## How It Works

**Calibration** captures the normalised iris position (relative to eye socket centre and width) at each of 9 screen locations, along with face geometry for head pose reference. It also records the continuous iris trajectory *between* calibration points, building a dense iris→screen position map.

**During tracking**, the 3D model calculates a gaze ray for each eye from the eye sphere centre through the iris, then finds where those rays intersect a virtual screen plane. This output is blended with the KD-tree interpolation result (80/20 weighting), then corrected for head movement, IPD change, and blink events before smoothing.

---

## Accuracy

Typical performance on a 1920×1080 display at ~60cm:

- Mean error: ~80–120px under stable conditions
- Fixation smoothing reduces jitter to ~20–30px during held gaze
- Corner accuracy is anchored to calibration positions (0px error at exact match)

Accuracy is highly dependent on camera quality, lighting, and how still the user holds their head.

---

## Project Status

Functional and tested. Possible future directions:
- Gaze-contingent UI interactions
- Dwell-click functionality
- Export of gaze data to CSV for analysis
