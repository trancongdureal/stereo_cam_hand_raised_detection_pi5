# Stereo Vision Calibration System

A portable, complete, and reproducible pipeline for capturing and calibrating a stereo camera pair.

## SUPPORTED HARDWARE
* **Raspberry Pi OS / Linux**: Raspberry Pi Camera via Picamera2/libcamera (Highly Recommended for hardware sync proximity)
* **Generic PC (Windows/Linux/Mac)**: USB webcams via OpenCV.

*Note: This system uses software synchronization. Absolute hardware sync is not provided by standard webcams or software APIs. Static calibration handles this fine.*

---

## 🛑 USER MUST FILL 🛑
You **MUST** edit `config.py` before running. Do not guess these values.

| Field | Example | Unit | How to obtain | Where to put |
|---|---|---|---|---|
| **CAMERA_BACKEND** | `'auto'` | - | Use 'auto', 'opencv', or 'picamera2' | `config.py` |
| **LEFT_CAMERA_ID** | `0` | int | Find via `camera_test.py` | `config.py` |
| **RIGHT_CAMERA_ID** | `1` | int | Find via `camera_test.py` | `config.py` |
| **IMAGE_WIDTH** | `1280` | px | Camera specification | `config.py` |
| **IMAGE_HEIGHT** | `720` | px | Camera specification | `config.py` |
| **FPS** | `30` | fps | Camera specification | `config.py` |
| **CHESSBOARD_PATTERN_SIZE** | `(9, 7)` | count | **INNER CORNERS** of chessboard | `config.py` |
| **SQUARE_SIZE** | `0.02` | meters | Measure 1 printed square with ruler | `config.py` |
| **EXPECTED_BASELINE** | `0.12` | meters | Measure distance between lenses | `config.py` |
| **MINIMUM_CALIBRATION_PAIRS** | `20` | count | Recommended 20-40 | `config.py` |

---

## 🚀 QUICK START & WORKFLOW

### STEP 0: Hardware requirements
- 2 Cameras connected to the same machine.
- A printed chessboard mounted on a rigid, flat surface (clipboard, foam board).

### STEP 1: Copy project to new machine
Ensure you don't copy old `.venv` or `__pycache__` folders. (See `CLEAN_DISTRIBUTION.md`).

### STEP 2: Install Dependencies & Create Venv
Depending on your OS, run the appropriate setup script from the root of the project:
**Windows:**
```powershell
.\setup\setup_windows.ps1
.venv\Scripts\Activate.ps1
```
**Linux (Debian/Ubuntu):**
```bash
./setup/setup_linux.sh
source .venv/bin/activate
```
**Raspberry Pi:**
```bash
./setup/setup_raspberry_pi.sh
source .venv/bin/activate
```

### STEP 3: Run environment check
**Command:** `python environment_check.py`
**Expected Output:** `ENVIRONMENT STATUS: PASS`

### STEP 4: Run camera test
**Command:** `python camera_test.py`
**Expected Output:** A window showing LEFT and RIGHT cameras. Check if LEFT/RIGHT match physical setup. Edit `config.py` if needed.

### STEP 5: Edit config.py
Open `config.py` and fill in the **USER MUST FILL** parameters.

### STEP 6: Capture calibration dataset
**Command:** `python capture_stereo.py`
- Press `A` to toggle Auto-capture.
- Move the chessboard through the entire Field of View.
- Press `Q` to quit.

### STEP 7: Run stereo calibration
**Command:** `python calibrate_stereo.py`
- Computes matrices and saves to `calibration_result/`.

### STEP 8: Read calibration report & validation
**Command:** `python validate_calibration.py`
- Checks baseline, RMS error, and generates a visual rectification test.

### STEP 9: Run stereo 3D test
**Command:** `python stereo_3d.py`
- Tests triangulation.
