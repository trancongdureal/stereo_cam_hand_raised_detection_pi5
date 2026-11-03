# MASTER SETUP PROMPT

Copy the block below and paste it directly into Antigravity on the NEW machine.

---

```
Please help me set up this Stereo Vision Calibration project on this new machine. Do not assume any pre-existing environment. Please strictly follow these steps sequentially:

STEP 1: Audit project
Check the project structure to ensure all core files (`config.py`, `requirements.txt`, `setup/`, `src/`, and `.py` scripts) are present.

STEP 2: Detect OS
Identify the current Operating System (Windows, Linux, or macOS).

STEP 3: Detect architecture
Identify the system architecture (e.g., x86_64, ARM).

STEP 4: Detect Python
Verify that Python 3 is installed and accessible.

STEP 5: Create .venv
Run the appropriate setup script from the `setup/` directory for this OS to create a fresh virtual environment (`.venv`). Do NOT use any existing `.venv`. 

STEP 6: Install correct dependencies
Activate the virtual environment and ensure all dependencies from `requirements.txt` (or `requirements-rpi.txt` if on Raspberry Pi) are installed successfully.

STEP 7: Check OpenCV
Verify `cv2` can be imported.

STEP 8: Check Picamera2 if applicable
If on a Raspberry Pi, check if `picamera2` is available.

STEP 9: Detect available camera backend
Determine if we should use `opencv` or `picamera2` based on the system.

STEP 10: Run environment check
Run `python environment_check.py` and ensure it outputs "PASS".

STEP 11: Read config.py
Read the `config.py` file to understand the current configuration.

STEP 12: Tell USER exactly what must be filled
Ask the user to physically check their setup and provide the following details:
- LEFT_CAMERA_ID
- RIGHT_CAMERA_ID
- IMAGE_WIDTH
- IMAGE_HEIGHT
- FPS
- CHESSBOARD_PATTERN_SIZE (Inner corners)
- SQUARE_SIZE (in meters)
- EXPECTED_BASELINE (in meters)
- MINIMUM_CALIBRATION_PAIRS

Do NOT proceed until the user provides these details. Do not guess or hallucinate these values.

STEP 13: Validate camera IDs
Once the user provides the values, update `config.py`. Then ask the user to run `python camera_test.py` to confirm the IDs are correct and the left camera physically corresponds to the left image.

STEP 14: Validate resolution
Confirm the user's resolution is supported by the cameras.

STEP 15: Validate chessboard config
Confirm the user understands that PATTERN_SIZE means inner corners.

STEP 16: Validate square size
Confirm SQUARE_SIZE is in meters.

STEP 17: Validate baseline
Confirm EXPECTED_BASELINE is in meters.

STEP 18: Check dataset
Check if the `calibration_images/left` and `calibration_images/right` directories contain valid image pairs. 

STEP 19: Capture if user requests
If the dataset is missing or the user wants to capture a new one, instruct the user to run `python capture_stereo.py` and wait for them to finish.

STEP 20: Run dataset validation
(The dataset is implicitly validated during calibration, or via existing dataset scripts if present).

STEP 21: Run stereo calibration
Run `python calibrate_stereo.py` to generate the calibration matrices. If dataset is unavailable, print "CALIBRATION NOT EXECUTED — DATASET UNAVAILABLE".

STEP 22: Validate RMS/errors
Run `python validate_calibration.py`. Check the output for the Stereo RMS value.

STEP 23: Validate baseline
In the output of `validate_calibration.py`, verify that the estimated baseline matches the EXPECTED_BASELINE within a reasonable threshold.

STEP 24: Validate rectification
Check that `validate_calibration.py` successfully generated `rectification_test.png`. Instruct the user to open it and verify the horizontal epipolar lines.

STEP 25: Validate Q
Ensure the `Q` matrix exists in `stereo_calibration.npz`.

STEP 26: Validate stereo 3D
Run `python stereo_3d.py` to test the triangulation pipeline.

STEP 27: Print final status
Output a final report formatted exactly like this:
ENVIRONMENT: [PASS / FAIL]
CODE: [PASS / FAIL]
CALIBRATION PIPELINE: [PASS / FAIL]
CAMERA: [PASS / NOT TESTED / FAIL]
DATASET: [READY / NOT READY]
RECTIFICATION: [PASS / NOT TESTED / FAIL]
STEREO 3D: [PASS / NOT TESTED / FAIL]
DOCUMENTATION: [PASS / FAIL]
PORTABILITY: [PASS / FAIL]
PACKAGE STATUS: [READY TO PACKAGE / NOT READY]
```
