# CLEAN DISTRIBUTION GUIDE

## WHAT TO COPY
When moving this project to a new machine, **ONLY** copy the following files and directories:

*   `src/` (All source code)
*   `setup/` (Setup scripts)
*   `config.py`
*   `requirements.txt`
*   `requirements-rpi.txt`
*   `README.md`
*   `MASTER_SETUP_PROMPT.md`
*   `CLEAN_DISTRIBUTION.md`
*   `.gitignore`
*   `environment_check.py`
*   `camera_test.py`
*   `capture_stereo.py`
*   `calibrate_stereo.py`
*   `validate_calibration.py`
*   `stereo_3d.py`

## WHAT NOT TO COPY
Do **NOT** copy any of the following directories or files to the new machine:

*   `.venv/` (Virtual environment is machine-specific and MUST be recreated on the new machine)
*   `__pycache__/` (Compiled python code)
*   `calibration_images/` (Unless you intend to calibrate using the exact same old dataset)
*   `calibration_result/` (Unless you want to use the old calibration matrices on the new machine, which is usually wrong if hardware moved)
*   `debug/` (Temporary images and logs)
*   Any `.log` files
*   Any IDE cache folders (e.g., `.vscode/`, `.idea/`)

## EXPLANATION
Python virtual environments (`.venv`) contain absolute paths and machine-specific binaries. Copying them will cause broken imports, `pip` errors, and python path issues. Always recreate the environment using the provided `setup` scripts.
