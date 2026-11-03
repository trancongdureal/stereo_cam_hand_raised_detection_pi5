import os

# ==============================================================================
# USER MUST EDIT THESE VARIABLES BEFORE RUNNING
# ==============================================================================

# CAMERA_BACKEND
# Options: 'auto', 'picamera2', 'opencv'
# 'auto': Tries to detect picamera2, falls back to opencv.
# 'picamera2': Forces usage of Raspberry Pi Camera via Picamera2.
# 'opencv': Forces usage of standard OpenCV VideoCapture (for USB cameras).
CAMERA_BACKEND = 'auto'

# Camera IDs or Indexes
LEFT_CAMERA_ID = 0
RIGHT_CAMERA_ID = 1

# ==============================================================================
# DUAL RASPBERRY PI 5 NETWORK CONFIGURATION
# ==============================================================================
# Khi dùng 2 con Pi 5: Pi #1 (Master) gắn Camera Trái, Pi #2 (Worker) gắn Camera Phải
USE_NETWORK_RIGHT_CAMERA = True
RIGHT_CAMERA_IP = '192.168.88.2'  # Thay bằng IP thực tế của con Pi #2
RIGHT_CAMERA_PORT = 5555

# Resolution for calibration (LEFT and RIGHT must be exactly the same)
# Ensure your cameras support this resolution.
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# FPS for live preview (and capture rate limit)
FPS = 30

# CHESSBOARD_PATTERN_SIZE
# MUST BE THE NUMBER OF INNER CORNERS, NOT THE NUMBER OF SQUARES.
# Example: If your printed chessboard has 10 x 8 squares, 
# then the inner corners are 9 x 7.
CHESSBOARD_PATTERN_SIZE = (8, 5)

# SQUARE_SIZE
# The real-world physical size of a single square on the printed chessboard.
# Measure this carefully with a caliper or ruler.
# Units: Meters (e.g., 20mm = 0.02)
# This unit determines the unit for translation (T) and baseline in 3D.
SQUARE_SIZE = 0.042

# NUMBER_OF_CALIBRATION_PAIRS
# Recommended: 20-40 good pairs. 
# Capture more, then the calibration script will reject bad ones.
NUMBER_OF_CALIBRATION_PAIRS = 30

# AUTO_CAPTURE_INTERVAL
# Time in seconds between automatic captures when "AUTO" mode is enabled.
AUTO_CAPTURE_INTERVAL = 1.5

# OUTPUT_DIRECTORY
# Where to save the images, results, and debug data.
OUTPUT_DIRECTORY = 'stereo_vision'

# EXPECTED_BASELINE
# The physically measured distance between the left and right camera lenses.
# Units must match SQUARE_SIZE (e.g., in meters).
# Used purely for validation/warnings, it does not force the calibration.
EXPECTED_BASELINE = 0.5045  # Example: 12 cm = 0.12 m

# ==============================================================================
# ADVANCED SETTINGS (Usually don't need to edit)
# ==============================================================================

# Require chessboard to be found in both cameras to save a frame pair in AUTO mode?
REQUIRE_CHESSBOARD_FOR_CAPTURE = True

# Calibration distortion model
# If True, computes k1, k2, p1, p2, k3. 
# If False, might use rational model (up to k6) if OpenCV is configured, 
# but standard is sufficient for normal lenses.
USE_RATIONAL_MODEL = False

# Baseline warning threshold (in percentage)
# If the estimated baseline differs from EXPECTED_BASELINE by more than this %,
# a warning is printed.
BASELINE_WARNING_THRESHOLD_PERCENT = 5.0

# Sub-directories setup
DIR_CALIB_IMAGES = os.path.join(OUTPUT_DIRECTORY, 'calibration_images')
DIR_LEFT_IMAGES = os.path.join(DIR_CALIB_IMAGES, 'left')
DIR_RIGHT_IMAGES = os.path.join(DIR_CALIB_IMAGES, 'right')
DIR_CALIB_RESULT = os.path.join(OUTPUT_DIRECTORY, 'calibration_result')
DIR_DEBUG = os.path.join(OUTPUT_DIRECTORY, 'debug')
