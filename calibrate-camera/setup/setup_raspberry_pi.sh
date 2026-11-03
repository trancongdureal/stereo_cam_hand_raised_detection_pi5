#!/bin/bash
# Setup script for Raspberry Pi OS

echo "==========================================="
echo "   Stereo Vision - Raspberry Pi Setup      "
echo "==========================================="

echo "[1/5] Checking System Dependencies..."
echo "For Picamera2, we need libcamera and python3-picamera2."
echo "If this fails, you may need to run: sudo apt install libcamera-apps python3-picamera2"

if ! dpkg -s python3-picamera2 &> /dev/null; then
    echo "WARNING: python3-picamera2 is not installed via apt."
    echo "This is highly recommended for Picamera2 support on Raspberry Pi."
    echo "Run: sudo apt install python3-picamera2"
else
    echo "python3-picamera2 is installed."
fi

echo "[2/5] Creating virtual environment in .venv..."
# On Pi, we often need --system-site-packages to access python3-picamera2 installed via apt
if [ ! -d "../.venv" ]; then
    python3 -m venv --system-site-packages ../.venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment. Do you have python3-venv installed?"
        echo "Try: sudo apt install python3-venv"
        exit 1
    fi
else
    echo "Virtual environment already exists."
fi

echo "[3/5] Activating virtual environment & installing dependencies..."
source ../.venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r ../requirements-rpi.txt

echo "[4/5] Creating necessary directories..."
mkdir -p ../calibration_images/left
mkdir -p ../calibration_images/right
mkdir -p ../calibration_result
mkdir -p ../debug

echo "==========================================="
echo "Setup Complete!"
echo "To begin, activate the environment:"
echo "    source .venv/bin/activate"
echo "Then run the environment check:"
echo "    python environment_check.py"
echo "==========================================="
