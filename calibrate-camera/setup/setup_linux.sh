#!/bin/bash
# Setup script for generic Linux (Ubuntu/Debian)

echo "==========================================="
echo "   Stereo Vision - Linux Setup Script      "
echo "==========================================="

echo "[1/4] Checking Python 3..."
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install it first."
    exit 1
fi

echo "[2/4] Creating virtual environment in .venv..."
if [ ! -d "../.venv" ]; then
    python3 -m venv ../.venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment. Do you have python3-venv installed?"
        echo "Try: sudo apt install python3-venv"
        exit 1
    fi
else
    echo "Virtual environment already exists."
fi

echo "[3/4] Activating virtual environment & installing dependencies..."
source ../.venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r ../requirements.txt

echo "[4/4] Creating necessary directories..."
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
