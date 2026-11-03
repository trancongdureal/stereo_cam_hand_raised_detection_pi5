import os
import sys
import platform
import subprocess

def check_package(package_name):
    try:
        if package_name == 'cv2':
            import cv2
            return True, cv2.__version__
        elif package_name == 'numpy':
            import numpy
            return True, numpy.__version__
        elif package_name == 'picamera2':
            import picamera2
            return True, getattr(picamera2, '__version__', 'Installed')
        else:
            return False, "Unknown package"
    except ImportError as e:
        return False, str(e)

def main():
    print("===========================================")
    print("       ENVIRONMENT CHECK                   ")
    print("===========================================")
    
    print(f"OS:             {platform.system()} {platform.release()}")
    print(f"Architecture:   {platform.machine()}")
    print(f"Python:         {platform.python_version()} ({sys.executable})")
    
    # Check pip version safely
    try:
        pip_ver = subprocess.check_output([sys.executable, "-m", "pip", "--version"]).decode().split()[1]
        print(f"pip:            {pip_ver}")
    except:
        print("pip:            FAIL")

    errors = []

    # OpenCV
    ok, ver = check_package('cv2')
    print(f"OpenCV:         {ver if ok else 'FAIL'}")
    if not ok:
        errors.append(("OpenCV", ver, "pip install opencv-python"))

    # NumPy
    ok, ver = check_package('numpy')
    print(f"NumPy:          {ver if ok else 'FAIL'}")
    if not ok:
        errors.append(("NumPy", ver, "pip install numpy"))

    # Picamera2 (Optional, mainly for Pi)
    if platform.system() == "Linux" and ("arm" in platform.machine() or "aarch" in platform.machine()):
        ok, ver = check_package('picamera2')
        print(f"Picamera2:      {ver if ok else 'FAIL'}")
        if not ok:
            errors.append(("Picamera2", ver, "Run: sudo apt install python3-picamera2 (And use --system-site-packages in venv)"))
    else:
        print("Picamera2:      N/A (Not Raspberry Pi)")

    # Read config
    try:
        import config
        print(f"Camera backend: {config.CAMERA_BACKEND}")
    except Exception as e:
        print(f"Config file:    FAIL")
        errors.append(("config.py", str(e), "Ensure config.py exists and is syntax-error free"))

    print("\n===========================================")
    if len(errors) == 0:
        print("ENVIRONMENT STATUS: PASS")
    else:
        print("ENVIRONMENT STATUS: FAIL")
        print("-------------------------------------------")
        for pkg, err, fix in errors:
            print(f"PACKAGE: {pkg}")
            print(f"ERROR:   {err}")
            print(f"FIX:     {fix}")
            print("-")
    print("===========================================")

if __name__ == "__main__":
    main()
