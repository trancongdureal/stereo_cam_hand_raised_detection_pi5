import os
import glob
import numpy as np
import cv2
import logging
import config

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def draw_horizontal_lines(img, line_gap=50, color=(0, 255, 0)):
    """Draws horizontal lines on the image for epipolar alignment visual check."""
    out = img.copy()
    h, w = out.shape[:2]
    for y in range(0, h, line_gap):
        cv2.line(out, (0, y), (w, y), color, 1, cv2.LINE_AA)
    return out

def main():
    print("===========================================")
    print("         CALIBRATION VALIDATION            ")
    print("===========================================")
    
    calib_file = os.path.join(config.DIR_CALIB_RESULT, "stereo_calibration.npz")
    maps_file = os.path.join(config.DIR_CALIB_RESULT, "rectification_maps.npz")
    
    if not os.path.exists(calib_file):
        print("STATUS: FAIL")
        print(f"Reason: Calibration file not found at {calib_file}")
        return

    # 1. EVALUATE CALIBRATION QUALITY
    data = np.load(calib_file)
    
    T = data['T']
    baseline = np.linalg.norm(T)
    expected = config.EXPECTED_BASELINE
    diff = abs(baseline - expected)
    percent_error = (diff / expected) * 100 if expected > 0 else 0
    
    print(f"UNIT SYSTEM:          meter (based on SQUARE_SIZE={config.SQUARE_SIZE})")
    print(f"Measured baseline:    {expected:.4f} m")
    print(f"Estimated baseline:   {baseline:.4f} m")
    print(f"Difference:           {diff:.4f} m")
    print(f"Percentage:           {percent_error:.2f}%")
    
    status = "PASS"
    warnings = []
    
    if percent_error > config.BASELINE_WARNING_THRESHOLD_PERCENT:
        status = "WARNING"
        warnings.append(f"Baseline difference ({percent_error:.2f}%) exceeds threshold ({config.BASELINE_WARNING_THRESHOLD_PERCENT}%).")
        
    if percent_error > 15.0:
        status = "FAIL"
        warnings.append("Baseline difference is extreme (>15%). Calibration is likely completely wrong.")
        
    report_file = os.path.join(config.DIR_CALIB_RESULT, "calibration_report.txt")
    if os.path.exists(report_file):
        with open(report_file, "r") as f:
            content = f.read()
            for line in content.split("\n"):
                if "Stereo RMS" in line:
                    rms = float(line.split(":")[1].strip())
                    print(f"Stereo RMS:           {rms:.4f}")
                    if rms > 1.0:
                        status = "WARNING" if status == "PASS" else status
                        warnings.append(f"Stereo RMS ({rms:.4f}) is high. Aim for < 0.5")
                    if rms > 2.0:
                        status = "FAIL"
                        warnings.append(f"Stereo RMS ({rms:.4f}) is very high. Calibration failed geometrically.")

    print("\nCALIBRATION STATUS:", status)
    if warnings:
        for w in warnings:
            print(f" - {w}")

    # 2. EVALUATE RECTIFICATION
    print("\n--- Testing Rectification ---")
    if not os.path.exists(maps_file):
        print("STATUS: FAIL (Maps not found)")
        return
        
    maps_data = np.load(maps_file)
    map1_left, map2_left = maps_data['map1_left'], maps_data['map2_left']
    map1_right, map2_right = maps_data['map1_right'], maps_data['map2_right']

    left_images = sorted(glob.glob(os.path.join(config.DIR_LEFT_IMAGES, '*.png')))
    right_images = sorted(glob.glob(os.path.join(config.DIR_RIGHT_IMAGES, '*.png')))

    if not left_images or not right_images:
        print("STATUS: SKIPPED (No test images found)")
        return

    test_l_path = left_images[0]
    test_r_path = right_images[0]
    
    print(f"Using {os.path.basename(test_l_path)} and {os.path.basename(test_r_path)} for rectification test.")

    img_l = cv2.imread(test_l_path)
    img_r = cv2.imread(test_r_path)

    if img_l is None or img_r is None:
        print("STATUS: FAIL (Could not read images)")
        return

    rectified_l = cv2.remap(img_l, map1_left, map2_left, cv2.INTER_LINEAR)
    rectified_r = cv2.remap(img_r, map1_right, map2_right, cv2.INTER_LINEAR)

    combined = cv2.hconcat([rectified_l, rectified_r])
    combined_with_lines = draw_horizontal_lines(combined, line_gap=40, color=(0, 255, 0))

    out_path = os.path.join(config.DIR_CALIB_RESULT, "rectification_test.png")
    cv2.imwrite(out_path, combined_with_lines)
    print(f"Saved rectification test image to: {out_path}")
    print("-> Please inspect this image manually. Features should lie on the exact same horizontal green lines.")

if __name__ == "__main__":
    main()
