import cv2
import numpy as np
import glob
import os
import csv
import logging
from datetime import datetime

import config
from src.chessboard_detector import find_chessboard, draw_chessboard

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def calibrate_single_camera(objpoints, imgpoints, image_size, cam_name):
    """Calibrates a single camera and returns intrinsic matrix, distortion, and RMS."""
    logger.info(f"--- Intrinsic Calibration for {cam_name} ---")
    
    flags = 0
    if config.USE_RATIONAL_MODEL:
        flags |= cv2.CALIB_RATIONAL_MODEL
        
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None, flags=flags
    )
    
    logger.info(f"{cam_name} RMS Error: {ret:.4f}")
    logger.info(f"{cam_name} Matrix K:\n{K}")
    logger.info(f"{cam_name} Distortion D:\n{D.ravel()}")
    
    # Calculate per-view reprojection error to identify outliers
    per_view_errors = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        pts_orig = imgpoints[i].reshape(-1, 2).astype(np.float32)
        pts_proj = imgpoints2.reshape(-1, 2).astype(np.float32)
        error = float(np.mean(np.linalg.norm(pts_orig - pts_proj, axis=1)))
        per_view_errors.append(error)
        
    mean_error = np.mean(per_view_errors)
    logger.info(f"{cam_name} Mean Reprojection Error: {mean_error:.4f} pixels\n")
    
    return ret, K, D, per_view_errors

def main():
    logger.info("Starting Stereo Calibration Process")
    
    os.makedirs(config.DIR_CALIB_RESULT, exist_ok=True)
    os.makedirs(config.DIR_DEBUG, exist_ok=True)

    left_images = sorted(glob.glob(os.path.join(config.DIR_LEFT_IMAGES, '*.png')))
    right_images = sorted(glob.glob(os.path.join(config.DIR_RIGHT_IMAGES, '*.png')))

    if len(left_images) != len(right_images) or len(left_images) == 0:
        logger.error(f"Mismatch or zero images! Left: {len(left_images)}, Right: {len(right_images)}")
        return

    pattern_size = config.CHESSBOARD_PATTERN_SIZE
    square_size = config.SQUARE_SIZE

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = [] # 2d points in image plane.
    
    valid_pairs = []
    report_data = []

    image_size = None

    logger.info("Detecting chessboards...")
    for l_path, r_path in zip(left_images, right_images):
        pair_id = os.path.basename(l_path).split('_')[1].split('.')[0]
        
        img_l = cv2.imread(l_path)
        img_r = cv2.imread(r_path)
        
        if image_size is None:
            image_size = (img_l.shape[1], img_l.shape[0])
            
        if (img_l.shape[1], img_l.shape[0]) != image_size or (img_r.shape[1], img_r.shape[0]) != image_size:
            logger.error(f"Resolution mismatch in pair {pair_id}!")
            continue

        found_l, corners_l, gray_l = find_chessboard(img_l, pattern_size)
        found_r, corners_r, gray_r = find_chessboard(img_r, pattern_size)
        
        if found_l and found_r:
            objpoints.append(objp)
            imgpoints_left.append(corners_l)
            imgpoints_right.append(corners_r)
            valid_pairs.append(pair_id)
            
            # Save debug images for visual inspection
            dbg_l = draw_chessboard(img_l, pattern_size, corners_l, found_l)
            dbg_r = draw_chessboard(img_r, pattern_size, corners_r, found_r)
            cv2.imwrite(os.path.join(config.DIR_DEBUG, f"left_{pair_id}_detected.png"), dbg_l)
            cv2.imwrite(os.path.join(config.DIR_DEBUG, f"right_{pair_id}_detected.png"), dbg_r)
            
            report_data.append({
                'pair_id': pair_id, 'left_image': os.path.basename(l_path), 'right_image': os.path.basename(r_path),
                'left_corner_found': True, 'right_corner_found': True, 'valid': True, 'reason': ''
            })
        else:
            reason = []
            if not found_l: reason.append("Left missing")
            if not found_r: reason.append("Right missing")
            report_data.append({
                'pair_id': pair_id, 'left_image': os.path.basename(l_path), 'right_image': os.path.basename(r_path),
                'left_corner_found': found_l, 'right_corner_found': found_r, 'valid': False, 'reason': " & ".join(reason)
            })

    if len(valid_pairs) < 10:
        logger.error(f"Not enough valid pairs for calibration! Found only {len(valid_pairs)}.")
        return
        
    logger.info(f"Found {len(valid_pairs)} valid pairs out of {len(left_images)}")

    # Intrinsic Calibration
    rms_l, K1, D1, errs_l = calibrate_single_camera(objpoints, imgpoints_left, image_size, "LEFT")
    rms_r, K2, D2, errs_r = calibrate_single_camera(objpoints, imgpoints_right, image_size, "RIGHT")

    # Update report with reprojection errors
    med_l = float(np.median(errs_l))
    med_r = float(np.median(errs_r))
    thresh_l = max(med_l * 1.8, 4.0)
    thresh_r = max(med_r * 1.8, 4.0)
    logger.info(f"Outlier rejection thresholds: LEFT > {thresh_l:.2f} px, RIGHT > {thresh_r:.2f} px")

    idx = 0
    for row in report_data:
        if row['valid']:
            row['left_reprojection_error'] = f"{errs_l[idx]:.4f}"
            row['right_reprojection_error'] = f"{errs_r[idx]:.4f}"
            
            # Simple outlier rejection logic (flagging)
            if errs_l[idx] > thresh_l or errs_r[idx] > thresh_r:
                row['valid'] = False
                row['reason'] = f"High reprojection error (L:{errs_l[idx]:.2f}, R:{errs_r[idx]:.2f})"
            idx += 1
        else:
            row['left_reprojection_error'] = ""
            row['right_reprojection_error'] = ""

    # Filter out outliers for stereo calibration
    stereo_objpoints = []
    stereo_imgpoints_left = []
    stereo_imgpoints_right = []
    stereo_pairs = []
    
    idx = 0
    for row in report_data:
        if row['valid'] and row['left_reprojection_error'] != "":
            stereo_objpoints.append(objpoints[idx])
            stereo_imgpoints_left.append(imgpoints_left[idx])
            stereo_imgpoints_right.append(imgpoints_right[idx])
            stereo_pairs.append(row['pair_id'])
            idx += 1
        elif row['left_reprojection_error'] != "":
            # Was originally valid, but flagged as outlier
            idx += 1

    if len(stereo_pairs) < 10:
        logger.warning(f"Only {len(stereo_pairs)} pairs passed outlier filter. Falling back to all {len(valid_pairs)} valid pairs.")
        stereo_objpoints = list(objpoints)
        stereo_imgpoints_left = list(imgpoints_left)
        stereo_imgpoints_right = list(imgpoints_right)
        stereo_pairs = list(valid_pairs)

    logger.info(f"Using {len(stereo_pairs)} pairs for stereo calibration (after outlier rejection).")

    # Stereo Calibration
    logger.info("--- Stereo Calibration ---")
    flags_stereo = cv2.CALIB_FIX_INTRINSIC
    
    # termination criteria
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    ret_stereo, K1_s, D1_s, K2_s, D2_s, R, T, E, F = cv2.stereoCalibrate(
        stereo_objpoints, stereo_imgpoints_left, stereo_imgpoints_right,
        K1, D1, K2, D2, image_size, criteria=criteria_stereo, flags=flags_stereo
    )

    logger.info(f"Stereo RMS Error: {ret_stereo:.4f}")
    logger.info(f"Rotation Matrix R (Right w.r.t Left):\n{R}")
    logger.info(f"Translation Vector T (Right w.r.t Left):\n{T.ravel()}")
    
    # Baseline check
    baseline = np.linalg.norm(T)
    logger.info(f"Measured baseline (EXPECTED_BASELINE): {config.EXPECTED_BASELINE:.4f} m")
    logger.info(f"Estimated baseline (norm(T)):          {baseline:.4f} m")
    
    diff = abs(baseline - config.EXPECTED_BASELINE)
    percent_error = (diff / config.EXPECTED_BASELINE) * 100
    logger.info(f"Absolute error: {diff:.4f} m")
    logger.info(f"Percentage error: {percent_error:.2f}%")
    
    if percent_error > config.BASELINE_WARNING_THRESHOLD_PERCENT:
        logger.warning("****************************************************************")
        logger.warning("WARNING: Estimated stereo baseline differs significantly from measured baseline.")
        logger.warning("Check your SQUARE_SIZE, pattern size, or re-capture images.")
        logger.warning("****************************************************************")

    # Stereo Rectification
    logger.info("--- Stereo Rectification ---")
    R1, R2, P1, P2, Q, ROI1, ROI2 = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T, alpha=0 # alpha=0 crops invalid pixels
    )
    
    logger.info("Computing Rectification Maps...")
    map1_left, map2_left = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_16SC2)
    map1_right, map2_right = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_16SC2)

    # Save outputs
    logger.info("Saving calibration parameters...")
    
    # Save NPZ
    np.savez(os.path.join(config.DIR_CALIB_RESULT, "stereo_calibration.npz"),
             K1=K1, D1=D1, K2=K2, D2=D2, 
             R=R, T=T, E=E, F=F, 
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             ROI1=ROI1, ROI2=ROI2, image_size=image_size)
             
    # Save Rectification Maps
    np.savez(os.path.join(config.DIR_CALIB_RESULT, "rectification_maps.npz"),
             map1_left=map1_left, map2_left=map2_left,
             map1_right=map1_right, map2_right=map2_right)
             
    # Save YAML
    fs = cv2.FileStorage(os.path.join(config.DIR_CALIB_RESULT, "stereo_calibration.yml"), cv2.FILE_STORAGE_WRITE)
    fs.write("K1", K1)
    fs.write("D1", D1)
    fs.write("K2", K2)
    fs.write("D2", D2)
    fs.write("R", R)
    fs.write("T", T)
    fs.write("E", E)
    fs.write("F", F)
    fs.write("R1", R1)
    fs.write("R2", R2)
    fs.write("P1", P1)
    fs.write("P2", P2)
    fs.write("Q", Q)
    fs.release()
    
    # Save TXT Report
    with open(os.path.join(config.DIR_CALIB_RESULT, "calibration_report.txt"), "w") as f:
        f.write(f"Stereo Calibration Report\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"OpenCV Version: {cv2.__version__}\n")
        f.write(f"Image Resolution: {image_size}\n")
        f.write(f"Pattern Size (Inner Corners): {pattern_size}\n")
        f.write(f"Square Size: {square_size} m\n")
        f.write("\n--- LEFT CAMERA ---\n")
        f.write(f"K:\n{K1}\n")
        f.write(f"D:\n{D1}\n")
        f.write(f"RMS: {rms_l:.4f}\n")
        
        f.write("\n--- RIGHT CAMERA ---\n")
        f.write(f"K:\n{K2}\n")
        f.write(f"D:\n{D2}\n")
        f.write(f"RMS: {rms_r:.4f}\n")
        
        f.write("\n--- STEREO EXTRINSICS ---\n")
        f.write(f"R (Right w.r.t Left):\n{R}\n")
        f.write(f"T (Right w.r.t Left):\n{T}\n")
        f.write(f"E:\n{E}\n")
        f.write(f"F:\n{F}\n")
        f.write(f"Stereo RMS: {ret_stereo:.4f}\n")
        f.write(f"Baseline (Estimated): {baseline:.4f} m\n")
        
        f.write("\n--- BASELINE CHECK ---\n")
        f.write(f"Measured (EXPECTED): {config.EXPECTED_BASELINE} m\n")
        f.write(f"Difference: {diff:.4f} m\n")
        f.write(f"Percentage Error: {percent_error:.2f}%\n")
        
        f.write("\n--- RECTIFICATION MATRICES ---\n")
        f.write(f"R1:\n{R1}\n")
        f.write(f"R2:\n{R2}\n")
        f.write(f"P1:\n{P1}\n")
        f.write(f"P2:\n{P2}\n")
        f.write(f"Q (Disparity-to-Depth):\n{Q}\n")

    # Save CSV Report
    with open(os.path.join(config.DIR_CALIB_RESULT, "calibration_report.csv"), "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'pair_id', 'left_image', 'right_image', 
            'left_corner_found', 'right_corner_found', 
            'left_reprojection_error', 'right_reprojection_error', 
            'valid', 'reason'
        ])
        writer.writeheader()
        writer.writerows(report_data)

    logger.info("Calibration complete. All files saved to calibration_result/.")

if __name__ == "__main__":
    main()
