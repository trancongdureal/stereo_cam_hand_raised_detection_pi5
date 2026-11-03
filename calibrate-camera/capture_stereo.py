import cv2
import os
import sys
import time
import csv
import logging
from datetime import datetime

import config
from src.camera_manager import CameraManager
from src.chessboard_detector import find_chessboard, draw_chessboard

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def draw_overlay(frame, text_lines, is_error=False):
    """Draws text on the image frame."""
    color = (0, 0, 255) if is_error else (0, 255, 0)
    y0, dy = 30, 30
    for i, line in enumerate(text_lines):
        y = y0 + i * dy
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def main():
    logger.info("Starting Stereo Capture Tool")

    # Initialize cameras
    cam_left = CameraManager(
        backend=config.CAMERA_BACKEND,
        camera_id=config.LEFT_CAMERA_ID,
        width=config.IMAGE_WIDTH,
        height=config.IMAGE_HEIGHT,
        fps=config.FPS
    )
    
    cam_right_backend = 'network' if getattr(config, 'USE_NETWORK_RIGHT_CAMERA', False) else config.CAMERA_BACKEND
    cam_right = CameraManager(
        backend=cam_right_backend,
        camera_id=config.RIGHT_CAMERA_ID,
        width=config.IMAGE_WIDTH,
        height=config.IMAGE_HEIGHT,
        fps=config.FPS,
        host=getattr(config, 'RIGHT_CAMERA_IP', '172.20.10.3'),
        port=getattr(config, 'RIGHT_CAMERA_PORT', 5555)
    )

    pair_id = 0
    auto_mode = False
    last_auto_capture_time = 0

    # Ensure directories exist
    os.makedirs(config.DIR_LEFT_IMAGES, exist_ok=True)
    os.makedirs(config.DIR_RIGHT_IMAGES, exist_ok=True)

    csv_path = os.path.join(config.DIR_CALIB_IMAGES, "pairs.csv")
    file_exists = os.path.isfile(csv_path)
    csv_file = open(csv_path, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    if not file_exists:
        csv_writer.writerow(['pair_id', 'left_filename', 'right_filename', 'timestamp', 'width', 'height', 'capture_mode'])

    # Find next pair ID
    existing_left = [f for f in os.listdir(config.DIR_LEFT_IMAGES) if f.endswith('.png')]
    existing_right = [f for f in os.listdir(config.DIR_RIGHT_IMAGES) if f.endswith('.png')]
    
    # Simple logic to continue counter
    if existing_left:
        latest_id = max([int(f.split('_')[1].split('.')[0]) for f in existing_left if '_' in f])
        pair_id = latest_id

    cv2.namedWindow("Stereo Calibration Capture", cv2.WINDOW_NORMAL)
    
    logger.info("Controls:")
    logger.info("  SPACE or S : Capture a pair manually")
    logger.info("  A          : Toggle AUTO capture mode")
    logger.info("  R          : Reset counter to 0")
    logger.info("  Q or ESC   : Quit")

    try:
        while True:
            # Capture as close as possible in software
            ret_l, frame_l = cam_left.read()
            ret_r, frame_r = cam_right.read()
            capture_time = time.time()

            disp_l = frame_l.copy() if ret_l else None
            disp_r = frame_r.copy() if ret_r else None

            err_msg_l = "" if ret_l else "LEFT CAMERA ERROR: NO FRAME"
            err_msg_r = "" if ret_r else "RIGHT CAMERA ERROR: NO FRAME"

            both_valid = ret_l and ret_r

            if both_valid:
                # Check resolution mismatch
                h_l, w_l = frame_l.shape[:2]
                h_r, w_r = frame_r.shape[:2]
                if w_l != config.IMAGE_WIDTH or h_l != config.IMAGE_HEIGHT:
                    err_msg_l = f"RES MISMATCH L: {w_l}x{h_l}"
                    both_valid = False
                if w_r != config.IMAGE_WIDTH or h_r != config.IMAGE_HEIGHT:
                    err_msg_r = f"RES MISMATCH R: {w_r}x{h_r}"
                    both_valid = False

            # Auto mode logic
            do_capture = False
            mode_used = "MANUAL"
            if auto_mode and both_valid:
                if (capture_time - last_auto_capture_time) >= config.AUTO_CAPTURE_INTERVAL:
                    if config.REQUIRE_CHESSBOARD_FOR_CAPTURE:
                        found_l, corners_l, _ = find_chessboard(frame_l, config.CHESSBOARD_PATTERN_SIZE)
                        found_r, corners_r, _ = find_chessboard(frame_r, config.CHESSBOARD_PATTERN_SIZE)
                        if found_l and found_r:
                            do_capture = True
                            mode_used = "AUTO"
                            # Draw for preview feedback
                            disp_l = draw_chessboard(disp_l, config.CHESSBOARD_PATTERN_SIZE, corners_l, found_l)
                            disp_r = draw_chessboard(disp_r, config.CHESSBOARD_PATTERN_SIZE, corners_r, found_r)
                        else:
                            # Not found in both, don't capture yet
                            pass
                    else:
                        do_capture = True
                        mode_used = "AUTO"

            # Handle UI and Keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27: # Q or ESC
                break
            elif key == ord('s') or key == 32: # S or SPACE
                if both_valid:
                    do_capture = True
                    mode_used = "MANUAL"
                else:
                    logger.warning("Cannot capture manually: one or both frames invalid.")
            elif key == ord('a'): # A
                auto_mode = not auto_mode
                logger.info(f"Auto Mode: {'ON' if auto_mode else 'OFF'}")
            elif key == ord('r'): # R
                pair_id = 0
                logger.info("Pair ID counter reset to 0.")

            # Perform Capture
            if do_capture and both_valid:
                pair_id += 1
                fname_l = f"left_{pair_id:04d}.png"
                fname_r = f"right_{pair_id:04d}.png"
                
                path_l = os.path.join(config.DIR_LEFT_IMAGES, fname_l)
                path_r = os.path.join(config.DIR_RIGHT_IMAGES, fname_r)
                
                # Save lossless PNG
                cv2.imwrite(path_l, frame_l, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                cv2.imwrite(path_r, frame_r, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                
                timestamp_str = datetime.now().isoformat()
                csv_writer.writerow([pair_id, fname_l, fname_r, timestamp_str, config.IMAGE_WIDTH, config.IMAGE_HEIGHT, mode_used])
                csv_file.flush()
                
                last_auto_capture_time = time.time()
                logger.info(f"Captured Pair {pair_id:04d} ({mode_used})")
                
                # Visual flash
                if disp_l is not None:
                    cv2.rectangle(disp_l, (0,0), (config.IMAGE_WIDTH-1, config.IMAGE_HEIGHT-1), (0,255,0), 10)
                if disp_r is not None:
                    cv2.rectangle(disp_r, (0,0), (config.IMAGE_WIDTH-1, config.IMAGE_HEIGHT-1), (0,255,0), 10)

            # Draw HUD
            if disp_l is not None:
                lines_l = ["LEFT CAMERA", f"Res: {config.IMAGE_WIDTH}x{config.IMAGE_HEIGHT}", f"Pairs: {pair_id}"]
                if err_msg_l: lines_l.append(err_msg_l)
                draw_overlay(disp_l, lines_l, is_error=bool(err_msg_l))
                
            if disp_r is not None:
                lines_r = ["RIGHT CAMERA", f"Mode: {'AUTO' if auto_mode else 'MANUAL'}"]
                if err_msg_r: lines_r.append(err_msg_r)
                draw_overlay(disp_r, lines_r, is_error=bool(err_msg_r))

            # Combine side-by-side
            if disp_l is not None and disp_r is not None:
                combined = cv2.hconcat([disp_l, disp_r])
            elif disp_l is not None:
                # Mock right side
                mock_r = disp_l.copy()
                mock_r[:] = (0,0,255) # Red screen of death
                draw_overlay(mock_r, [err_msg_r], True)
                combined = cv2.hconcat([disp_l, mock_r])
            elif disp_r is not None:
                # Mock left side
                mock_l = disp_r.copy()
                mock_l[:] = (0,0,255)
                draw_overlay(mock_l, [err_msg_l], True)
                combined = cv2.hconcat([mock_l, disp_r])
            else:
                combined = None
                
            if combined is not None:
                # Resize for display if too large (display only, doesn't affect saved image)
                disp_w = 1280
                scale = disp_w / combined.shape[1]
                disp_h = int(combined.shape[0] * scale)
                combined_disp = cv2.resize(combined, (disp_w, disp_h))
                cv2.imshow("Stereo Calibration Capture", combined_disp)
            else:
                logger.error("Both cameras failed to provide frames. Exiting.")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cam_left.close()
        cam_right.close()
        csv_file.close()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    main()
