import cv2
import config
from src.camera_manager import CameraManager

def draw_hud(frame, text):
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

def main():
    print("===========================================")
    print("             CAMERA TEST                   ")
    print("===========================================")
    print(f"Backend: {config.CAMERA_BACKEND}")
    print(f"Target Resolution: {config.IMAGE_WIDTH}x{config.IMAGE_HEIGHT}")
    print("Press 'Q' or 'ESC' to exit.")
    print("Verify that the LEFT camera physically corresponds to the LEFT image.")

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

    cv2.namedWindow("Camera Test", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret_l, frame_l = cam_left.read()
            ret_r, frame_r = cam_right.read()

            disp_l = frame_l.copy() if ret_l else None
            disp_r = frame_r.copy() if ret_r else None

            if disp_l is not None:
                draw_hud(disp_l, f"LEFT CAM (ID: {config.LEFT_CAMERA_ID})")
            if disp_r is not None:
                draw_hud(disp_r, f"RIGHT CAM (ID: {config.RIGHT_CAMERA_ID})")

            # Combine or handle missing frames
            if disp_l is not None and disp_r is not None:
                combined = cv2.hconcat([disp_l, disp_r])
            elif disp_l is not None:
                mock = disp_l.copy()
                mock[:] = (0,0,255)
                draw_hud(mock, "RIGHT CAM: NO FRAME")
                combined = cv2.hconcat([disp_l, mock])
            elif disp_r is not None:
                mock = disp_r.copy()
                mock[:] = (0,0,255)
                draw_hud(mock, "LEFT CAM: NO FRAME")
                combined = cv2.hconcat([mock, disp_r])
            else:
                print("Both cameras failed to provide frames.")
                break

            # Scale down for display if too large
            disp_w = 1280
            scale = disp_w / combined.shape[1]
            disp_h = int(combined.shape[0] * scale)
            combined_disp = cv2.resize(combined, (disp_w, disp_h))
            
            cv2.imshow("Camera Test", combined_disp)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        cam_left.close()
        cam_right.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
