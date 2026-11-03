#!/usr/bin/env python3
"""
ĐO KHOẢNG CÁCH THỰC TẾ BẰNG HỆ THỐNG STEREO 2 CAMERA (check_distance_stereo.py)
---------------------------------------------------------------------------------
Phương pháp chuẩn mực duy nhất của dự án:
- Sử dụng đồng thời CẢ 2 CAMERA:
    + Camera Trái: Pi #1 (Master - camera CSI cục bộ)
    + Camera Phải: Pi #2 (Worker - nhận frame qua mạng LAN 192.168.88.2:5555)
- Phát hiện bàn cờ trên CẢ 2 KHUNG HÌNH đồng thời.
- Tự động kiểm tra và khớp nối chiều của góc bàn cờ (Corner Orientation Matching).
- Dùng thuật toán Tam giác đạc Stereo 3D (Triangulation) trên hệ tọa độ Rectified:
    + Baseline thực tế B = ||T||
    + Gốc tọa độ đo đặt chính xác tại TRUNG ĐIỂM CỦA ĐƯỜNG BASELINE (M = B/2)
- Đo đạc chính xác theo thời gian thực:
    + D_mid: Khoảng cách đường thẳng từ tâm bàn cờ đến TRUNG ĐIỂM BASELINE (cm)
    + Z: Khoảng cách vuông góc từ mặt phẳng 2 camera tới bàn cờ (cm)
    + X_mid: Vị trí lệch ngang (cm, 0: chính diện trung tâm giữa 2 camera)
- Giao diện Side-by-Side trực quan kèm bảng HUD và phát Web Stream tại http://<IP_PI>:5000.
"""

import os
import sys
import time
import argparse
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import cv2
import numpy as np

import config
from src.camera_manager import CameraManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# BỘ TÍNH TOÁN TAM GIÁC ĐẠC STEREO 3D (SELF-CONTAINED)
# ==============================================================================
class StereoTriangulator:
    """
    Nạp file stereo_calibration.npz và giải bài toán tam giác đạc 3D Stereo,
    chuyển gốc tọa độ về TRUNG ĐIỂM CỦA ĐƯỜNG BASELINE.
    """
    def __init__(self, calib_file):
        if not os.path.exists(calib_file):
            raise FileNotFoundError(f"Không tìm thấy file ma trận hiệu chuẩn tại: {calib_file}")
            
        data = np.load(calib_file)
        self.P1 = data['P1']
        self.P2 = data['P2']
        self.R1 = data['R1']
        self.R2 = data['R2']
        self.K1 = data['K1']
        self.D1 = data['D1']
        self.K2 = data['K2']
        self.D2 = data['D2']
        self.T = data['T']
        
        # Khoảng cách Baseline thực tế
        self.baseline = float(np.linalg.norm(self.T))
        self.half_baseline = self.baseline / 2.0
        
    def undistort_points_rectified(self, pts_left, pts_right):
        """Nắn thẳng và đưa tọa độ pixel 2 ảnh về hệ quy chiếu rectified song song"""
        pl = pts_left.reshape(-1, 1, 2).astype(np.float32)
        pr = pts_right.reshape(-1, 1, 2).astype(np.float32)
        
        pl_rect = cv2.undistortPoints(pl, self.K1, self.D1, R=self.R1, P=self.P1).reshape(-1, 2)
        pr_rect = cv2.undistortPoints(pr, self.K2, self.D2, R=self.R2, P=self.P2).reshape(-1, 2)
        return pl_rect, pr_rect

    def triangulate_points_batch(self, pts_left, pts_right):
        """
        Tam giác đạc cho tập hợp các cặp điểm (N, 2).
        Tự động kiểm tra và đảo chiều điểm nếu OpenCV dò góc bị ngược hướng.
        Returns:
            points_3d: Mảng (N, 3) tọa độ (X_mid, Y_mid, Z_mid) tính bằng mét
            d_mid: Mảng (N,) khoảng cách đường thẳng tới trung điểm baseline (mét)
        """
        pts_l = pts_left.reshape(-1, 2).astype(np.float32)
        pts_r = pts_right.reshape(-1, 2).astype(np.float32)
        
        pl_r, pr_r = self.undistort_points_rectified(pts_l, pts_r)
        
        # Kiểm tra chiều tương ứng của các góc cờ
        # Trong hệ rectified, các điểm tương ứng phải có y xấp xỉ nhau và x_l > x_r (disparity > 0)
        y_diff_norm = np.mean(np.abs(pl_r[:, 1] - pr_r[:, 1]))
        y_diff_flip = np.mean(np.abs(pl_r[:, 1] - pr_r[::-1, 1]))
        disp_norm = np.mean(pl_r[:, 0] - pr_r[:, 0])
        disp_flip = np.mean(pl_r[:, 0] - pr_r[::-1, 0])
        
        if y_diff_flip < y_diff_norm and disp_flip > 0:
            # Góc bàn cờ của Cam Phải bị phát hiện theo chiều ngược lại -> Đảo chiều
            pts_r = pts_r[::-1]
            pr_r = pr_r[::-1]
            
        pl_u = pl_r.T
        pr_u = pr_r.T
        
        point_4d = cv2.triangulatePoints(self.P1, self.P2, pl_u, pr_u)
        point_3d = point_4d[:3, :] / point_4d[3, :]  # (3, N)
        
        # Tọa độ trong hệ quy chiếu Rectified của Camera Trái:
        # Cam Trái ở (0,0,0), Cam Phải ở (Baseline, 0, 0)
        # Trung điểm Baseline ở (half_baseline, 0, 0)
        X_rect = point_3d[0, :]
        Y_rect = point_3d[1, :]
        Z_rect = point_3d[2, :]
        
        X_mid = X_rect - self.half_baseline
        Y_mid = Y_rect
        Z_mid = Z_rect
        
        D_mid = np.sqrt(X_mid**2 + Y_mid**2 + Z_mid**2)
        points_mid = np.vstack([X_mid, Y_mid, Z_mid]).T
        
        return points_mid, D_mid

# ==============================================================================
# WEB STREAMING SERVER
# ==============================================================================
global_frame_bytes = None
frame_lock = threading.Lock()

class WebStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Stereo 2 Camera Live Distance</title>
                <style>
                    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #fff; text-align: center; margin: 0; padding: 20px; }
                    h2 { color: #38bdf8; margin-bottom: 5px; }
                    p { color: #94a3b8; margin-top: 0; }
                    .video-container { display: inline-block; border: 3px solid #38bdf8; border-radius: 12px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.6); max-width: 95vw; }
                    img { display: block; width: 100%; height: auto; }
                    .info-card { max-width: 900px; margin: 20px auto; background: #1e293b; padding: 18px 24px; border-radius: 10px; text-align: left; font-size: 14px; line-height: 1.6; border: 1px solid #334155; }
                    .highlight { color: #facc15; font-weight: bold; }
                    .tag-left { color: #38bdf8; font-weight: bold; }
                    .tag-right { color: #f43f5e; font-weight: bold; }
                </style>
            </head>
            <body>
                <h2>🎯 ĐO KHOẢNG CÁCH THỜI GIAN THỰC BẰNG STEREO 2 CAMERA</h2>
                <p>Khung hình: <span class="tag-left">[NỬA TRÁI: CAM TRÁI PI #1]</span> | <span class="tag-right">[NỬA PHẢI: CAM PHẢI PI #2]</span></p>
                <div class="video-container">
                    <img src="/stream.mjpg" alt="Stereo 2 Camera Stream" />
                </div>
                <div class="info-card">
                    <strong>📌 Phương pháp đo Stereo 2 Camera (Tam giác đạc):</strong><br>
                    • Sử dụng đồng thời cả 2 camera (Baseline ~53 cm). Không phụ thuộc vào kích thước to nhỏ của vật thể.<br>
                    • <span class="highlight">D_mid:</span> Khoảng cách đường thẳng từ tâm bàn cờ đến <strong>TRUNG ĐIỂM CỦA ĐƯỜNG BASELINE</strong>.<br>
                    • <span class="highlight">Z:</span> Khoảng cách vuông góc từ mặt phẳng 2 camera tới bàn cờ.<br>
                    • <span class="highlight">X_mid:</span> Vị trí lệch ngang (0 cm: nằm chính giữa trục 2 camera).
                </div>
            </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            while True:
                with frame_lock:
                    if global_frame_bytes is None:
                        time.sleep(0.02)
                        continue
                    current_bytes = global_frame_bytes
                try:
                    self.wfile.write(b'--jpgboundary\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(current_bytes)))
                    self.end_headers()
                    self.wfile.write(current_bytes)
                    self.wfile.write(b'\r\n')
                    time.sleep(0.03)
                except Exception:
                    break

def start_web_server(port=5000):
    try:
        server = HTTPServer(('0.0.0.0', port), WebStreamHandler)
        logger.info(f"Web Stream Stereo kích hoạt tại: http://0.0.0.0:{port}")
        server.serve_forever()
    except Exception as e:
        logger.warning(f"Không thể mở Web Server tại cổng {port}: {e}")

def resolve_calibration_file(custom_path=None):
    """Tìm file stereo_calibration.npz chuẩn xác nhất"""
    candidates = [
        custom_path,
        os.path.join(config.DIR_CALIB_RESULT, "stereo_calibration.npz"),
        os.path.join(os.path.expanduser("~"), "hand_detection/calibrate-camera/stereo_vision/calibration_result/stereo_calibration.npz"),
        "stereo_vision/calibration_result/stereo_calibration.npz"
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError("Không tìm thấy file stereo_calibration.npz! Vui lòng chỉ định đường dẫn qua --calib_file.")

def main():
    parser = argparse.ArgumentParser(description="Đo khoảng cách thời gian thực bằng Stereo 2 Camera (Tam giác đạc)")
    parser.add_argument('--calib_file', type=str, default=None,
                        help="Đường dẫn đến file stereo_calibration.npz (mặc định tự tìm)")
    parser.add_argument('--port', type=int, default=5000,
                        help="Cổng Web Stream (mặc định: 5000)")
    args = parser.parse_args()

    # 1. Khởi tạo đối tượng StereoTriangulator
    calib_file = resolve_calibration_file(args.calib_file)
    logger.info(f"Đang nạp tham số hiệu chuẩn Stereo từ: {calib_file}")
    s3d = StereoTriangulator(calib_file=calib_file)
    logger.info(f"Baseline Stereo thực tế tính được: {s3d.baseline*100:.2f} cm (Nửa khoảng cách B/2 = {s3d.half_baseline*100:.2f} cm)")

    # 2. Khởi tạo Camera Trái (Local Pi #1)
    logger.info("Đang kết nối Camera Trái (Local Pi #1)...")
    cam_left = CameraManager(
        backend=config.CAMERA_BACKEND,
        camera_id=config.LEFT_CAMERA_ID,
        width=config.IMAGE_WIDTH,
        height=config.IMAGE_HEIGHT,
        fps=config.FPS
    )

    # 3. Khởi tạo Camera Phải (Network TCP từ Pi #2)
    cam_right_backend = 'network' if getattr(config, 'USE_NETWORK_RIGHT_CAMERA', True) else config.CAMERA_BACKEND
    r_ip = getattr(config, 'RIGHT_CAMERA_IP', '192.168.88.2')
    r_port = getattr(config, 'RIGHT_CAMERA_PORT', 5555)
    logger.info(f"Đang kết nối Camera Phải qua {cam_right_backend} ({r_ip}:{r_port})...")
    cam_right = CameraManager(
        backend=cam_right_backend,
        camera_id=config.RIGHT_CAMERA_ID,
        width=config.IMAGE_WIDTH,
        height=config.IMAGE_HEIGHT,
        fps=config.FPS,
        host=r_ip,
        port=r_port
    )

    # 4. Kích hoạt Web Server nền
    t_web = threading.Thread(target=start_web_server, args=(args.port,), daemon=True)
    t_web.start()

    logger.info("==================================================================")
    logger.info("BẮT ĐẦU ĐO KHOẢNG CÁCH STEREO 2 CAMERA")
    logger.info(f"- Bàn cờ mục tiêu: {config.CHESSBOARD_PATTERN_SIZE[0]}x{config.CHESSBOARD_PATTERN_SIZE[1]} (ô {config.SQUARE_SIZE*100:.1f} cm)")
    logger.info(f"- Xem trực tiếp: http://<IP_PI>:{args.port} hoặc xem trên màn hình Pi")
    logger.info("Nhấn 'q' hoặc Ctrl+C để dừng.")
    logger.info("==================================================================")

    pattern_size = config.CHESSBOARD_PATTERN_SIZE
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    has_display = bool(os.environ.get('DISPLAY', None))
    window_name = "Stereo 2 Camera Realtime Distance"
    if has_display:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        except Exception:
            has_display = False

    global global_frame_bytes
    frame_count = 0
    fps_display = 0.0
    start_time = time.time()

    try:
        while True:
            ret_l, frame_l = cam_left.read()
            ret_r, frame_r = cam_right.read()

            if not ret_l or frame_l is None or not ret_r or frame_r is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            if frame_count % 15 == 0:
                elapsed = time.time() - start_time
                fps_display = 15.0 / elapsed if elapsed > 0 else 0.0
                start_time = time.time()

            h, w = frame_l.shape[:2]
            gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

            # Dò tìm góc bàn cờ trên CẢ 2 CAMERA
            flags_cb = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
            found_l, corners_l = cv2.findChessboardCorners(gray_l, pattern_size, flags_cb)
            found_r, corners_r = cv2.findChessboardCorners(gray_r, pattern_size, flags_cb)

            disp_l = frame_l.copy()
            disp_r = frame_r.copy()

            dist_mid_cm = None
            dist_z_cm = None
            offset_x_cm = None
            offset_y_cm = None
            disparity_px = None

            if found_l and found_r:
                corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), subpix_criteria)
                corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), subpix_criteria)

                cv2.drawChessboardCorners(disp_l, pattern_size, corners_l, found_l)
                cv2.drawChessboardCorners(disp_r, pattern_size, corners_r, found_r)

                pts_l_flat = corners_l.reshape(-1, 2)
                pts_r_flat = corners_r.reshape(-1, 2)
                c_l = np.mean(pts_l_flat, axis=0)
                c_r = np.mean(pts_r_flat, axis=0)

                # Vẽ điểm tâm to màu vàng trên cả 2 khung hình
                cv2.circle(disp_l, (int(c_l[0]), int(c_l[1])), 8, (0, 255, 255), -1)
                cv2.circle(disp_r, (int(c_r[0]), int(c_r[1])), 8, (0, 255, 255), -1)

                # Tam giác đạc Stereo 3D theo lô (Batch Triangulation) kèm tự động kiểm tra đảo chiều
                points_3d_mid, d_mid_all = s3d.triangulate_points_batch(pts_l_flat, pts_r_flat)

                # Giá trị trung bình 3D của cả bàn cờ
                offset_x_cm = float(np.mean(points_3d_mid[:, 0])) * 100.0
                offset_y_cm = float(np.mean(points_3d_mid[:, 1])) * 100.0
                dist_z_cm = float(np.mean(points_3d_mid[:, 2])) * 100.0
                dist_mid_cm = float(np.mean(d_mid_all)) * 100.0
                disparity_px = float(c_l[0] - c_r[0])

            # Ghép 2 ảnh nằm cạnh nhau (Side-by-Side)
            scale = 0.75
            w_small = int(w * scale)
            h_small = int(h * scale)
            small_l = cv2.resize(disp_l, (w_small, h_small))
            small_r = cv2.resize(disp_r, (w_small, h_small))

            cv2.putText(small_l, "CAM TRAI (PI #1 MASTER)", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (56, 189, 248), 2)
            cv2.putText(small_r, "CAM PHAI (PI #2 WORKER)", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (244, 63, 94), 2)

            combined_feed = cv2.hconcat([small_l, small_r])
            comb_h, comb_w = combined_feed.shape[:2]

            # BẢNG HUD HIỂN THỊ KẾT QUẢ ĐO STEREO
            hud_height = 140
            hud_panel = np.zeros((hud_height, comb_w, 3), dtype=np.uint8)
            hud_panel[:] = (15, 23, 42)

            cv2.line(hud_panel, (0, 0), (comb_w, 0), (56, 189, 248), 2)
            cv2.line(hud_panel, (w_small, 0), (w_small, hud_height), (51, 65, 85), 1)

            if dist_mid_cm is not None:
                cv2.putText(hud_panel, f"KHOANG CACH TOI TRUNG DIEM (D_mid): {dist_mid_cm:.2f} cm", 
                            (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (250, 204, 21), 2)
                
                pos_str = "CHINH GIUA" if abs(offset_x_cm) < 1.5 else (f"LECH PHAI {offset_x_cm:+.1f}cm" if offset_x_cm > 0 else f"LECH TRAI {abs(offset_x_cm):.1f}cm")
                cv2.putText(hud_panel, f"Z (Vuong goc): {dist_z_cm:.2f} cm | X_mid: [{pos_str}] | Y: {offset_y_cm:+.1f} cm", 
                            (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (74, 222, 128), 2)

                cv2.putText(hud_panel, f"Giai phap: STEREO 2 CAMERA (Baseline: {s3d.baseline*100:.1f}cm) | Thi sai: {disparity_px:.1f}px | FPS: {fps_display:.1f}", 
                            (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (148, 163, 184), 1)
            else:
                cv2.putText(hud_panel, "DANG TIM BAN CO DONG THOI TREN CA 2 CAMERA...", 
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 165, 255), 2)
                status_l = "Cam Trai: OK" if found_l else "Cam Trai: CHUA THAY"
                status_r = "Cam Phai: OK" if found_r else "Cam Phai: CHUA THAY"
                cv2.putText(hud_panel, f"Trang thai: [{status_l}] - [{status_r}] | Vui long dua ban co vao vung nhin chung cua 2 cam", 
                            (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (203, 213, 225), 1)

            final_display = cv2.vconcat([combined_feed, hud_panel])

            _, jpg = cv2.imencode('.jpg', final_display, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            with frame_lock:
                global_frame_bytes = jpg.tobytes()

            if has_display:
                cv2.imshow(window_name, final_display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break

    except KeyboardInterrupt:
        logger.info("Dừng chương trình.")
    finally:
        cam_left.close()
        cam_right.close()
        if has_display:
            cv2.destroyAllWindows()
        logger.info("Hoàn tất giải phóng 2 camera.")

if __name__ == '__main__':
    main()
