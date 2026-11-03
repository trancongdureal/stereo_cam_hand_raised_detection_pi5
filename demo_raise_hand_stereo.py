#!/usr/bin/env python3
"""
DEMO TEST LUỒNG HOẠT ĐỘNG: KẾT HỢP NHẬN DIỆN NGƯỜI GIƠ TAY & STEREO 3D
(demo_raise_hand_stereo.py)
---------------------------------------------------------------------------------
Yêu cầu kỹ thuật:
1. Tính khoảng cách từ TÂM CỦA BOUNDING BOX người đến TRUNG ĐIỂM BASELINE.
2. Tự động nạp file hiệu chuẩn stereo_calibration.npz có sẵn trên Pi #1:
     - ~/hand_detection/calibrate-camera/stereo_vision/calibration_result/stereo_calibration.npz
     - stereo_vision/calibration_result/stereo_calibration.npz
   (Người dùng chỉ cần nạp duy nhất file code này sang Pi #1, không cần chép thêm thư mục calibrate).
3. Nhận diện người & phát hiện tư thế giơ tay (YOLO Pose NCNN / PyTorch):
   - Đổi viền sang Đỏ và nhãn 'HAND RAISED' khi cổ tay cao hơn vai (conf > 0.5).
4. Đo đạc Stereo 3D theo thời gian thực:
   - D_mid: Khoảng cách đường thẳng từ tâm bounding box đến trung điểm baseline (cm).
   - Z: Khoảng cách vuông góc từ mặt phẳng 2 camera tới người (cm).
   - X_mid: Vị trí lệch ngang so với tâm giữa 2 camera (-: lệch trái, +: lệch phải).
5. Giao diện:
   - Ghép Side-by-Side (Cam Trái | Cam Phải).
   - Bảng HUD thời gian thực.
   - Phát Web Stream tại http://<IP_PI>:5000.
"""

import os
import sys
import time
import socket
import struct
import argparse
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Thư viện ultralytics chưa được cài đặt. Chạy: pip install ultralytics")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("StereoPoseDemo")

# ==============================================================================
# 1. ĐỊNH NGHĨA KHỚP XƯƠNG & QUY TẮC GIƠ TAY
# ==============================================================================
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_WRIST, RIGHT_WRIST = 9, 10

SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16)
]

def check_hand_raise(keypoints, conf_thresh=0.5):
    """
    Kiểm tra điều kiện giơ tay từ 17 keypoints.
    Returns:
        is_raised (bool): Có giơ tay hay không
        side (str): 'LEFT', 'RIGHT', 'BOTH' hoặc None
        wrist_idx (int): 9 hoặc 10
    """
    left_raised = False
    right_raised = False

    # Tay trái giơ lên (Wrist y < Shoulder y trong hệ tọa độ ảnh)
    if keypoints[LEFT_WRIST][2] > conf_thresh and keypoints[LEFT_SHOULDER][2] > conf_thresh:
        if keypoints[LEFT_WRIST][1] < keypoints[LEFT_SHOULDER][1]:
            left_raised = True

    # Tay phải giơ lên
    if keypoints[RIGHT_WRIST][2] > conf_thresh and keypoints[RIGHT_SHOULDER][2] > conf_thresh:
        if keypoints[RIGHT_WRIST][1] < keypoints[RIGHT_SHOULDER][1]:
            right_raised = True

    if left_raised and right_raised:
        wrist_idx = LEFT_WRIST if keypoints[LEFT_WRIST][1] < keypoints[RIGHT_WRIST][1] else RIGHT_WRIST
        return True, "BOTH", wrist_idx
    elif left_raised:
        return True, "LEFT", LEFT_WRIST
    elif right_raised:
        return True, "RIGHT", RIGHT_WRIST
    else:
        return False, None, None


# ==============================================================================
# 2. BỘ NẠP & TÍNH TOÁN HIỆU CHUẨN STEREO 3D
# ==============================================================================
class StereoTriangulator:
    """
    Nạp ma trận hiệu chuẩn và giải tam giác đạc 3D.
    Gốc tọa độ được dời về TRUNG ĐIỂM ĐOẠN BASELINE M = (B/2, 0, 0).
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
        
        # Chiều dài baseline thực tế B = ||T||
        self.baseline = float(np.linalg.norm(self.T))
        self.half_baseline = self.baseline / 2.0
        logger.info(f"Đã nạp file hiệu chuẩn thành công. Baseline: {self.baseline*100:.2f} cm (B/2 = {self.half_baseline*100:.2f} cm)")

    def triangulate_single_point(self, pt_left, pt_right):
        """
        Tam giác đạc cho 1 cặp điểm tương ứng (u, v) trên Cam Trái và Cam Phải.
        Returns:
            X_mid, Y_mid, Z_mid (mét)
            D_mid (mét, khoảng cách thẳng tới trung điểm baseline)
        """
        pl = np.array(pt_left, dtype=np.float32).reshape(-1, 1, 2)
        pr = np.array(pt_right, dtype=np.float32).reshape(-1, 1, 2)

        pl_rect = cv2.undistortPoints(pl, self.K1, self.D1, R=self.R1, P=self.P1).reshape(-1, 2)
        pr_rect = cv2.undistortPoints(pr, self.K2, self.D2, R=self.R2, P=self.P2).reshape(-1, 2)

        pt_4d = cv2.triangulatePoints(self.P1, self.P2, pl_rect.T, pr_rect.T)
        pt_3d = pt_4d[:3, :] / pt_4d[3, :]

        X_rect = float(pt_3d[0, 0])
        Y_rect = float(pt_3d[1, 0])
        Z_rect = float(pt_3d[2, 0])

        # Chuyển gốc tọa độ về TRUNG ĐIỂM BASELINE: M = (half_baseline, 0, 0)
        X_mid = X_rect - self.half_baseline
        Y_mid = Y_rect
        Z_mid = Z_rect

        # Khoảng cách đường thẳng Euclidean từ điểm đến trung điểm baseline
        D_mid = float(np.sqrt(X_mid**2 + Y_mid**2 + Z_mid**2))
        return X_mid, Y_mid, Z_mid, D_mid

    def match_epipolar_patch(self, frame_left, frame_right, pt_left, patch_size=35, search_width=350):
        """
        Fallback dò điểm tương ứng trên Cam Phải dọc theo đường epipolar (nếu Cam Phải không bắt được bbox).
        """
        h, w = frame_left.shape[:2]
        x_l, y_l = int(pt_left[0]), int(pt_left[1])
        half_p = patch_size // 2

        if x_l < half_p or x_l >= w - half_p or y_l < half_p or y_l >= h - half_p:
            return None

        patch = frame_left[y_l - half_p:y_l + half_p + 1, x_l - half_p:x_l + half_p + 1]

        y_min = max(0, y_l - 8)
        y_max = min(h, y_l + 9)
        x_min = max(0, x_l - search_width)
        x_max = min(w, x_l + 10)

        strip = frame_right[y_min:y_max, x_min:x_max]
        if strip.shape[0] < patch.shape[0] or strip.shape[1] < patch.shape[1]:
            return None

        res = cv2.matchTemplate(strip, patch, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > 0.40:
            matched_x = x_min + max_loc[0] + half_p
            matched_y = y_min + max_loc[1] + half_p
            return (matched_x, matched_y)
        return None


# ==============================================================================
# 3. QUẢN LÝ THU NHẬN HÌNH ẢNH 2 CAMERA (MASTER PI #1 & WORKER PI #2)
# ==============================================================================
class CameraStreamClient:
    """Client nhận luồng TCP video từ Pi #2 (Worker - Camera Phải)"""
    def __init__(self, host='192.168.88.2', port=5555):
        self.host = host
        self.port = port
        self.sock = None
        self.connected = False
        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def _recv_all(self, n):
        data = bytearray()
        while len(data) < n:
            packet = self.sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def _connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(3.0)
            self.sock.connect((self.host, self.port))
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.connected = True
            logger.info(f"Đã kết nối thành công tới Camera Phải Pi #2 ({self.host}:{self.port})")
            return True
        except Exception:
            self.connected = False
            return False

    def _worker_loop(self):
        while self.running:
            if not self.connected:
                if not self._connect():
                    time.sleep(1.0)
                    continue

            try:
                header = self._recv_all(4)
                if not header:
                    self.connected = False
                    continue
                msg_len = struct.unpack(">I", header)[0]
                frame_data = self._recv_all(msg_len)
                if not frame_data:
                    self.connected = False
                    continue

                np_arr = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    with self.lock:
                        self.latest_frame = frame
            except Exception:
                self.connected = False
                time.sleep(0.5)

    def read(self):
        with self.lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
            return False, None

    def close(self):
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except:
                pass


def init_left_camera(width=1280, height=720):
    """Khởi tạo Camera Trái trên Pi #1 (Picamera2 hoặc OpenCV)"""
    try:
        from picamera2 import Picamera2
        logger.info("Đang khởi tạo Camera Trái qua Picamera2...")
        try:
            picam2 = Picamera2(0)
        except TypeError:
            picam2 = Picamera2()
        cfg = picam2.create_preview_configuration(main={"size": (width, height), "format": "RGB888"})
        picam2.configure(cfg)
        picam2.start()
        logger.info("Camera Trái (Picamera2) đã sẵn sàng.")
        return ("picam2", picam2)
    except Exception as e:
        logger.warning(f"Không thể dùng Picamera2 ({e}). Chuyển sang fallback cv2.VideoCapture...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        logger.info("Camera Trái (OpenCV VideoCapture) đã sẵn sàng.")
        return ("opencv", cap)

    raise RuntimeError("Không tìm thấy Camera Trái trên Pi #1!")


# ==============================================================================
# 4. WEB STREAMING SERVER (XEM TRỰC TIẾP TRÊN TRÌNH DUYỆT TỪ LAPTOP)
# ==============================================================================
global_stream_jpeg = None
stream_lock = threading.Lock()

class WebStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global global_stream_jpeg
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Stereo Pose & Hand Raise Demo</title>
                <style>
                    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0b0f19; color: #fff; text-align: center; margin: 0; padding: 20px; }
                    h1 { color: #38bdf8; margin-bottom: 6px; font-size: 24px; }
                    p.sub { color: #94a3b8; margin-top: 0; font-size: 14px; }
                    .video-wrapper { display: inline-block; border: 3px solid #0284c7; border-radius: 12px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.8); max-width: 95vw; }
                    img { display: block; width: 100%; height: auto; }
                    .info-grid { max-width: 980px; margin: 18px auto; display: grid; grid-template-columns: 1fr 1fr; gap: 15px; text-align: left; }
                    .card { background: #1e293b; padding: 15px 20px; border-radius: 10px; border: 1px solid #334155; font-size: 13.5px; line-height: 1.6; }
                    .card h3 { margin-top: 0; color: #facc15; font-size: 15px; }
                    .tag-l { color: #38bdf8; font-weight: bold; }
                    .tag-r { color: #f43f5e; font-weight: bold; }
                </style>
            </head>
            <body>
                <h1>🎯 DEMO LUỒNG HOẠT ĐỘNG: NHẬN DIỆN GIƠ TAY & STEREO 3D</h1>
                <p class="sub">Khung hình ghép: <span class="tag-l">[CAM TRÁI: PI #1 MASTER]</span> | <span class="tag-r">[CAM PHẢI: PI #2 WORKER]</span></p>
                <div class="video-wrapper">
                    <img src="/stream.mjpg" alt="Stereo Pose Live Stream" />
                </div>
                <div class="info-grid">
                    <div class="card">
                        <h3>📌 Đo Khoảng Cách Tâm Bounding Box</h3>
                        • <strong>Tâm Bounding Box:</strong> Điểm giữa của khung bao quanh người (dấu chữ thập vàng).<br>
                        • <strong>D_mid:</strong> Khoảng cách đường thẳng từ tâm bounding box tới <strong>TRUNG ĐIỂM BASELINE</strong>.<br>
                        • <strong>Z:</strong> Khoảng cách vuông góc từ mặt phẳng 2 camera tới người.<br>
                        • <strong>X_mid:</strong> Vị trí lệch ngang (0: chính giữa trục nối 2 camera).
                    </div>
                    <div class="card">
                        <h3>🤖 AI Nhận Diện Giơ Tay (YOLO Pose)</h3>
                        • Nhận diện 17 khớp xương người bằng mô hình tối ưu NCNN trên CPU Pi 5.<br>
                        • Tự động phát hiện <strong>HAND RAISED</strong> khi cổ tay vượt lên cao hơn vai.<br>
                        • Khi giơ tay: Viền hộp đổi sang <strong>ĐỎ</strong> và hiển thị thông số 3D tức thì.
                    </div>
                </div>
            </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    with stream_lock:
                        if global_stream_jpeg is None:
                            continue
                        frame_bytes = global_stream_jpeg
                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(frame_bytes)))
                    self.end_headers()
                    self.wfile.write(frame_bytes)
                    self.wfile.write(b'\r\n')
                    time.sleep(0.03)
            except Exception:
                pass
        else:
            self.send_error(404)

def start_web_server(port=5000):
    try:
        server = HTTPServer(('0.0.0.0', port), WebStreamHandler)
        server.serve_forever()
    except Exception as e:
        logger.warning(f"Lỗi khởi tạo Web Server tại port {port}: {e}")


# ==============================================================================
# 5. TÌM KIẾM MÔ HÌNH VÀ FILE HIỆU CHUẨN CÓ SẴN TRÊN PI #1
# ==============================================================================
def resolve_calibration_file(custom_path=None):
    """
    Tự động tìm kiếm file stereo_calibration.npz có sẵn trên Pi #1.
    Không yêu cầu người dùng phải copy thêm bất kỳ thư mục dữ liệu nào.
    """
    candidates = [
        custom_path,
        # Đường dẫn chuẩn trên Pi #1 trong dự án:
        os.path.join(os.path.expanduser("~"), "hand_detection/calibrate-camera/stereo_vision/calibration_result/stereo_calibration.npz"),
        os.path.join(os.path.expanduser("~"), "hand_detection/stereo_vision/calibration_result/stereo_calibration.npz"),
        "stereo_vision/calibration_result/stereo_calibration.npz",
        "calibrate-camera/stereo_vision/calibration_result/stereo_calibration.npz",
        # Dự phòng đường dẫn máy tính Host PC nếu chạy test tại chỗ:
        "prototype_5_8x5_0,042_0,526/calibration_result/stereo_calibration.npz",
        "../prototype_5_8x5_0,042_0,526/calibration_result/stereo_calibration.npz"
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "Không tìm thấy file ma trận stereo_calibration.npz trên hệ thống!\n"
        "Vui lòng chỉ định đường dẫn qua tham số --calib_file."
    )

def resolve_model_path():
    candidates = [
        "yolo26n-pose_ncnn_model",
        "yolov26n-pose_ncnn_model",
        "yolo26n-pose.pt",
        "yolov26n-pose.pt",
        os.path.join(os.path.expanduser("~"), "hand_detection/yolo26n-pose_ncnn_model"),
        os.path.join(os.path.expanduser("~"), "hand_detection/yolo26n-pose.pt")
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "yolo26n-pose.pt"


# ==============================================================================
# 6. VẼ KHUNG XƯƠNG
# ==============================================================================
def draw_skeleton_and_kpts(frame, keypoints, is_raised=False):
    """Vẽ khung xương và các điểm khớp"""
    color_bone = (0, 255, 255) if not is_raised else (0, 165, 255)
    for a, b in SKELETON:
        xa, ya, ca = keypoints[a]
        xb, yb, cb = keypoints[b]
        if ca > 0.25 and cb > 0.25:
            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), color_bone, 2, cv2.LINE_AA)

    for idx in range(5, len(keypoints)):
        x, y, conf = keypoints[idx]
        if conf > 0.25:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)


# ==============================================================================
# 7. CHƯƠNG TRÌNH CHÍNH (MAIN LOOP)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Demo Test Luồng: Nhận Diện Giơ Tay + Đo Khoảng Cách Tâm Bounding Box Đến Trung Điểm Baseline")
    parser.add_argument('--calib_file', type=str, default=None, 
                        help="Đường dẫn file stereo_calibration.npz (mặc định tự tìm file có sẵn trên Pi #1)")
    parser.add_argument('--right_ip', type=str, default='192.168.88.2', 
                        help="IP mạng dây của Pi #2 (mặc định: 192.168.88.2)")
    parser.add_argument('--right_port', type=int, default=5555, 
                        help="Cổng TCP nhận hình từ Pi #2 (mặc định: 5555)")
    parser.add_argument('--port', type=int, default=5000, 
                        help="Cổng Web Stream (mặc định: 5000)")
    parser.add_argument('--conf', type=float, default=0.3, 
                        help="Ngưỡng tin cậy phát hiện người (mặc định: 0.3)")
    parser.add_argument('--standalone', action='store_true', 
                        help="Chế độ test độc lập không cần Pi #2 (nhân bản frame để kiểm thử)")
    args = parser.parse_args()

    logger.info("==================================================================")
    logger.info("  DEMO TEST LUỒNG: GIƠ TAY & ĐO KHOẢNG CÁCH TÂM BOUNDING BOX      ")
    logger.info("==================================================================")

    # 1. Tự động nạp file hiệu chuẩn stereo_calibration.npz có sẵn trên Pi #1
    calib_path = resolve_calibration_file(args.calib_file)
    logger.info(f"Nạp ma trận hiệu chuẩn từ: {calib_path}")
    stereo_solver = StereoTriangulator(calib_path)

    # 2. Khởi tạo mô hình AI YOLO Pose (ưu tiên NCNN trên Pi 5)
    model_path = resolve_model_path()
    logger.info(f"Đang nạp mô hình YOLO Pose từ: {model_path}...")
    model = YOLO(model_path)
    logger.info("Mô hình AI đã sẵn sàng.")

    # 3. Khởi tạo Camera Trái (Local Pi #1)
    cam_l_type, cam_left = init_left_camera()

    # 4. Khởi tạo Camera Phải (Network TCP Socket từ Pi #2)
    client_right = None
    if not args.standalone:
        logger.info(f"Đang kết nối luồng TCP Camera Phải từ Pi #2 ({args.right_ip}:{args.right_port})...")
        client_right = CameraStreamClient(host=args.right_ip, port=args.right_port)
    else:
        logger.warning("[CHẾ ĐỘ STANDALONE] Nhân bản Camera Trái để kiểm thử.")

    # 5. Khởi động Web Server phát video
    t_web = threading.Thread(target=start_web_server, args=(args.port,), daemon=True)
    t_web.start()
    logger.info(f"Web Stream phát trực tiếp tại: http://<IP_PI>:{args.port}")

    has_display = bool(os.environ.get('DISPLAY', None))
    window_name = "Stereo Pose Demo - Bounding Box Distance"
    if has_display:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        except:
            has_display = False

    global global_stream_jpeg
    frame_count = 0
    fps_timer = time.time()
    fps = 0.0

    logger.info("Bắt đầu vòng lặp xử lý chính. Nhấn Ctrl+C để dừng.")

    try:
        while True:
            # Thu nhận Frame từ Camera Trái
            if cam_l_type == "picam2":
                frame_l = cam_left.capture_array("main")
            else:
                ret_l, frame_l = cam_left.read()
                if not ret_l or frame_l is None:
                    time.sleep(0.01)
                    continue

            # Thu nhận Frame từ Camera Phải
            if args.standalone or client_right is None:
                frame_r = frame_l.copy()
            else:
                ret_r, frame_r = client_right.read()
                if not ret_r or frame_r is None:
                    # Nếu chưa có frame từ Pi #2, hiển thị chờ kết nối
                    frame_r = np.zeros_like(frame_l)
                    cv2.putText(frame_r, f"DANG KET NOI PI #2 ({args.right_ip}:{args.right_port})...", 
                                (40, frame_r.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)

            frame_count += 1
            if frame_count % 15 == 0:
                elapsed = time.time() - fps_timer
                fps = 15.0 / elapsed if elapsed > 0 else 0.0
                fps_timer = time.time()

            disp_l = frame_l.copy()
            disp_r = frame_r.copy()

            # Chạy YOLO Pose trên Camera Trái (imgsz=480)
            results_l = model(disp_l, imgsz=480, conf=args.conf, device="cpu", verbose=False)
            res_l = results_l[0]

            # Chạy YOLO Pose trên Camera Phải để đối sánh Bounding Box & Khớp xương
            res_r = None
            if client_right and client_right.connected:
                results_r = model(disp_r, imgsz=480, conf=args.conf, device="cpu", verbose=False)
                res_r = results_r[0]

            person_count = 0
            raised_count = 0
            primary_person_info = None  # Lưu thông tin người gần nhất / giơ tay

            if res_l.boxes is not None and res_l.keypoints is not None:
                boxes_l = res_l.boxes.xyxy.cpu().numpy()
                confs_l = res_l.boxes.conf.cpu().numpy()
                kpts_l = res_l.keypoints.data.cpu().numpy()

                person_count = len(boxes_l)

                for i in range(len(boxes_l)):
                    x1, y1, x2, y2 = map(int, boxes_l[i])
                    kpt_l = kpts_l[i]
                    is_raised, side, wrist_idx = check_hand_raise(kpt_l)

                    # TÍNH TÂM CỦA BOUNDING BOX TRÊN CAMERA TRÁI
                    cx_l = float((x1 + x2) / 2.0)
                    cy_l = float((y1 + y2) / 2.0)
                    bbox_center_l = (cx_l, cy_l)

                    # Tìm Tâm Bounding Box tương ứng trên Camera Phải
                    bbox_center_r = None

                    if res_r and res_r.boxes is not None and len(res_r.boxes) > 0:
                        boxes_r = res_r.boxes.xyxy.cpu().numpy()
                        kpts_r = res_r.keypoints.data.cpu().numpy()

                        # Ghép đôi người tương ứng bên Cam Phải (dựa trên sự gần gũi trục dọc y và disparity >= 0)
                        best_match_idx = 0
                        min_cost = float('inf')
                        for j in range(len(boxes_r)):
                            rx1, ry1, rx2, ry2 = boxes_r[j]
                            rcx = (rx1 + rx2) / 2.0
                            rcy = (ry1 + ry2) / 2.0
                            # Trong rectified stereo: rcy ~ cy_l và rcx <= cx_l
                            cost = abs(rcy - cy_l) * 2.0 + max(0, rcx - cx_l) * 5.0
                            if cost < min_cost:
                                min_cost = cost
                                best_match_idx = j

                        rx1, ry1, rx2, ry2 = map(int, boxes_r[best_match_idx])
                        bbox_center_r = (float((rx1 + rx2) / 2.0), float((ry1 + ry2) / 2.0))

                        # Vẽ khung và tâm bên Cam Phải
                        box_color_r = (0, 0, 255) if is_raised else (0, 255, 0)
                        cv2.rectangle(disp_r, (rx1, ry1), (rx2, ry2), box_color_r, 2)
                        cv2.drawMarker(disp_r, (int(bbox_center_r[0]), int(bbox_center_r[1])), 
                                       (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
                        draw_skeleton_and_kpts(disp_r, kpts_r[best_match_idx], is_raised)

                    # Nếu Cam Phải không bắt được bbox bằng Pose, dùng Epipolar Patch Matching ngay tại tâm bbox
                    if bbox_center_r is None:
                        bbox_center_r = stereo_solver.match_epipolar_patch(frame_l, frame_r, bbox_center_l)
                        if bbox_center_r is not None:
                            cv2.drawMarker(disp_r, (int(bbox_center_r[0]), int(bbox_center_r[1])), 
                                           (0, 255, 255), cv2.MARKER_CROSS, 16, 2)

                    # TAM GIÁC ĐẠC 3D: TỪ TÂM BOUNDING BOX TỚI TRUNG ĐIỂM BASELINE
                    dist_label = ""
                    if bbox_center_r is not None:
                        X_mid, Y_mid, Z_mid, D_mid = stereo_solver.triangulate_single_point(bbox_center_l, bbox_center_r)
                        d_cm = D_mid * 100.0
                        z_cm = Z_mid * 100.0
                        x_cm = X_mid * 100.0

                        dist_label = f"D_mid: {d_cm:.1f}cm (Z:{z_cm:.1f}cm)"

                        # Ghi nhận người để hiển thị trên HUD
                        if primary_person_info is None or is_raised:
                            primary_person_info = (d_cm, z_cm, x_cm, is_raised, side)

                    # Trực quan hóa trên Cam Trái
                    box_color_l = (0, 0, 255) if is_raised else (0, 255, 0)
                    cv2.rectangle(disp_l, (x1, y1), (x2, y2), box_color_l, 2)
                    draw_skeleton_and_kpts(disp_l, kpt_l, is_raised)

                    # Vẽ dấu chữ thập vàng tại TÂM BOUNDING BOX
                    cv2.drawMarker(disp_l, (int(cx_l), int(cy_l)), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)

                    # Nhãn trạng thái
                    if is_raised:
                        raised_count += 1
                        wx_l, wy_l = int(kpt_l[wrist_idx][0]), int(kpt_l[wrist_idx][1])
                        cv2.circle(disp_l, (wx_l, wy_l), 10, (0, 0, 255), -1)
                        header_txt = f"HAND RAISED [{side}]"
                    else:
                        header_txt = f"Person {confs_l[i]:.2f}"

                    cv2.putText(disp_l, header_txt, (x1, max(25, y1 - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, box_color_l, 2)

                    # Hiển thị khoảng cách từ tâm bounding box tới trung điểm baseline ngay trên khung người
                    if dist_label:
                        cv2.putText(disp_l, dist_label, (x1, max(50, y1 - 32)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (250, 204, 21), 2)

            # Ghép khung hình Side-by-Side (Cam Trái | Cam Phải)
            scale = 0.75
            h_orig, w_orig = disp_l.shape[:2]
            w_s, h_s = int(w_orig * scale), int(h_orig * scale)
            s_l = cv2.resize(disp_l, (w_s, h_s))
            s_r = cv2.resize(disp_r, (w_s, h_s))

            cv2.putText(s_l, "CAM TRAI (PI #1 MASTER)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (56, 189, 248), 2)
            cv2.putText(s_r, "CAM PHAI (PI #2 WORKER)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (244, 63, 94), 2)

            combined_feed = cv2.hconcat([s_l, s_r])
            comb_h, comb_w = combined_feed.shape[:2]

            # BẢNG HUD HIỂN THỊ KẾT QUẢ ĐO STEREO 3D
            hud_height = 145
            hud_panel = np.zeros((hud_height, comb_w, 3), dtype=np.uint8)
            hud_panel[:] = (15, 23, 42)

            cv2.line(hud_panel, (0, 0), (comb_w, 0), (56, 189, 248), 2)
            cv2.line(hud_panel, (w_s, 0), (w_s, hud_height), (51, 65, 85), 1)

            if primary_person_info is not None:
                d_cm, z_cm, x_cm, is_r, side_str = primary_person_info
                pos_str = "CHINH GIUA" if abs(x_cm) < 4.0 else (f"LECH PHAI {x_cm:+.1f}cm" if x_cm > 0 else f"LECH TRAI {abs(x_cm):.1f}cm")
                
                if is_r:
                    cv2.putText(hud_panel, f"PHAT HIEN NGUOI GIO TAY [{side_str}]! D_mid: {d_cm:.1f} cm", 
                                (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 255), 2)
                else:
                    cv2.putText(hud_panel, f"KHOANG CACH TAM BBOX -> TRUNG DIEM BASELINE: {d_cm:.1f} cm", 
                                (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (250, 204, 21), 2)

                cv2.putText(hud_panel, f"Z (Vuong goc): {z_cm:.1f} cm | Vi tri ngang X_mid: [{pos_str}]", 
                            (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (74, 222, 128), 2)
            else:
                status_txt = f"SO NGUOI: {person_count} | KHONG CO NGUOI GIO TAY" if person_count > 0 else "DANG QUAN SAT... CHUA THAY NGUOI"
                cv2.putText(hud_panel, status_txt, (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.85, 
                            (74, 222, 128) if person_count > 0 else (148, 163, 184), 2)
                cv2.putText(hud_panel, "Do khoang cach tu Tam Bounding Box den Trung diem Baseline giua 2 camera", 
                            (30, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (148, 163, 184), 1)

            cv2.putText(hud_panel, f"Baseline Stereo: {stereo_solver.baseline*100:.1f}cm | Web Stream: http://<IP>:5000 | FPS: {fps:.1f}",
                        (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (148, 163, 184), 1)

            final_view = cv2.vconcat([combined_feed, hud_panel])

            # Nén JPEG gửi Web Stream
            ret_enc, jpeg_data = cv2.imencode(".jpg", final_view, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ret_enc:
                with stream_lock:
                    global_stream_jpeg = jpeg_data.tobytes()

            if has_display:
                cv2.imshow(window_name, final_view)
                if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
                    break

    except KeyboardInterrupt:
        logger.info("\nNhận tín hiệu dừng từ người dùng.")
    finally:
        if cam_l_type == "picam2":
            cam_left.stop()
            cam_left.close()
        else:
            cam_left.release()
        if client_right:
            client_right.close()
        if has_display:
            cv2.destroyAllWindows()
        logger.info("Đã dọn dẹp tài nguyên và tắt ứng dụng thành công.")

if __name__ == "__main__":
    main()
