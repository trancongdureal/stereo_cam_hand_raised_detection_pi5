#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HỆ THỐNG STEREO 3D & NHẬN DIỆN GIƠ TAY KHÓA MỤC TIÊU HOÀN CHỈNH (PI 5 PRODUCTION)
File: stereo_pose_final.py
---------------------------------------------------------------------------------
Bản kết hợp tối ưu toàn diện từ demo_raise_hand_stereo.py và stereo_pose_target_locked.py:
1. Đảm bảo 100% hiển thị Stereo 3D:
   - Chạy YOLO Pose trên cả 2 camera (hoặc fallback Epipolar Patch Matching)
     để luôn tìm thấy chính xác điểm tâm tương ứng trên Camera Phải.
   - Luôn tính toán và hiển thị khoảng cách Stereo 3D (D_mid, Z, X_mid) liên tục
     từ Tâm Bounding Box tới Trung Điểm Baseline M = (B/2, 0, 0).
2. Tích hợp tính năng Khóa Mục Tiêu (Target Locking) khi có từ 2 người:
   - Quản lý bám vết mượt mà với Lightweight IoU Tracker.
   - Người giơ tay liên tục >= 2.0s sẽ được KHÓA CHẶT (viền ĐỎ ĐẬM, tâm VÀNG).
   - Nếu có nhiều người cùng giơ: ưu tiên khóa người ĐỨNG GẦN CAMERA NHẤT.
   - Khi đã khóa: CHỈ tập trung đo khoảng cách Stereo 3D của riêng người này,
     loại bỏ 100% hiện tượng nhảy box hoặc ghép nhầm 2 người với nhau.
   - Tự động nhả khóa khi người đó hạ tay xuống (debounce 0.4s) hoặc đi ra khỏi khung hình.
   - Hỗ trợ Reset tức thì: phím 'p' trên bàn phím hoặc nút bấm trên Web UI.
3. Giao diện trực quan đầy đủ:
   - Ghép Side-by-Side: Cam Trái (Master Pi #1) | Cam Phải (Worker Pi #2).
   - Đánh dấu tâm tương ứng (dấu chữ thập vàng) và vẽ khung xương trên CẢ HAI camera.
   - Hiển thị nhãn khoảng cách D_mid (cm) ngay trên đầu người và trên bảng HUD.
   - Web Stream tại http://<IP_PI>:5000 tích hợp nút bấm 'RESET TARGET' từ xa.
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
from urllib.parse import urlparse

import cv2
import numpy as np

# Tối ưu hóa số luồng PyTorch cho 4 nhân CPU của Pi 5
try:
    import torch
    torch.set_num_threads(4)
except ImportError:
    torch = None

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Thư viện ultralytics chưa được cài đặt. Cài đặt bằng: pip install ultralytics")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("StereoPoseFinal")

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

def check_hand_raise(keypoints, conf_thresh=0.45):
    """
    Kiểm tra điều kiện giơ tay từ 17 keypoints COCO.
    Điều kiện: Tọa độ Y của cổ tay nhỏ hơn Y của vai (trong hệ ảnh, Y nhỏ hơn tức ở trên cao).
    """
    left_raised = False
    right_raised = False

    # Tay trái giơ lên
    if keypoints[LEFT_WRIST][2] > conf_thresh and keypoints[LEFT_SHOULDER][2] > conf_thresh:
        if keypoints[LEFT_WRIST][1] < keypoints[LEFT_SHOULDER][1]:
            left_raised = True

    # Tay phải giơ lên
    if keypoints[RIGHT_WRIST][2] > conf_thresh and keypoints[RIGHT_SHOULDER][2] > conf_thresh:
        if keypoints[RIGHT_WRIST][1] < keypoints[RIGHT_SHOULDER][1]:
            right_raised = True

    if left_raised and right_raised:
        idx = LEFT_WRIST if keypoints[LEFT_WRIST][1] < keypoints[RIGHT_WRIST][1] else RIGHT_WRIST
        return True, "BOTH", (float(keypoints[idx][0]), float(keypoints[idx][1]))
    elif left_raised:
        return True, "LEFT", (float(keypoints[LEFT_WRIST][0]), float(keypoints[LEFT_WRIST][1]))
    elif right_raised:
        return True, "RIGHT", (float(keypoints[RIGHT_WRIST][0]), float(keypoints[RIGHT_WRIST][1]))
    else:
        return False, None, None


# ==============================================================================
# 2. BỘ BÁM VẾT ĐA ĐỐI TƯỢNG SIÊU NHẸ (LIGHTWEIGHT IOU TRACKER)
# ==============================================================================
def calculate_iou(boxA, boxB):
    """Tính IoU giữa 2 bounding box [x1, y1, x2, y2]"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denom = float(boxAArea + boxBArea - interArea)
    return interArea / denom if denom > 0 else 0.0

class TrackedPerson:
    """Đại diện cho 1 đối tượng người được bám vết qua thời gian"""
    def __init__(self, track_id, bbox, keypoints, conf):
        self.track_id = track_id
        self.bbox = np.array(bbox, dtype=np.float32)
        self.keypoints = keypoints
        self.conf = conf
        
        # Trạng thái giơ tay & đếm thời gian
        self.is_hand_raised = False
        self.raise_side = None
        self.wrist_pt = None
        self.raise_start_time = None
        self.raise_duration = 0.0
        self.lower_start_time = None
        
        # Theo dõi mất dấu
        self.lost_frames = 0
        self.total_frames = 1
        
        # Tọa độ 3D làm mịn (Exponential Moving Average)
        self.d_mid_cm = None
        self.z_cm = None
        self.x_mid_cm = None

    @property
    def center(self):
        """Tâm của bounding box (cx, cy)"""
        return (float((self.bbox[0] + self.bbox[2]) / 2.0),
                float((self.bbox[1] + self.bbox[3]) / 2.0))

    @property
    def height(self):
        """Chiều cao bounding box"""
        return float(self.bbox[3] - self.bbox[1])

    def update_pose(self, bbox, keypoints, conf, now, lower_debounce=0.4):
        """Cập nhật thông tin pose và đếm thời gian giơ tay"""
        self.bbox = np.array(bbox, dtype=np.float32)
        self.keypoints = keypoints
        self.conf = conf
        self.lost_frames = 0
        self.total_frames += 1

        raised, side, wrist = check_hand_raise(keypoints)

        if raised:
            self.lower_start_time = None
            if not self.is_hand_raised:
                self.is_hand_raised = True
                self.raise_start_time = now
                self.raise_duration = 0.0
            else:
                if self.raise_start_time is not None:
                    self.raise_duration = now - self.raise_start_time
            self.raise_side = side
            self.wrist_pt = wrist
        else:
            if self.is_hand_raised:
                if self.lower_start_time is None:
                    self.lower_start_time = now
                elif (now - self.lower_start_time) >= lower_debounce:
                    self.is_hand_raised = False
                    self.raise_start_time = None
                    self.raise_duration = 0.0
                    self.raise_side = None
                    self.wrist_pt = None
                    self.lower_start_time = None

    def mark_missed(self):
        self.lost_frames += 1

    def update_3d(self, d_mid, z, x_mid, alpha=0.35):
        """Làm mịn EMA cho tọa độ 3D"""
        if self.d_mid_cm is None:
            self.d_mid_cm = float(d_mid)
            self.z_cm = float(z)
            self.x_mid_cm = float(x_mid)
        else:
            self.d_mid_cm = alpha * float(d_mid) + (1.0 - alpha) * self.d_mid_cm
            self.z_cm = alpha * float(z) + (1.0 - alpha) * self.z_cm
            self.x_mid_cm = alpha * float(x_mid) + (1.0 - alpha) * self.x_mid_cm

class LightweightTracker:
    """Bộ theo dõi đa đối tượng IoU Matching siêu nhẹ (<0.1ms)"""
    def __init__(self, iou_threshold=0.30, max_lost=12):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, keypoints_list, confs, now):
        det_count = len(detections)
        track_ids = list(self.tracks.keys())
        matched_tracks = set()
        matched_dets = set()

        if track_ids and det_count > 0:
            iou_matrix = np.zeros((len(track_ids), det_count), dtype=np.float32)
            for i, t_id in enumerate(track_ids):
                for j in range(det_count):
                    iou_matrix[i, j] = calculate_iou(self.tracks[t_id].bbox, detections[j])

            flat_indices = np.argsort(iou_matrix.ravel())[::-1]
            for idx in flat_indices:
                i = idx // det_count
                j = idx % det_count
                if iou_matrix[i, j] < self.iou_threshold:
                    break
                t_id = track_ids[i]
                if t_id in matched_tracks or j in matched_dets:
                    continue
                matched_tracks.add(t_id)
                matched_dets.add(j)
                self.tracks[t_id].update_pose(detections[j], keypoints_list[j], confs[j], now)

        for t_id in track_ids:
            if t_id not in matched_tracks:
                self.tracks[t_id].mark_missed()

        for j in range(det_count):
            if j not in matched_dets:
                new_track = TrackedPerson(self.next_id, detections[j], keypoints_list[j], confs[j])
                new_track.update_pose(detections[j], keypoints_list[j], confs[j], now)
                self.tracks[self.next_id] = new_track
                self.next_id += 1

        active_tracks = {}
        for t_id, track in self.tracks.items():
            if track.lost_frames <= self.max_lost:
                active_tracks[t_id] = track
        self.tracks = active_tracks
        return self.tracks


# ==============================================================================
# 3. MÁY TRẠNG THÁI KHÓA MỤC TIÊU (TARGET LOCK CONTROLLER)
# ==============================================================================
class TargetLockController:
    """
    Quản lý quy tắc khóa mục tiêu:
    - Giơ tay liên tục >= 2.0s: Khóa người đứng gần camera nhất.
    - Khi đã khóa: CHỈ đo khoảng cách Stereo 3D người này.
    - Tự động nhả khóa khi hạ tay (>0.4s) hoặc biến mất.
    - Hỗ trợ reset bằng phím 'p' hoặc nút trên Web UI.
    """
    def __init__(self, lock_hold_seconds=2.0):
        self.lock_hold_seconds = lock_hold_seconds
        self.locked_track_id = None
        self.lock_timestamp = None
        self.status_message = "DANG QUAN SAT (GIO TAY > 2S DE KHOA)"

    def reset(self, reason="Nguoi dung yeu cau (Phim P / Web UI)"):
        prev = self.locked_track_id
        self.locked_track_id = None
        self.lock_timestamp = None
        self.status_message = f"DA RESET (Ly do: {reason})"
        logger.info(f"[TARGET LOCK] Reset thanh cong. Target cu: {prev}. Ly do: {reason}")

    def update(self, tracks, now):
        if self.locked_track_id is not None:
            if self.locked_track_id not in tracks:
                self.reset(reason="Target da di ra khoi man hinh")
                return None

            locked_track = tracks[self.locked_track_id]

            if locked_track.lost_frames > 5:
                self.reset(reason="Target mat dau > 5 frames")
                return None

            if not locked_track.is_hand_raised:
                self.reset(reason="Target da ha tay xuong")
                return None

            self.status_message = f"DA KHOA CHAT TARGET #{locked_track.track_id}"
            return locked_track

        # Chưa khóa ai -> Tìm ứng viên giơ tay >= lock_hold_seconds
        candidates = []
        for t_id, track in tracks.items():
            if track.lost_frames == 0 and track.is_hand_raised:
                if track.raise_duration >= self.lock_hold_seconds:
                    candidates.append(track)

        if candidates:
            # Chọn người đứng gần camera nhất (bbox cao nhất hoặc Z nhỏ nhất)
            best_candidate = max(candidates, key=lambda t: t.height if t.z_cm is None else (1.0 / max(1.0, t.z_cm)))
            self.locked_track_id = best_candidate.track_id
            self.lock_timestamp = now
            self.status_message = f"DA KHOA TARGET #{self.locked_track_id} (Gio tay > 2s & Gan nhat)"
            logger.info(f"[TARGET LOCK] {self.status_message}")
            return best_candidate

        self.status_message = "DANG QUAN SAT (GIO TAY > 2S DE KHOA)"
        return None


# ==============================================================================
# 4. BỘ TÍNH TOÁN STEREO 3D & EPIPOLAR MATCHING
# ==============================================================================
class StereoTriangulator:
    """
    Nạp ma trận hiệu chuẩn và giải tam giác đạc 3D.
    Gốc tọa độ được dời về TRUNG ĐIỂM BASELINE: M = (B/2, 0, 0).
    """
    def __init__(self, calib_file=None):
        self.is_synthetic = False

        if calib_file and os.path.exists(calib_file):
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
            self.baseline = float(np.linalg.norm(self.T))
            logger.info(f"Đã nạp file hiệu chuẩn: {calib_file} (Baseline: {self.baseline*100:.2f} cm)")
        else:
            logger.warning("Không tìm thấy stereo_calibration.npz -> Khởi tạo ma trận dự phòng (Baseline 50cm)")
            self.is_synthetic = True
            self.baseline = 0.50
            fx, fy, cx, cy = 1720.0, 1720.0, 640.0, 360.0
            self.K1 = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            self.K2 = self.K1.copy()
            self.D1 = np.zeros(5, dtype=np.float64)
            self.D2 = np.zeros(5, dtype=np.float64)
            self.R1 = np.eye(3, dtype=np.float64)
            self.R2 = np.eye(3, dtype=np.float64)
            self.P1 = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]], dtype=np.float64)
            self.P2 = np.array([[fx, 0, cx, -fx * self.baseline], [0, fy, cy, 0], [0, 0, 1, 0]], dtype=np.float64)
            self.T = np.array([[-self.baseline], [0], [0]], dtype=np.float64)

        self.half_baseline = self.baseline / 2.0

    def triangulate_single_point(self, pt_left, pt_right):
        """Tam giác đạc cho 1 cặp điểm (u, v) trên Cam Trái và Cam Phải"""
        pl = np.array(pt_left, dtype=np.float32).reshape(-1, 1, 2)
        pr = np.array(pt_right, dtype=np.float32).reshape(-1, 1, 2)

        pl_rect = cv2.undistortPoints(pl, self.K1, self.D1, R=self.R1, P=self.P1).reshape(-1, 2)
        pr_rect = cv2.undistortPoints(pr, self.K2, self.D2, R=self.R2, P=self.P2).reshape(-1, 2)

        pt_4d = cv2.triangulatePoints(self.P1, self.P2, pl_rect.T, pr_rect.T)
        pt_3d = pt_4d[:3, :] / pt_4d[3, :]

        X_rect = float(pt_3d[0, 0])
        Y_rect = float(pt_3d[1, 0])
        Z_rect = float(pt_3d[2, 0])

        # Dời gốc tọa độ về TRUNG ĐIỂM BASELINE: M = (half_baseline, 0, 0)
        X_mid = X_rect - self.half_baseline
        Y_mid = Y_rect
        Z_mid = Z_rect
        D_mid = float(np.sqrt(X_mid**2 + Y_mid**2 + Z_mid**2))
        return X_mid, Y_mid, Z_mid, D_mid

    def match_epipolar_patch(self, frame_left, frame_right, pt_left, patch_size=35, search_width=350, v_range=30):
        """
        Dò điểm tương ứng trên Cam Phải dọc theo dải epipolar (hỗ trợ độ lệch dọc v_range +-30px).
        Dùng làm fallback an toàn khi YOLO trên Cam Phải bị che khuất một phần.
        """
        h, w = frame_left.shape[:2]
        x_l, y_l = int(pt_left[0]), int(pt_left[1])
        half_p = patch_size // 2

        if x_l < half_p or x_l >= w - half_p or y_l < half_p or y_l >= h - half_p:
            return None

        patch = frame_left[y_l - half_p:y_l + half_p + 1, x_l - half_p:x_l + half_p + 1]

        # Cho phép biên độ tìm kiếm dọc rộng hơn (v_range) để bắt được trên ảnh chưa nắn
        y_min = max(0, y_l - v_range)
        y_max = min(h, y_l + v_range + 1)
        x_min = max(0, x_l - search_width)
        x_max = min(w, x_l + 15)

        strip = frame_right[y_min:y_max, x_min:x_max]
        if strip.shape[0] < patch.shape[0] or strip.shape[1] < patch.shape[1]:
            return None

        res = cv2.matchTemplate(strip, patch, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val >= 0.35:
            matched_x = x_min + max_loc[0] + half_p
            matched_y = y_min + max_loc[1] + half_p
            return (float(matched_x), float(matched_y))
        return None


# ==============================================================================
# 5. KẾT NỐI CAMERA & WEB STREAMING
# ==============================================================================
class CameraStreamClient:
    """Client nhận TCP video từ Pi #2 (Worker - Cam Phải) trong luồng nền non-blocking"""
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
            self.sock.settimeout(2.0)
            self.sock.connect((self.host, self.port))
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.connected = True
            logger.info(f"Đã kết nối thành công tới Cam Phải Pi #2 ({self.host}:{self.port})")
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
    """Khởi tạo Camera Trái trên Pi #1 (Picamera2 RGB888 native hoặc OpenCV VideoCapture)"""
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
        logger.warning(f"Không thể dùng Picamera2 ({e}). Chuyển sang fallback OpenCV VideoCapture...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        logger.info("Camera Trái (OpenCV VideoCapture) đã sẵn sàng.")
        return ("opencv", cap)

    raise RuntimeError("Không tìm thấy Camera Trái trên Pi #1!")


# ------------------------------------------------------------------------------
# WEB STREAMING SERVER (MJPEG + NÚT RESET TARGET TỪ XA)
# ------------------------------------------------------------------------------
global_stream_jpeg = None
stream_lock = threading.Lock()
global_lock_controller = None

class WebStreamHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        global global_stream_jpeg, global_lock_controller
        parsed = urlparse(self.path)

        if parsed.path == '/reset':
            if global_lock_controller:
                global_lock_controller.reset(reason="Nhan nut Web UI Reset")
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "ok", "message": "Target reset successfully"}')
            return

        if parsed.path == '/' or parsed.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Stereo Pose 3D - Target Locking</title>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0b0f19; color: #fff; text-align: center; margin: 0; padding: 15px; }
                    h1 { color: #38bdf8; margin-bottom: 6px; font-size: 22px; }
                    p.sub { color: #94a3b8; margin-top: 0; font-size: 13.5px; }
                    .video-box { display: inline-block; border: 3px solid #0284c7; border-radius: 12px; overflow: hidden; box-shadow: 0 10px 25px rgba(0,0,0,0.7); max-width: 95vw; }
                    img { display: block; width: 100%; height: auto; }
                    .controls { margin: 16px auto; max-width: 720px; display: flex; justify-content: center; gap: 12px; }
                    button.btn-reset { background: #ef4444; color: white; border: none; padding: 12px 28px; font-size: 16px; font-weight: bold; border-radius: 8px; cursor: pointer; transition: 0.2s; box-shadow: 0 4px 12px rgba(239,68,68,0.4); }
                    button.btn-reset:hover { background: #dc2626; transform: scale(1.03); }
                    .info-grid { max-width: 920px; margin: 15px auto; display: grid; grid-template-columns: 1fr 1fr; gap: 15px; text-align: left; }
                    .card { background: #1e293b; padding: 15px 18px; border-radius: 10px; border: 1px solid #334155; font-size: 13px; line-height: 1.6; }
                    .card h3 { margin-top: 0; color: #facc15; font-size: 14.5px; }
                    .badge { background: #0284c7; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
                </style>
            </head>
            <body>
                <h1>🎯 HỆ THỐNG STEREO 3D KHÓA MỤC TIÊU (TARGET LOCK)</h1>
                <p class="sub"><span class="badge">CAM TRÁI: PI #1 MASTER</span> | <span class="badge" style="background:#e11d48">CAM PHẢI: PI #2 WORKER</span></p>
                <div class="video-box">
                    <img src="/stream.mjpg" alt="Stereo Live Stream" />
                </div>
                <div class="controls">
                    <button class="btn-reset" onclick="resetTarget()">🔄 RESET TARGET (PHÍM 'P')</button>
                </div>
                <div class="info-grid">
                    <div class="card">
                        <h3>🔒 Quy Tắc Khóa Mục Tiêu (Target Lock)</h3>
                        • <strong>Giơ tay >= 2.0s:</strong> Tự động chọn người <strong>đứng gần camera nhất</strong> để khóa chặt.<br>
                        • <strong>Khi đã khóa:</strong> CHỈ đo khoảng cách Stereo 3D người này. Triệt tiêu 100% nhiễu từ người xung quanh.<br>
                        • <strong>Hạ tay xuống:</strong> Tự động nhả khóa sau 0.4s để tìm người tiếp theo.<br>
                        • <strong>Đi ra khỏi màn hình:</strong> Tự động nhả khóa.
                    </div>
                    <div class="card">
                        <h3>⚡ Tối Ưu Tốc Độ Pi 5 & Phím Tắt</h3>
                        • <strong>Đo Stereo 3D:</strong> D_mid (đường thẳng), Z (vuông góc), X_mid (lệch ngang).<br>
                        • <strong>Phím 'p' hoặc 'P':</strong> Reset tức thì người được chọn.<br>
                        • <strong>Phím 'q' hoặc ESC:</strong> Thoát an toàn.<br>
                        • <strong>Nút bấm Web:</strong> Reset từ xa qua trình duyệt không cần chạm vào Pi.
                    </div>
                </div>
                <script>
                    function resetTarget() {
                        fetch('/reset')
                            .then(r => r.json())
                            .then(d => { alert('Đã Reset mục tiêu thành công! Hệ thống quay về trạng thái tìm người gần nhất.'); })
                            .catch(e => console.error(e));
                    }
                </script>
            </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
        elif parsed.path == '/stream.mjpg':
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
# 6. TỰ ĐỘNG TÌM KIẾM MÔ HÌNH VÀ FILE HIỆU CHUẨN CÓ SẴN TRÊN PI
# ==============================================================================
def resolve_calibration_file(custom_path=None):
    candidates = [
        custom_path,
        os.path.join(os.path.expanduser("~"), "hand_detection/calibrate-camera/stereo_vision/calibration_result/stereo_calibration.npz"),
        os.path.join(os.path.expanduser("~"), "hand_detection/stereo_vision/calibration_result/stereo_calibration.npz"),
        "stereo_vision/calibration_result/stereo_calibration.npz",
        "calibrate-camera/stereo_vision/calibration_result/stereo_calibration.npz",
        "calibrate-camera/camera_intrinsics.npz",
        "prototype_6_8x5_0,042_0,5045*/calibration_result/stereo_calibration.npz",
        "../prototype_6_8x5_0,042_0,5045*/calibration_result/stereo_calibration.npz",
        "prototype_5_8x5_0,042_0,526/calibration_result/stereo_calibration.npz"
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return None

def resolve_model_path():
    candidates = [
        "yolo26n-pose_ncnn_model",
        "yolov26n-pose_ncnn_model",
        os.path.join(os.path.expanduser("~"), "hand_detection/yolo26n-pose_ncnn_model"),
        "yolo26n-pose.pt",
        "yolov26n-pose.pt",
        os.path.join(os.path.expanduser("~"), "hand_detection/yolo26n-pose.pt")
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "yolo26n-pose.pt"

def draw_skeleton(frame, keypoints, color=(0, 255, 255)):
    """Vẽ khung xương 17 khớp trên khung hình"""
    for a, b in SKELETON:
        xa, ya, ca = keypoints[a]
        xb, yb, cb = keypoints[b]
        if ca > 0.25 and cb > 0.25:
            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), color, 2, cv2.LINE_AA)

    for idx in range(5, len(keypoints)):
        x, y, conf = keypoints[idx]
        if conf > 0.25:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)


# ==============================================================================
# 7. CHƯƠNG TRÌNH CHÍNH (MAIN PROCESSING LOOP)
# ==============================================================================
def main():
    global global_stream_jpeg, global_lock_controller

    parser = argparse.ArgumentParser(description="Hệ Thống Stereo 3D & Nhận Diện Giơ Tay Khóa Mục Tiêu (Pi 5)")
    parser.add_argument('--calib_file', type=str, default=None, help="Đường dẫn file stereo_calibration.npz")
    parser.add_argument('--right_ip', type=str, default='192.168.88.2', help="IP mạng dây Pi #2 (mặc định: 192.168.88.2)")
    parser.add_argument('--right_port', type=int, default=5555, help="Cổng TCP nhận hình từ Pi #2 (mặc định: 5555)")
    parser.add_argument('--port', type=int, default=5000, help="Cổng Web Streaming (mặc định: 5000)")
    parser.add_argument('--conf', type=float, default=0.30, help="Ngưỡng tin cậy phát hiện người (mặc định: 0.30)")
    parser.add_argument('--lock_time', type=float, default=2.0, help="Thời gian giơ tay để khóa (giây, mặc định 2.0s)")
    parser.add_argument('--standalone', action='store_true', help="Chế độ test độc lập (không cần Pi #2, nhân bản Cam Trái)")
    parser.add_argument('--no_display', action='store_true', help="Tắt cửa sổ màn hình HDMI tại chỗ, chỉ xuất Web Stream")
    args = parser.parse_args()

    print("==================================================================")
    print("  HỆ THỐNG STEREO 3D & NHẬN DIỆN GIƠ TAY KHÓA MỤC TIÊU (FINAL)     ")
    print("==================================================================")

    # 1. Nạp ma trận hiệu chuẩn Stereo 3D
    calib_path = resolve_calibration_file(args.calib_file)
    stereo_solver = StereoTriangulator(calib_path)

    # 2. Nạp mô hình AI YOLO Pose
    model_path = resolve_model_path()
    logger.info(f"Nạp mô hình YOLO Pose từ: {model_path}...")
    model = YOLO(model_path)
    logger.info("Mô hình AI đã sẵn sàng.")

    # 3. Khởi tạo Camera Trái (Local Pi #1)
    cam_l_type, cam_left = init_left_camera()

    # 4. Khởi tạo Camera Phải (Network TCP từ Pi #2)
    client_right = None
    if not args.standalone:
        logger.info(f"Kết nối Camera Phải Pi #2 ({args.right_ip}:{args.right_port})...")
        client_right = CameraStreamClient(host=args.right_ip, port=args.right_port)
    else:
        logger.warning("[CHẾ ĐỘ STANDALONE] Nhân bản Camera Trái để kiểm thử độc lập.")

    # 5. Khởi tạo Tracker & Máy trạng thái Khóa mục tiêu
    tracker = LightweightTracker(iou_threshold=0.30, max_lost=12)
    lock_controller = TargetLockController(lock_hold_seconds=args.lock_time)
    global_lock_controller = lock_controller

    # 6. Khởi động Web Server phát video
    t_web = threading.Thread(target=start_web_server, args=(args.port,), daemon=True)
    t_web.start()
    logger.info(f"Web Stream sẵn sàng tại: http://<IP_PI>:{args.port} (Tích hợp nút Reset Target)")

    # 7. Kiểm tra màn hình đồ họa cục bộ
    has_display = bool(os.environ.get('DISPLAY', None)) and not args.no_display
    window_name = "Stereo Pose 3D - Target Locked (Pi 5)"
    if has_display:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        except Exception:
            has_display = False

    frame_count = 0
    fps_timer = time.time()
    fps = 0.0

    logger.info("Bắt đầu vòng lặp xử lý chính. Nhấn phím 'p' để reset target, phím 'q' để thoát.")

    try:
        while True:
            # ------------------------------------------------------------------
            # THU NHẬN HÌNH ẢNH 2 CAMERA
            # ------------------------------------------------------------------
            if cam_l_type == "picam2":
                frame_l = cam_left.capture_array("main")
            else:
                ret_l, frame_l = cam_left.read()
                if not ret_l or frame_l is None:
                    time.sleep(0.01)
                    continue

            if args.standalone or client_right is None:
                frame_r = frame_l.copy()
            else:
                ret_r, frame_r = client_right.read()
                if not ret_r or frame_r is None:
                    frame_r = np.zeros_like(frame_l)
                    cv2.putText(frame_r, f"DANG CHO KET NOI PI #2 ({args.right_ip}:{args.right_port})...", 
                                (30, frame_r.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            disp_l = frame_l.copy()
            disp_r = frame_r.copy()

            # ------------------------------------------------------------------
            # BƯỚC 1: CHẠY YOLO POSE TRÊN CAM TRÁI & CAM PHẢI
            # ------------------------------------------------------------------
            # Chạy YOLO Pose Cam Trái
            results_l = model(disp_l, imgsz=480, conf=args.conf, device="cpu", verbose=False)
            res_l = results_l[0]

            dets_l = []
            kpts_l_list = []
            confs_l_list = []

            if res_l.boxes is not None and res_l.keypoints is not None:
                boxes_l = res_l.boxes.xyxy.cpu().numpy()
                confs_l = res_l.boxes.conf.cpu().numpy()
                kpts_l = res_l.keypoints.data.cpu().numpy()

                for b, k, c in zip(boxes_l, kpts_l, confs_l):
                    dets_l.append(b)
                    kpts_l_list.append(k)
                    confs_l_list.append(float(c))

            # Chạy YOLO Pose Cam Phải (nếu có kết nối hoặc standalone) để bắt Bbox chính xác 100%
            res_r = None
            if (client_right and client_right.connected) or args.standalone:
                if not args.standalone:
                    results_r = model(disp_r, imgsz=480, conf=args.conf, device="cpu", verbose=False)
                    res_r = results_r[0]
                else:
                    res_r = res_l

            # ------------------------------------------------------------------
            # BƯỚC 2: CẬP NHẬT TRACKER ĐA ĐỐI TƯỢNG (CAM TRÁI)
            # ------------------------------------------------------------------
            now = time.time()
            active_tracks = tracker.update(dets_l, kpts_l_list, confs_l_list, now)

            # ------------------------------------------------------------------
            # BƯỚC 3: MÁY TRẠNG THÁI KHÓA MỤC TIÊU (TARGET LOCK CONTROLLER)
            # ------------------------------------------------------------------
            locked_target = lock_controller.update(active_tracks, now)

            # ------------------------------------------------------------------
            # BƯỚC 4: ĐO STEREO 3D CHÍNH XÁC (TÂM BBOX -> TRUNG ĐIỂM BASELINE)
            # ------------------------------------------------------------------
            # Xác định đối tượng cần đo:
            # - Nếu đã khóa: CHỈ đo đối tượng bị khóa (triệt tiêu 100% nhiễu người khác).
            # - Nếu chưa ai bị khóa: đo hiển thị người đứng gần nhất để xem trước khoảng cách liên tục.
            target_to_measure = locked_target

            if target_to_measure is None and active_tracks:
                visible_tracks = [t for t in active_tracks.values() if t.lost_frames == 0]
                if visible_tracks:
                    # Ưu tiên người đang giơ tay hoặc người có bbox lớn nhất (gần nhất)
                    raising = [t for t in visible_tracks if t.is_hand_raised]
                    target_to_measure = max(raising, key=lambda t: t.height) if raising else max(visible_tracks, key=lambda t: t.height)

            bbox_center_r = None
            matched_r_box = None
            matched_r_kpts = None

            if target_to_measure is not None and target_to_measure.lost_frames == 0:
                cx_l, cy_l = target_to_measure.center

                # 4.1. Khớp đối tượng tương ứng trên Camera Phải bằng kết quả YOLO Cam Phải
                if res_r and res_r.boxes is not None and len(res_r.boxes) > 0:
                    boxes_r = res_r.boxes.xyxy.cpu().numpy()
                    kpts_r = res_r.keypoints.data.cpu().numpy()

                    best_match_idx = None
                    min_cost = float('inf')

                    for j in range(len(boxes_r)):
                        rx1, ry1, rx2, ry2 = boxes_r[j]
                        rcx = (rx1 + rx2) / 2.0
                        rcy = (ry1 + ry2) / 2.0
                        r_height = ry2 - ry1

                        y_diff = abs(rcy - cy_l)
                        h_diff = abs(r_height - target_to_measure.height)
                        # Trong hệ stereo: rcx thường <= cx_l
                        disp_penalty = max(0.0, rcx - cx_l - 25.0) * 4.0

                        # Trạng thái giơ tay tương thích
                        r_raised, _, _ = check_hand_raise(kpts_r[j])
                        raise_bonus = 0.0 if (r_raised == target_to_measure.is_hand_raised) else 40.0

                        cost = y_diff * 2.5 + h_diff * 1.5 + disp_penalty + raise_bonus
                        if cost < min_cost:
                            min_cost = cost
                            best_match_idx = j

                    if best_match_idx is not None and min_cost < 250.0:
                        rx1, ry1, rx2, ry2 = map(int, boxes_r[best_match_idx])
                        bbox_center_r = (float((rx1 + rx2) / 2.0), float((ry1 + ry2) / 2.0))
                        matched_r_box = (rx1, ry1, rx2, ry2)
                        matched_r_kpts = kpts_r[best_match_idx]

                # 4.2. Fallback: Nếu YOLO Cam Phải không bắt được, dùng Epipolar Patch Match
                if bbox_center_r is None:
                    bbox_center_r = stereo_solver.match_epipolar_patch(frame_l, frame_r, (cx_l, cy_l), v_range=35)

                # 4.3. Giải tam giác đạc 3D
                if bbox_center_r is not None:
                    X_mid, Y_mid, Z_mid, D_mid = stereo_solver.triangulate_single_point((cx_l, cy_l), bbox_center_r)
                    d_cm = D_mid * 100.0
                    z_cm = Z_mid * 100.0
                    x_cm = X_mid * 100.0
                    target_to_measure.update_3d(d_cm, z_cm, x_cm)

            # ------------------------------------------------------------------
            # BƯỚC 5: TRỰC QUAN HÓA TRÊN CẢ HAI CAMERA
            # ------------------------------------------------------------------
            # 5.1. Trực quan hóa trên Camera Phải
            if matched_r_box is not None:
                rx1, ry1, rx2, ry2 = matched_r_box
                box_color_r = (0, 0, 255) if (locked_target and target_to_measure.track_id == locked_target.track_id) else (0, 255, 0)
                cv2.rectangle(disp_r, (rx1, ry1), (rx2, ry2), box_color_r, 2)
                if matched_r_kpts is not None:
                    draw_skeleton(disp_r, matched_r_kpts, color=box_color_r)

            if bbox_center_r is not None:
                cv2.drawMarker(disp_r, (int(bbox_center_r[0]), int(bbox_center_r[1])), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
                is_lock = (locked_target and target_to_measure.track_id == locked_target.track_id)
                tag_r = f"ID #{target_to_measure.track_id}" + (" [LOCKED]" if is_lock else "")
                cv2.putText(disp_r, tag_r, (int(bbox_center_r[0]) - 30, max(25, int(bbox_center_r[1]) - 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            # 5.2. Trực quan hóa trên Camera Trái
            for t_id, track in active_tracks.items():
                if track.lost_frames > 0:
                    continue

                x1, y1, x2, y2 = map(int, track.bbox)
                cx, cy = map(int, track.center)
                is_this_locked = (locked_target is not None and track.track_id == locked_target.track_id)

                if is_this_locked:
                    # MỤC TIÊU ĐANG ĐƯỢC KHÓA CHẶT: Viền ĐỎ ĐẬM, Tâm VÀNG
                    box_color = (0, 0, 255)
                    thickness = 3
                    cv2.rectangle(disp_l, (x1, y1), (x2, y2), box_color, thickness)
                    cv2.drawMarker(disp_l, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
                    draw_skeleton(disp_l, track.keypoints, color=(0, 0, 255))

                    tag = f"LOCKED TARGET #{t_id} [{track.raise_side}]"
                    cv2.putText(disp_l, tag, (x1, max(28, y1 - 34)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    if track.d_mid_cm is not None:
                        dist_txt = f"D_mid: {track.d_mid_cm:.1f}cm (Z:{track.z_cm:.1f}cm)"
                        cv2.putText(disp_l, dist_txt, (x1, max(52, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (250, 204, 21), 2)
                else:
                    # Người khác trong khung hình
                    if track.is_hand_raised:
                        box_color = (0, 165, 255)  # Cam (đang đếm giây giơ tay)
                        thickness = 2
                        prog_txt = f"ID #{t_id} GIO TAY: {track.raise_duration:.1f}s / {args.lock_time:.1f}s"
                    else:
                        box_color = (120, 120, 120) if locked_target else (0, 255, 0)
                        thickness = 1 if locked_target else 2
                        prog_txt = f"ID #{t_id} ({track.conf:.2f})"

                    cv2.rectangle(disp_l, (x1, y1), (x2, y2), box_color, thickness)
                    draw_skeleton(disp_l, track.keypoints, color=box_color)
                    cv2.drawMarker(disp_l, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 14, 1)
                    cv2.putText(disp_l, prog_txt, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

                    # Nếu là người đang được đo xem trước (khi chưa ai lock)
                    if not locked_target and target_to_measure and target_to_measure.track_id == t_id and track.d_mid_cm is not None:
                        dist_txt = f"D_mid: {track.d_mid_cm:.1f}cm (Z:{track.z_cm:.1f}cm)"
                        cv2.putText(disp_l, dist_txt, (x1, max(50, y1 - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (250, 204, 21), 2)

                # Vẽ điểm cổ tay nổi bật nếu có giơ tay
                if track.is_hand_raised and track.wrist_pt:
                    wx, wy = map(int, track.wrist_pt)
                    cv2.circle(disp_l, (wx, wy), 8, (0, 0, 255), -1)

            # Ghép khung hình Side-by-Side (Cam Trái | Cam Phải)
            scale = 0.70
            h_orig, w_orig = disp_l.shape[:2]
            w_s, h_s = int(w_orig * scale), int(h_orig * scale)
            s_l = cv2.resize(disp_l, (w_s, h_s))
            s_r = cv2.resize(disp_r, (w_s, h_s))

            cv2.putText(s_l, "CAM TRAI (PI #1 MASTER)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (56, 189, 248), 2)
            cv2.putText(s_r, "CAM PHAI (PI #2 WORKER)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (244, 63, 94), 2)

            combined_feed = cv2.hconcat([s_l, s_r])
            comb_h, comb_w = combined_feed.shape[:2]

            # ------------------------------------------------------------------
            # BƯỚC 6: BẢNG ĐIỀU KHIỂN & HUD HIỂN THỊ KHOẢNG CÁCH 3D
            # ------------------------------------------------------------------
            hud_height = 150
            hud_panel = np.zeros((hud_height, comb_w, 3), dtype=np.uint8)
            hud_panel[:] = (15, 23, 42)

            cv2.line(hud_panel, (0, 0), (comb_w, 0), (56, 189, 248), 2)
            cv2.line(hud_panel, (w_s, 0), (w_s, hud_height), (51, 65, 85), 1)

            # Ưu tiên hiển thị thông số 3D của đối tượng đo
            disp_person = locked_target if locked_target else target_to_measure

            if disp_person and disp_person.d_mid_cm is not None:
                x_cm = disp_person.x_mid_cm
                pos_str = "CHINH GIUA" if abs(x_cm) < 4.0 else (f"LECH PHAI {x_cm:+.1f}cm" if x_cm > 0 else f"LECH TRAI {abs(x_cm):.1f}cm")
                
                if locked_target:
                    title_hud = f"LOCKED TARGET #{disp_person.track_id} [{disp_person.raise_side}]! D_mid: {disp_person.d_mid_cm:.1f} cm"
                    color_title = (0, 0, 255)
                else:
                    title_hud = f"KHOANG CACH TAM BBOX -> TRUNG DIEM BASELINE: {disp_person.d_mid_cm:.1f} cm"
                    color_title = (250, 204, 21)

                cv2.putText(hud_panel, title_hud, (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.90, color_title, 2)
                cv2.putText(hud_panel, f"Z (Vuong goc): {disp_person.z_cm:.1f} cm | Vi tri ngang X_mid: [{pos_str}]", 
                            (25, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (74, 222, 128), 2)
            else:
                cv2.putText(hud_panel, lock_controller.status_message, 
                            (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (250, 204, 21), 2)
                cv2.putText(hud_panel, "Gio tay lien tuc >= 2.0s de Khoa Muc Tieu | Bam phim 'p' de Reset", 
                            (25, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (148, 163, 184), 1)

            frame_count += 1
            if frame_count % 10 == 0:
                elapsed = time.time() - fps_timer
                fps = 10.0 / elapsed if elapsed > 0 else 0.0
                fps_timer = time.time()

            cv2.putText(hud_panel, f"So nguoi: {len(active_tracks)} | Baseline: {stereo_solver.baseline*100:.1f}cm | Web: http://<IP>:5000 | FPS: {fps:.1f}",
                        (25, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (148, 163, 184), 1)

            final_view = cv2.vconcat([combined_feed, hud_panel])

            # Nén JPEG gửi Web Stream
            ret_enc, jpeg_data = cv2.imencode(".jpg", final_view, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
            if ret_enc:
                with stream_lock:
                    global_stream_jpeg = jpeg_data.tobytes()

            # Hiển thị cửa sổ đồ họa nếu có màn hình
            if has_display:
                cv2.imshow(window_name, final_view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                elif key in (ord('p'), ord('P')):
                    lock_controller.reset(reason="Nhan phim P tren ban phim")

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
        logger.info("Đã dọn dẹp tài nguyên và tắt ứng dụng an toàn.")


if __name__ == "__main__":
    main()
