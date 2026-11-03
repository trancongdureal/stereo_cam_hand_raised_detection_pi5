#!/usr/bin/env python3
"""
camera_worker_server.py
Chạy trên Raspberry Pi #2 (Worker - Camera Phải).
Nhiệm vụ:
  - Khởi tạo Camera Pi v2 bằng Picamera2 (chuẩn màu BGR).
  - Mở cổng TCP 5555 để stream khung hình độ trễ cực thấp sang Pi #1 (Master).
"""

import sys
import time
import socket
import struct
import logging
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("CameraWorker")

HOST = '0.0.0.0'
PORT = 5555
WIDTH = 1280
HEIGHT = 720
JPEG_QUALITY = 85

def init_picamera2():
    try:
        from picamera2 import Picamera2
        logger.info(f"Đang khởi tạo Picamera2 trên Pi #2 ({WIDTH}x{HEIGHT})...")
        try:
            picam2 = Picamera2(0)
        except TypeError:
            picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (WIDTH, HEIGHT), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        logger.info("Picamera2 trên Pi #2 đã sẵn sàng hoạt động.")
        return picam2
    except Exception as e:
        logger.error(f"Lỗi khởi tạo Picamera2: {e}")
        logger.info("Thử chuyển sang fallback OpenCV VideoCapture(0)...")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        if not cap.isOpened():
            logger.error("Không tìm thấy camera nào trên Pi #2!")
            sys.exit(1)
        return cap

def main():
    logger.info("===========================================")
    logger.info("  PI #2 - CAMERA WORKER SERVER (CAM PHẢI)  ")
    logger.info("===========================================")

    cam = init_picamera2()
    is_picam2 = hasattr(cam, 'capture_array')

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((HOST, PORT))
    server_sock.listen(2)
    logger.info(f"Server đang lắng nghe kết nối tại cổng TCP {PORT}...")

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

    try:
        while True:
            logger.info("Đang chờ Pi #1 (Master) kết nối...")
            conn, addr = server_sock.accept()
            logger.info(f"Pi #1 đã kết nối từ địa chỉ: {addr}")
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            try:
                while True:
                    if is_picam2:
                        # Trong Picamera2 với format RGB888, mảng trả về đã là BGR chuẩn OpenCV
                        frame = cam.capture_array("main")
                    else:
                        ret, frame = cam.read()
                        if not ret:
                            continue

                    # Nén JPEG nhẹ và gửi qua TCP
                    ret, jpeg = cv2.imencode('.jpg', frame, encode_param)
                    if not ret:
                        continue

                    data = jpeg.tobytes()
                    # Header 4 byte độ dài + dữ liệu ảnh
                    header = struct.pack(">I", len(data))
                    conn.sendall(header + data)

            except (BrokenPipeError, ConnectionResetError):
                logger.warning(f"Pi #1 ({addr}) đã ngắt kết nối.")
            finally:
                conn.close()

    except KeyboardInterrupt:
        logger.info("Đang dừng server theo yêu cầu người dùng...")
    finally:
        server_sock.close()
        if is_picam2:
            cam.stop()
            cam.close()
        else:
            cam.release()
        logger.info("Đã tắt camera và đóng server trên Pi #2.")

if __name__ == "__main__":
    main()
