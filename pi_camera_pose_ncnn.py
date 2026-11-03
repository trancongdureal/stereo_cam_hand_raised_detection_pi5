import os
import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading

# ============================================================
# 1. CẤU HÌNH & KHỞI TẠO MÔ HÌNH NCNN
# ============================================================
# Tự động ưu tiên mô hình NCNN (đã tối ưu hóa cho ARM CPU Pi 5) nếu có
if os.path.exists("yolo26n-pose_ncnn_model"):
    MODEL_PATH = "yolo26n-pose_ncnn_model"
elif os.path.exists("yolov26n-pose_ncnn_model"):
    MODEL_PATH = "yolov26n-pose_ncnn_model"
elif os.path.exists("yolo26n-pose.pt"):
    MODEL_PATH = "yolo26n-pose.pt"
else:
    MODEL_PATH = "yolov26n-pose.pt"

print(f"[INFO] Đang khởi tạo mô hình: {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
DEVICE = "cpu"  # Pi 5 chạy trên CPU ARM64

LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_WRIST, RIGHT_WRIST = 9, 10

SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16)
]

def draw_person(frame, bbox, keypoints, person_conf):
    x1, y1, x2, y2 = map(int, bbox)
    raised_hand = False

    # Tay trái
    if keypoints[LEFT_WRIST][2] > 0.5 and keypoints[LEFT_SHOULDER][2] > 0.5:
        if keypoints[LEFT_WRIST][1] < keypoints[LEFT_SHOULDER][1]:
            raised_hand = True

    # Tay phải
    if keypoints[RIGHT_WRIST][2] > 0.5 and keypoints[RIGHT_SHOULDER][2] > 0.5:
        if keypoints[RIGHT_WRIST][1] < keypoints[RIGHT_SHOULDER][1]:
            raised_hand = True

    bbox_color = (0, 0, 255) if raised_hand else (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)

    cv2.putText(
        frame, f"Person {person_conf:.2f}",
        (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, bbox_color, 2, cv2.LINE_AA
    )

    if raised_hand:
        cv2.putText(
            frame, "HAND RAISED",
            (x1, max(45, y1 - 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA
        )

    for a, b in SKELETON:
        xa, ya, ca = keypoints[a]
        xb, yb, cb = keypoints[b]
        if ca > 0.25 and cb > 0.25:
            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), (255, 255, 0), 2, cv2.LINE_AA)

    for idx in range(5, len(keypoints)):
        x, y, conf = keypoints[idx]
        if conf > 0.25:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1)

    return raised_hand

# ============================================================
# 2. BỘ ĐỆM FRAME ĐỂ STREAM WEB (XEM TỪ LAPTOP QUA TRÌNH DUYỆT)
# ============================================================
latest_jpeg = None
lock = threading.Lock()

class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global latest_jpeg
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            html = """
            <html>
                <head>
                    <title>Pi 5 - Hand Raise Detection Stream</title>
                    <style>
                        body { background: #111; color: #eee; font-family: sans-serif; text-align: center; margin: 0; padding: 20px; }
                        h2 { margin-bottom: 10px; }
                        img { border-radius: 8px; border: 2px solid #444; max-width: 90vw; }
                    </style>
                </head>
                <body>
                    <h2>Raspberry Pi 5: Hand Raise Detection (NCNN)</h2>
                    <img src="/stream.mjpg" />
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
                    with lock:
                        if latest_jpeg is None:
                            continue
                        frame_data = latest_jpeg
                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(frame_data)))
                    self.end_headers()
                    self.wfile.write(frame_data)
                    self.wfile.write(b'\r\n')
                    time.sleep(0.03)
            except Exception:
                pass
        else:
            self.send_error(404)

def start_web_server(port=5000):
    server = HTTPServer(('0.0.0.0', port), StreamingHandler)
    server.serve_forever()

# ============================================================
# 3. KHỞI TẠO CAMERA (PICAMERA2 HOẶC OPENCV)
# ============================================================
def init_camera():
    # Ưu tiên Picamera2 nếu có (chuẩn tối ưu cho Pi Camera v2 trên Pi 5)
    try:
        from picamera2 import Picamera2
        print("[INFO] Đang khởi tạo Pi Camera v2 qua Picamera2...")
        picam2 = Picamera2()
        config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
        picam2.configure(config)
        picam2.start()
        print("[INFO] Pi Camera v2 khởi tạo thành công.")
        return ("picam2", picam2)
    except Exception as e:
        print(f"[WARN] Không thể dùng Picamera2 ({e}). Thử cv2.VideoCapture...")

    # Fallback sang OpenCV VideoCapture (V4L2)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("[INFO] OpenCV VideoCapture khởi tạo thành công.")
        return ("opencv", cap)

    raise RuntimeError("Lỗi: Không tìm thấy camera nào trên Pi 5!")

# ============================================================
# 4. CHƯƠNG TRÌNH CHÍNH
# ============================================================
def main():
    global latest_jpeg

    # Bật web stream chạy ngầm trên port 5000
    server_thread = threading.Thread(target=start_web_server, args=(5000,), daemon=True)
    server_thread.start()
    print("[INFO] Web Stream đã chạy tại: http://<IP_CỦA_PI>:5000")

    cam_type, cam = init_camera()
    has_display = os.environ.get("DISPLAY") is not None

    frame_count = 0
    fps_timer = time.time()
    fps = 0.0

    print("[INFO] Bắt đầu nhận diện. Nhấn Ctrl+C để dừng.")

    try:
        while True:
            if cam_type == "picam2":
                # Lưu ý kỹ thuật: Trong libcamera / Picamera2, định dạng 'RGB888' quy định
                # thứ tự byte trong bộ nhớ là [B, G, R], trùng khớp hoàn toàn với chuẩn BGR của OpenCV.
                # Không chuyển cv2.cvtColor(..., cv2.COLOR_RGB2BGR) vì sẽ đảo ngược R và B (khiến da người bị ám xanh).
                frame = cam.capture_array()
            else:
                ret, frame = cam.read()
                if not ret or frame is None:
                    continue

            # Inference YOLO Pose (imgsz=480)
            results = model(
                frame,
                imgsz=480,
                conf=0.25,
                device=DEVICE,
                verbose=False
            )
            result = results[0]

            person_count = 0
            raised_count = 0

            if result.boxes is not None and result.keypoints is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                person_confs = result.boxes.conf.cpu().numpy()
                kpts = result.keypoints.data.cpu().numpy()

                person_count = len(boxes)
                for i in range(len(boxes)):
                    raised = draw_person(frame, boxes[i], kpts[i], float(person_confs[i]))
                    if raised:
                        raised_count += 1

            # Tính FPS
            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_timer = time.time()

            # Hiển thị thông số
            cv2.rectangle(frame, (0, 0), (260, 70), (0, 0, 0), -1)
            cv2.putText(frame, f"Persons: {person_count}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Raised: {raised_count} | FPS: {fps:.1f}", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

            # Nén JPEG cho Web Stream
            success, jpeg_bytes = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
            if success:
                with lock:
                    latest_jpeg = jpeg_bytes.tobytes()

            # Nếu có màn hình đồ họa trực tiếp
            if has_display:
                cv2.imshow("Hand Raise Detection - Pi 5", frame)
                if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Nhận tín hiệu dừng từ người dùng.")
    finally:
        if cam_type == "picam2":
            cam.stop()
        else:
            cam.release()
        if has_display:
            cv2.destroyAllWindows()
        print("[INFO] Đã giải phóng camera và tắt chương trình.")

if __name__ == "__main__":
    main()
