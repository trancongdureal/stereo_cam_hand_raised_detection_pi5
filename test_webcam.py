import time
import cv2
import torch
from ultralytics import YOLO

import os

# ============================================================
# CẤU HÌNH & KHỞI TẠO MÔ HÌNH YOLOV26-POSE
# ============================================================
MODEL_PATH = "yolov26n-pose.pt" if os.path.exists("yolov26n-pose.pt") else "yolo26n-pose.pt"
print(f"Đang khởi tạo mô hình YOLOv26n-Pose ({MODEL_PATH})...")
model = YOLO(MODEL_PATH)  # Tự động tải nếu chưa có
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print(f"Thiết bị tính toán: {'GPU' if torch.cuda.is_available() else 'CPU'}")
print("Mô hình đã sẵn sàng.")

# ============================================================
# COCO KEYPOINT INDEX (GIỮ NGUYÊN)
# ============================================================
LEFT_SHOULDER  = 5
RIGHT_SHOULDER = 6
LEFT_WRIST  = 9
RIGHT_WRIST = 10

# ============================================================
# COCO SKELETON CONNECTIONS (ĐÃ LOẠI BỎ KHỚP TRÊN MẶT: 0-4)
# ============================================================
SKELETON = [
    # Bỏ các liên kết mặt: (0, 1), (0, 2), (1, 3), (2, 4)
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16)
]

# ============================================================
# HÀM XỬ LÝ & TÍNH TOÁN DỰ ĐOÁN (GIỮ NGUYÊN CẤU TRÚC TÍNH TOÁN)
# ============================================================
def draw_person(frame, bbox, keypoints, person_conf):
    """
    bbox: [x1, y1, x2, y2]
    keypoints: [17, 3] -> x, y, confidence
    """
    x1, y1, x2, y2 = map(int, bbox)

    # --------------------------------------------------------
    # Xác định giơ tay
    # --------------------------------------------------------
    raised_hand = False

    # Tay trái
    if (
        keypoints[LEFT_WRIST][2] > 0.5
        and keypoints[LEFT_SHOULDER][2] > 0.5
    ):
        wrist_y = keypoints[LEFT_WRIST][1]
        shoulder_y = keypoints[LEFT_SHOULDER][1]
        if wrist_y < shoulder_y:
            raised_hand = True

    # Tay phải
    if (
        keypoints[RIGHT_WRIST][2] > 0.5
        and keypoints[RIGHT_SHOULDER][2] > 0.5
    ):
        wrist_y = keypoints[RIGHT_WRIST][1]
        shoulder_y = keypoints[RIGHT_SHOULDER][1]
        if wrist_y < shoulder_y:
            raised_hand = True

    # --------------------------------------------------------
    # Màu Bounding Box
    # --------------------------------------------------------
    if raised_hand:
        bbox_color = (0, 0, 255)  # Đỏ (BGR)
    else:
        bbox_color = (0, 255, 0)  # Xanh lá

    # Vẽ Bounding Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)

    # Độ tin cậy người
    conf_text = f"Person {person_conf:.2f}"
    cv2.putText(
        frame,
        conf_text,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        bbox_color,
        2,
        cv2.LINE_AA
    )

    # Nhãn HAND RAISED nếu giơ tay
    if raised_hand:
        label_y = max(45, y1 - 28)
        cv2.putText(
            frame,
            "HAND RAISED",
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    # Vẽ khung xương
    for a, b in SKELETON:
        xa, ya, ca = keypoints[a]
        xb, yb, cb = keypoints[b]
        if ca > 0.25 and cb > 0.25:
            xa, ya = int(xa), int(ya)
            xb, yb = int(xb), int(yb)
            cv2.line(frame, (xa, ya), (xb, yb), (255, 255, 0), 2, cv2.LINE_AA)

    # Vẽ các khớp nối (Bỏ qua 5 điểm trên mặt: 0=mũi, 1,2=mắt, 3,4=tai)
    for idx in range(5, len(keypoints)):
        x, y, conf = keypoints[idx]
        if conf > 0.25:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1)

    return raised_hand

# ============================================================
# CHẠY TEST WEBCAM TRỰC TIẾP TRÊN LAPTOP
# ============================================================
def main():
    print("Đang mở webcam laptop...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Không thể mở webcam index 0, đang thử index 1...")
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Lỗi: Không tìm thấy hoặc không thể kết nối tới webcam laptop!")
        return

    # Thiết lập độ phân giải khung hình
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Webcam đã hoạt động. Nhấn phím 'q' hoặc 'ESC' trên cửa sổ hình ảnh để dừng.")

    frame_count = 0
    fps_timer = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Không nhận được tín hiệu hình ảnh từ webcam.")
            break

        # Inference mô hình YOLOv26 Pose (giữ nguyên tham số)
        results = model(
            frame,
            imgsz=640,
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
                bbox = boxes[i]
                person_conf = float(person_confs[i])
                person_keypoints = kpts[i]

                raised = draw_person(
                    frame,
                    bbox,
                    person_keypoints,
                    person_conf
                )

                if raised:
                    raised_count += 1

        # Tính FPS
        frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_timer = time.time()

        # Bảng hiển thị thông tin
        cv2.rectangle(frame, (0, 0), (260, 70), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"Persons: {person_count}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            f"Raised: {raised_count} | FPS: {fps:.1f}",
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Hiển thị cửa sổ webcam
        cv2.imshow("Hand Raise Detection - YOLOv26 Pose", frame)

        # Nhấn phím 'q' hoặc phím ESC (27) để thoát
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("Đã nhận lệnh thoát từ bàn phím.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Đã tắt webcam và kết thúc chương trình.")

if __name__ == "__main__":
    main()
