# SPECIFICATION: TARGET LOCKING & STEREO POSE PIPELINE

Tài liệu kỹ thuật tổng hợp bài toán từ `kế hoạch.txt` và kiến trúc luồng dữ liệu (Flow) của 2 file mã nguồn mới (`stereo_pose_target_locked.py` và `test_stereo_target_locked.py`).

---

## 1. Yêu cầu nghiệp vụ (Trích xuất từ `kế hoạch.txt`)

* **Bài toán gốc:** Khi có từ 2 người trở lên trong khung hình, code stereo cũ bị nhảy bounding box liên tục giữa các frame và ghép chéo sai giữa Cam Trái - Cam Phải, dẫn đến lỗi tính khoảng cách 3D.
* **Quy tắc giải quyết:**
  1. **Điều kiện Khóa (Lock Target):** Một người phải giơ tay liên tục $\ge 2.0$ giây. Nếu có nhiều người cùng thỏa mãn, hệ thống tự động chọn người **đứng gần camera nhất** (chiều cao bounding box lớn nhất / khoảng cách $Z$ nhỏ nhất).
  2. **Trạng thái Khóa chặt (Locked):** Khi đã khóa một người, hệ thống **chỉ đo khoảng cách Stereo 3D của riêng người này**; toàn bộ người khác trong khung hình bị bỏ qua để triệt tiêu 100% xung đột dữ liệu.
  3. **Tự động nhả khóa (Auto-Unlock):**
     * Khi người bị khóa hạ tay xuống (debounce $> 0.4$s) $\rightarrow$ Tự động nhả khóa để tìm người khác.
     * Khi người bị khóa đi ra khỏi màn hình (mất dấu $> 5$ frame) $\rightarrow$ Tự động nhả khóa.
  4. **Can thiệp thủ công (Manual Reset):** Nhấn phím `p` trên bàn phím (hoặc nút bấm trên Web UI) $\rightarrow$ Reset ngay lập tức về trạng thái ban đầu để quét lại người gần nhất.

---

## 2. Kiến trúc & Flow của `stereo_pose_target_locked.py`

File chạy chính trên Raspberry Pi 5 (Master Pi #1) kết nối với Pi #2 (Worker Cam Phải).

### Sơ đồ luồng (Pipeline Flow)

```
[Cam Trái Pi #1] ──► YOLO Pose (Single-Pass) ──► Lightweight IoU Tracker ──► TargetLockController
                                                                                     │
[Cam Phải Pi #2] ──► Epipolar Match (<1.5ms) ◄── [CHỈ ĐỐI VỚI LOCKED TARGET] ◄──────┘
                            │
                            ▼
              StereoTriangulator (3D: D_mid, Z, X_mid)
                            │
                            ▼
           Side-by-Side Visualizer + HUD + Web Stream (:5000)
```

### Chi tiết các module:

1. **Tối ưu hóa phần cứng Pi 5 (High Performance):**
   - Chỉ chạy YOLO Pose duy nhất trên Cam Trái (imgsz=480). Không chạy YOLO trên Cam Phải để giảm 50% tải CPU.
   - Tự động ưu tiên nạp mô hình NCNN (`yolo26n-pose_ncnn_model`) nếu có.
2. **`LightweightTracker`:**
   - Thuật toán bám vết dựa trên IoU matching, chi phí tính toán $< 0.1$ms.
   - Gán `track_id` cố định cho từng người qua các frame, theo dõi biến `raise_duration` (thời gian giơ tay liên tục).
3. **`TargetLockController` (Máy trạng thái):**
   - `SEARCHING`: Quét ứng viên, đếm thời gian giơ tay $0 \rightarrow 2.0$s.
   - `LOCKED`: Khóa chặt `locked_track_id`. Chuyển màu bbox sang ĐỎ ĐẬM, vẽ tâm vàng.
   - `UNLOCK`: Kích hoạt khi hạ tay, mất dấu ra ngoài biên, hoặc nhận tín hiệu phím `p`/Web `/reset`.
4. **`StereoTriangulator`:**
   - Dùng ma trận hiệu chuẩn dời gốc tọa độ về trung điểm baseline: $M = (B/2, 0, 0)$.
   - `fast_epipolar_match`: Tìm điểm tương ứng trên Cam Phải bằng template matching dọc dải epipolar (chỉ quét cho mục tiêu đã lock).
   - Bộ lọc EMA ($\alpha = 0.35$) làm mịn $D_{mid}, Z, X_{mid}$ hiển thị lên HUD.
5. **Giao diện & Điều khiển:**
   - Ghép Side-by-Side (Cam Trái | Cam Phải) + Bảng HUD hiển thị trạng thái và khoảng cách.
   - Web Server tại cổng 5000: Có tích hợp endpoint `/reset` và nút bấm trực quan để reset từ xa khi chạy headless qua SSH.

---

## 3. Kiến trúc & Flow của `test_stereo_target_locked.py`

File kiểm thử tự động toàn diện, chạy độc lập mà không cần phần cứng camera hay Raspberry Pi.

### Danh mục 9 kịch bản kiểm thử:

| Test Case | Mục tiêu kiểm chứng | Kết quả |
| :--- | :--- | :--- |
| **TEST 1** | Hàm `check_hand_raise`: phát hiện chính xác cổ tay vượt trên vai. | PASS |
| **TEST 2** | `LightweightTracker`: Giữ nguyên Track ID khi YOLO đổi thứ tự trả về của 2 người. | PASS |
| **TEST 3** | Điều kiện thời gian: Giơ tay $< 2.0$s (1.5s) hệ thống không kích hoạt khóa. | PASS |
| **TEST 4** | Điều kiện kích hoạt: Giơ tay $\ge 2.0$s $\rightarrow$ Khóa chặt đúng ID đối tượng. | PASS |
| **TEST 5** | Cách ly nhiễu: Người thứ 2 di chuyển/giơ tay không làm ảnh hưởng đến người đang lock. | PASS |
| **TEST 6** | Tự động nhả khóa khi người bị lock hạ tay xuống quá ngưỡng debounce (0.4s). | PASS |
| **TEST 7** | Tự động nhả khóa khi người bị lock biến mất khỏi khung hình ($> 5$ frame). | PASS |
| **TEST 8** | Chức năng gọi `reset()` (phím `p` / API Web) xóa ngay lập tức trạng thái khóa. | PASS |
| **TEST 9** | Benchmark Stereo 3D: Tính toán tam giác đạc và dải cực phẳng đạt $< 0.05$ms/frame. | PASS |

---

## 4. Hướng dẫn thực thi cho Agent / Developer kế thừa

- **Chạy bộ kiểm thử tự động:**
  ```bash
  python test_stereo_target_locked.py
  ```
- **Chạy pipeline thực tế (Pi 5 hoặc Laptop):**
  ```bash
  # Chế độ thử nghiệm 1 camera / standalone:
  python stereo_pose_target_locked.py --standalone

  # Chế độ thực tế với Pi #2:
  python3 stereo_pose_target_locked.py --right_ip 192.168.88.2 --right_port 5555
  ```
- **Phím tắt điều khiển:**
  - `p` / `P`: Reset mục tiêu về trạng thái tìm kiếm ban đầu.
  - `q` / `ESC`: Thoát ứng dụng an toàn.
  - Web UI: `http://<IP>:5000` (có nút Reset tương đương phím `p`).

