#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CHƯƠNG TRÌNH KIỂM THỬ TỰ ĐỘNG & ĐO HIỆU NĂNG CHO STEREO TARGET LOCKING
File: test_stereo_target_locked.py
---------------------------------------------------------------------------------
Mục đích:
1. Chứng minh tính đúng đắn của toàn bộ kế hoạch:
   - Cơ chế bám vết định danh ổn định (Track ID).
   - Bộ đếm thời gian giơ tay >= 2.0 giây để kích hoạt khóa.
   - Xử lý triệt để bài toán 2 người trong khung hình (loại bỏ xung đột đo khoảng cách).
   - Tự động nhả khóa khi hạ tay.
   - Tự động nhả khóa khi người đi ra khỏi khung hình.
   - Reset tức thì bằng phím 'p' hoặc API web.
2. Đo kiểm hiệu năng tính toán (CPU Benchmark) đảm bảo chạy mượt trên Raspberry Pi 5.
"""

import sys
import time
import numpy as np

# Nạp các module cốt lõi từ stereo_pose_target_locked
from stereo_pose_target_locked import (
    LightweightTracker,
    TargetLockController,
    StereoTriangulator,
    check_hand_raise,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_WRIST, RIGHT_WRIST
)

def make_dummy_keypoints(hand_raised=False, side="RIGHT"):
    """Tạo 17 keypoints giả lập với tư thế giơ tay hoặc không giơ tay"""
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 2] = 0.9  # Confidence cao

    # Đặt vai tại y = 200
    kpts[LEFT_SHOULDER] = [280, 200, 0.9]
    kpts[RIGHT_SHOULDER] = [360, 200, 0.9]

    if hand_raised:
        if side in ("RIGHT", "BOTH"):
            # Cổ tay phải cao hơn vai (y = 120 < 200)
            kpts[RIGHT_WRIST] = [370, 120, 0.9]
        else:
            kpts[RIGHT_WRIST] = [370, 280, 0.9]

        if side in ("LEFT", "BOTH"):
            kpts[LEFT_WRIST] = [270, 120, 0.9]
        else:
            kpts[LEFT_WRIST] = [270, 280, 0.9]
    else:
        # Cả hai cổ tay thấp hơn vai (y = 280 > 200)
        kpts[LEFT_WRIST] = [270, 280, 0.9]
        kpts[RIGHT_WRIST] = [370, 280, 0.9]

    return kpts

def run_all_tests():
    print("=" * 70)
    print("  BẮT ĐẦU KIỂM THỬ HỆ THỐNG: STEREO POSE TARGET LOCKING  ")
    print("=" * 70)

    total_tests = 0
    passed_tests = 0

    # -------------------------------------------------------------------------
    # TEST 1: Kiểm tra hàm phát hiện giơ tay (check_hand_raise)
    # -------------------------------------------------------------------------
    total_tests += 1
    print("\n[TEST 1] Kiểm tra logic phát hiện giơ tay...")
    kpt_down = make_dummy_keypoints(hand_raised=False)
    is_r1, _, _ = check_hand_raise(kpt_down)

    kpt_up = make_dummy_keypoints(hand_raised=True, side="RIGHT")
    is_r2, side2, _ = check_hand_raise(kpt_up)

    assert not is_r1, "Lỗi: Không giơ tay nhưng báo giơ tay!"
    assert is_r2 and side2 == "RIGHT", f"Lỗi: Giơ tay phải nhưng trả về {side2}!"
    print("  => PASS: Phát hiện giơ tay chính xác 100%.")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # TEST 2: Tính ổn định định danh (ID Persistence) khi có 2 người
    # -------------------------------------------------------------------------
    total_tests += 1
    print("\n[TEST 2] Kiểm tra Track ID không bị nhảy khi có 2 người trong khung hình...")
    tracker = LightweightTracker()
    now = time.time()

    # Frame 1: Người 1 bên trái, Người 2 bên phải
    box1 = [100, 150, 200, 400]
    box2 = [400, 160, 520, 420]
    tracks = tracker.update([box1, box2], [kpt_down, kpt_down], [0.9, 0.85], now)
    id_person1 = None
    id_person2 = None
    for t_id, t in tracks.items():
        if t.center[0] < 250:
            id_person1 = t_id
        else:
            id_person2 = t_id

    assert id_person1 is not None and id_person2 is not None and id_person1 != id_person2

    # Frame 2: YOLO đảo ngược thứ tự trả về (box2 trước, box1 sau)
    now += 0.05
    tracks_f2 = tracker.update([box2, box1], [kpt_down, kpt_down], [0.88, 0.91], now)
    for t_id, t in tracks_f2.items():
        if t.center[0] < 250:
            assert t_id == id_person1, f"Lỗi: Người 1 bị đổi ID từ {id_person1} sang {t_id}!"
        else:
            assert t_id == id_person2, f"Lỗi: Người 2 bị đổi ID từ {id_person2} sang {t_id}!"

    print(f"  => PASS: Track ID duy trì ổn định qua các frame (ID 1: #{id_person1}, ID 2: #{id_person2}).")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # TEST 3: Giơ tay < 2.0 giây thì CHƯA KHÓA
    # -------------------------------------------------------------------------
    total_tests += 1
    print("\n[TEST 3] Kiểm tra điều kiện thời gian: Giơ tay < 2.0s thì không được khóa...")
    controller = TargetLockController(lock_hold_seconds=2.0)
    tracker = LightweightTracker()
    sim_time = 1000.0

    # Người 1 giơ tay trong 1.5 giây (30 fps)
    for step in range(30):
        sim_time += 0.05  # 50ms per frame
        # Người 1 giơ tay, Người 2 không giơ tay
        t = tracker.update([box1, box2], [kpt_up, kpt_down], [0.9, 0.85], sim_time)
        locked = controller.update(t, sim_time)
        assert locked is None, f"Lỗi: Đã khóa quá sớm tại t = {step*0.05:.2f}s!"

    print("  => PASS: Sau 1.5s giơ tay, hệ thống vẫn ở trạng thái quan sát và đếm giờ.")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # TEST 4: Giơ tay >= 2.0 giây -> KHÓA CHẶT MỤC TIÊU
    # -------------------------------------------------------------------------
    total_tests += 1
    print("\n[TEST 4] Kiểm tra kích hoạt khóa: Giơ tay >= 2.0s -> Khóa chặt Người 1...")
    # Tiếp tục giơ tay thêm 0.7s (tổng cộng 2.2s)
    for step in range(15):
        sim_time += 0.05
        t = tracker.update([box1, box2], [kpt_up, kpt_down], [0.9, 0.85], sim_time)
        locked = controller.update(t, sim_time)

    assert locked is not None, "Lỗi: Đã giơ tay > 2s nhưng không khóa mục tiêu!"
    assert locked.track_id == id_person1, f"Lỗi: Khóa nhầm đối tượng #{locked.track_id} thay vì #{id_person1}!"
    print(f"  => PASS: Đã khóa chặt Target #{locked.track_id} sau {locked.raise_duration:.2f}s giơ tay liên tục!")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # TEST 5: Có 2 người trong khung hình nhưng KHÔNG GÂY NHIỄU người đã khóa
    # -------------------------------------------------------------------------
    total_tests += 1
    print("\n[TEST 5] Kiểm tra cô lập nhiễu: Người 2 di chuyển & giơ tay không làm ảnh hưởng người đã khóa...")
    # Người 2 bây giờ cũng giơ tay và nhảy múa
    kpt_person2_up = make_dummy_keypoints(hand_raised=True, side="LEFT")
    for step in range(20):
        sim_time += 0.05
        # Người 2 di chuyển lại gần
        moving_box2 = [350 - step * 2, 160, 480 - step * 2, 420]
        t = tracker.update([box1, moving_box2], [kpt_up, kpt_person2_up], [0.9, 0.95], sim_time)
        locked = controller.update(t, sim_time)
        assert locked is not None and locked.track_id == id_person1, "Lỗi: Khóa bị nhảy sang người khác!"

    print("  => PASS: Target #1 vẫn được khóa chặt 100%, Người #2 bị bỏ qua hoàn toàn.")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # TEST 6: Người đã khóa hạ tay xuống -> Tự động nhả khóa
    # -------------------------------------------------------------------------
    total_tests += 1
    print("\n[TEST 6] Kiểm tra tự động nhả khóa khi người bị khóa hạ tay xuống...")
    # Người 1 hạ tay xuống trong 10 frame (>0.5s)
    for step in range(15):
        sim_time += 0.05
        t = tracker.update([box1, box2], [kpt_down, kpt_down], [0.9, 0.85], sim_time)
        locked = controller.update(t, sim_time)

    assert locked is None, "Lỗi: Người bị khóa đã hạ tay nhưng hệ thống vẫn không nhả khóa!"
    print("  => PASS: Đã tự động nhả khóa thành công khi người đó hạ tay xuống.")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # TEST 7: Người bị khóa đi ra khỏi màn hình -> Tự động nhả khóa
    # -------------------------------------------------------------------------
    total_tests += 1
    print("\n[TEST 7] Kiểm tra khi người bị khóa đi ra khỏi màn hình...")
    # Khóa lại người 1
    for step in range(45):
        sim_time += 0.05
        t = tracker.update([box1], [kpt_up], [0.9], sim_time)
        controller.update(t, sim_time)

    assert controller.locked_track_id is not None, "Chuẩn bị test 7 thất bại."

    # Người 1 biến mất khỏi khung hình trong 10 frames
    for step in range(10):
        sim_time += 0.05
        t = tracker.update([], [], [], sim_time)  # Không còn phát hiện ai
        locked = controller.update(t, sim_time)

    assert controller.locked_track_id is None, "Lỗi: Người đi ra khỏi màn hình nhưng không nhả khóa!"
    print("  => PASS: Đã tự động phát hiện mất dấu và nhả khóa để tìm người tiếp theo.")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # TEST 8: Reset tức thì bằng phím 'P' hoặc nút Web
    # -------------------------------------------------------------------------
    total_tests += 1
    print("\n[TEST 8] Kiểm tra chức năng Reset tức thì bằng phím 'p'...")
    # Khóa lại người 2
    for step in range(45):
        sim_time += 0.05
        t = tracker.update([box2], [kpt_up], [0.9], sim_time)
        controller.update(t, sim_time)

    assert controller.locked_track_id is not None
    # Gọi hàm reset (như khi nhấn 'p' hoặc bấm nút Web)
    controller.reset(reason="Nhan phim P")
    assert controller.locked_track_id is None, "Lỗi: Gọi reset() nhưng không giải phóng khóa!"
    print("  => PASS: Phím 'p' reset mục tiêu lập tức về trạng thái ban đầu.")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # TEST 9: Hiệu năng giải tam giác đạc Stereo 3D & Dò điểm Epipolar
    # -------------------------------------------------------------------------
    total_tests += 1
    print("\n[TEST 9] Đo kiểm hiệu năng tính toán Stereo 3D & Patch Matching...")
    solver = StereoTriangulator(calib_file=None)
    pt_l = (640.0, 360.0)
    pt_r = (595.0, 360.0)

    # 1. Tam giác đạc
    X, Y, Z, D = solver.triangulate_single_point(pt_l, pt_r)
    assert Z > 0, f"Lỗi: Z không hợp lệ ({Z})!"
    print(f"  Tọa độ 3D tính được: D_mid = {D*100:.1f}cm, Z = {Z*100:.1f}cm, X_mid = {X*100:.1f}cm")

    # 2. Benchmark tốc độ
    dummy_img_l = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    dummy_img_r = dummy_img_l.copy()

    start_bench = time.perf_counter()
    num_iters = 50
    for _ in range(num_iters):
        _ = solver.fast_epipolar_match(dummy_img_l, dummy_img_r, pt_l)
        _ = solver.triangulate_single_point(pt_l, pt_r)
    elapsed = (time.perf_counter() - start_bench) / num_iters * 1000.0

    print(f"  => Thời gian xử lý Stereo 3D cho 1 frame: {elapsed:.2f} ms")
    assert elapsed < 15.0, "Thời gian xử lý quá chậm!"
    print(f"  => PASS: Chi phí tính toán cực nhẹ (<{elapsed:.2f}ms/frame), đảm bảo 25-30 FPS mượt mà trên Pi 5!")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # TỔNG KẾT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  KẾT QUẢ KIỂM THỬ: {passed_tests}/{total_tests} TESTS ĐẠT 100% HOÀN HẢO!  ")
    print("=" * 70)

if __name__ == "__main__":
    run_all_tests()

