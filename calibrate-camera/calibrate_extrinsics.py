#!/usr/bin/env python3
"""
HIỆU CHUẨN THÔNG SỐ NGOẠI TẠI STEREO CAMERA (EXTRINSIC CALIBRATION)
(calibrate_extrinsics.py)
---------------------------------------------------------------------------------
Mục đích:
- Dành cho trường hợp di chuyển cụm camera sang vị trí mới, điều chỉnh chân đế,
  hoặc thay đổi khoảng cách baseline giữa 2 camera Pi 5.
- KHÔNG CẦN chụp lại 30-40 tấm ảnh bàn cờ ở mọi góc độ để tính lại thông số nội tại.
- TỰ ĐỘNG NẠP ĐỘNG thông số nội tại K1, D1, K2, D2 từ file có sẵn (.npz) mà không
  hardcode cố định bất kỳ ma trận nào trong code.
- Khóa cố định thông số nội tại (CALIB_FIX_INTRINSIC), chỉ giải tìm ma trận xoay R,
  vector tịnh tiến T và tính toán ma trận nắn phẳng R1, R2, P1, P2, Q.
- Tự động sinh file kiểm định rectification_test.png với đường dóng ngang epipolar.
"""

import os
import sys
import glob
import csv
import argparse
import logging
from datetime import datetime

import cv2
import numpy as np

import config
from src.chessboard_detector import find_chessboard, draw_chessboard

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("ExtrinsicCalib")


def resolve_intrinsics_file(custom_path=None):
    """
    Tự động tìm kiếm file chứa thông số nội tại K1, D1, K2, D2.
    Thứ tự ưu tiên:
      1. Đường dẫn do người dùng truyền vào qua --intrinsics
      2. Biến INTRINSICS_FILE trong config.py (nếu có)
      3. File camera_intrinsics.npz trong thư mục hiện tại hoặc calibrate-camera/
      4. File stereo_calibration.npz trong thư mục kết quả (DIR_CALIB_RESULT)
      5. Thư mục bản chuẩn prototype_6
    """
    candidates = [
        custom_path,
        getattr(config, 'INTRINSICS_FILE', None),
        "camera_intrinsics.npz",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_intrinsics.npz"),
        os.path.join(config.DIR_CALIB_RESULT, "stereo_calibration.npz"),
        os.path.join(config.OUTPUT_DIRECTORY, "calibration_result", "stereo_calibration.npz"),
        "stereo_vision/calibration_result/stereo_calibration.npz",
        os.path.join(os.path.expanduser("~"), "hand_detection/calibrate-camera/stereo_vision/calibration_result/stereo_calibration.npz"),
        os.path.join(os.path.expanduser("~"), "hand_detection/calibrate-camera/camera_intrinsics.npz"),
        # Thư mục nguyên mẫu chuẩn trên hệ thống
        "prototype_6_8x5_0,042_0,5045*/calibration_result/stereo_calibration.npz",
        "../prototype_6_8x5_0,042_0,5045*/calibration_result/stereo_calibration.npz",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                     "prototype_6_8x5_0,042_0,5045*", "calibration_result", "stereo_calibration.npz"),
    ]

    for p in candidates:
        if not p:
            continue
        # Xử lý glob pattern nếu có ký tự đại diện *
        matches = glob.glob(p)
        if matches and os.path.exists(matches[0]):
            return os.path.abspath(matches[0])

    return None


def load_intrinsics(calib_path):
    """
    Đọc động ma trận K1, D1, K2, D2 và image_size từ file .npz bất kỳ.
    Không cố định mã cứng giá trị số.
    """
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"Không tìm thấy file thông số nội tại tại: {calib_path}")

    data = np.load(calib_path)
    required = ['K1', 'D1', 'K2', 'D2']
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"File {calib_path} không hợp lệ! Thiếu các ma trận bắt buộc: {missing}")

    K1 = data['K1'].astype(np.float64)
    D1 = data['D1'].astype(np.float64)
    K2 = data['K2'].astype(np.float64)
    D2 = data['D2'].astype(np.float64)

    image_size = None
    if 'image_size' in data:
        raw_size = data['image_size']
        image_size = (int(raw_size[0]), int(raw_size[1]))

    return K1, D1, K2, D2, image_size


def export_standalone_intrinsics(src_path, dst_path):
    """
    Trích xuất riêng bộ thông số nội tại (K1, D1, K2, D2, image_size)
    thành 1 file .npz độc lập để lưu trữ làm bản gốc vàng.
    """
    K1, D1, K2, D2, img_size = load_intrinsics(src_path)
    save_dict = {'K1': K1, 'D1': D1, 'K2': K2, 'D2': D2}
    if img_size is not None:
        save_dict['image_size'] = np.array(img_size)

    out_dir = os.path.dirname(os.path.abspath(dst_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez(dst_path, **save_dict)
    logger.info(f"Đã trích xuất thông số nội tại thành công sang: {dst_path}")


def draw_horizontal_lines(img, line_gap=40, color=(0, 255, 0)):
    """Vẽ các đường dóng ngang để kiểm tra độ thẳng hàng epipolar."""
    out = img.copy()
    h, w = out.shape[:2]
    for y in range(0, h, line_gap):
        cv2.line(out, (0, y), (w, y), color, 1, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Hiệu Chuẩn Thông Số Ngoại Tại Stereo Camera (CALIB_FIX_INTRINSIC)"
    )
    parser.add_argument('--intrinsics', type=str, default=None,
                        help="Đường dẫn file .npz chứa thông số nội tại K1, D1, K2, D2 (mặc định tự tìm)")
    parser.add_argument('--export_intrinsics', type=str, default=None,
                        help="Trích xuất thông số nội tại từ file nguồn sang file đích chỉ định rồi thoát")
    parser.add_argument('--show_intrinsics', action='store_true',
                        help="In thông số nội tại đang nạp ra màn hình rồi thoát")
    parser.add_argument('--left_dir', type=str, default=config.DIR_LEFT_IMAGES,
                        help="Thư mục ảnh bàn cờ camera trái")
    parser.add_argument('--right_dir', type=str, default=config.DIR_RIGHT_IMAGES,
                        help="Thư mục ảnh bàn cờ camera phải")
    parser.add_argument('--min_pairs', type=int, default=5,
                        help="Số lượng cặp ảnh bàn cờ tối thiểu cần đạt (mặc định: 5)")
    parser.add_argument('--alpha', type=float, default=0.0,
                        help="Tham số crop mép stereoRectify: 0 (cắt bỏ phần đen) đến 1 (giữ trọn hình)")
    args = parser.parse_args()

    print("==================================================================")
    print("      STEREO EXTRINSIC CALIBRATION (QUICK RE-CALIBRATION)         ")
    print("==================================================================")

    # 1. TÌM KIẾM VÀ NẠP FILE THÔNG SỐ NỘI TẠI
    calib_file = resolve_intrinsics_file(args.intrinsics)
    if not calib_file:
        logger.error("KHÔNG TÌM THẤY file thông số nội tại (.npz) nào trên hệ thống!")
        logger.error("Vui lòng truyền đường dẫn cụ thể qua tham số: --intrinsics <đường_dẫn_file.npz>")
        sys.exit(1)

    logger.info(f"Nguồn thông số nội tại: {calib_file}")
    try:
        K1, D1, K2, D2, file_img_size = load_intrinsics(calib_file)
    except Exception as e:
        logger.error(f"Lỗi khi đọc file thông số nội tại: {e}")
        sys.exit(1)

    # Xử lý tính năng Export Intrinsics độc lập nếu được yêu cầu
    if args.export_intrinsics:
        export_standalone_intrinsics(calib_file, args.export_intrinsics)
        sys.exit(0)

    # In kiểm tra thông số nội tại
    if args.show_intrinsics:
        print("\n--- THÔNG SỐ NỘI TẠI ĐƯỢC NẠP ---")
        print("Camera Trái K1:\n", K1)
        print("Độ méo Trái D1:\n", D1.ravel())
        print("Camera Phải K2:\n", K2)
        print("Độ méo Phải D2:\n", D2.ravel())
        if file_img_size:
            print("Độ phân giải chuẩn:", file_img_size)
        sys.exit(0)

    os.makedirs(config.DIR_CALIB_RESULT, exist_ok=True)
    os.makedirs(config.DIR_DEBUG, exist_ok=True)

    # 2. ĐỌC TẬP ẢNH BÀN CỜ ĐÃ CHỤP TẠI ĐỊA ĐIỂM MỚI
    left_images = sorted(glob.glob(os.path.join(args.left_dir, '*.png')))
    right_images = sorted(glob.glob(os.path.join(args.right_dir, '*.png')))

    if len(left_images) == 0 or len(right_images) == 0:
        logger.error(f"Không tìm thấy ảnh bàn cờ! Left: {len(left_images)}, Right: {len(right_images)}")
        logger.error(f"Hãy chạy capture_stereo.py để chụp 8-12 cặp ảnh bàn cờ ở vị trí mới.")
        sys.exit(1)

    if len(left_images) != len(right_images):
        logger.error(f"Số lượng ảnh 2 bên không khớp! Left: {len(left_images)}, Right: {len(right_images)}")
        sys.exit(1)

    pattern_size = config.CHESSBOARD_PATTERN_SIZE
    square_size = config.SQUARE_SIZE

    # Khởi tạo tọa độ 3D của các góc bàn cờ (Z = 0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp = objp * square_size

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []
    valid_pairs = []
    report_data = []
    current_image_size = None

    logger.info(f"Đang dò tìm góc bàn cờ trên {len(left_images)} cặp ảnh mới...")
    for l_path, r_path in zip(left_images, right_images):
        pair_id = os.path.basename(l_path).split('_')[1].split('.')[0]
        img_l = cv2.imread(l_path)
        img_r = cv2.imread(r_path)

        if img_l is None or img_r is None:
            continue

        if current_image_size is None:
            current_image_size = (img_l.shape[1], img_l.shape[0])
            if file_img_size and current_image_size != file_img_size:
                logger.warning(f"CẢNH BÁO: Độ phân giải ảnh mới {current_image_size} khác với độ phân giải của file nội tại {file_img_size}!")

        found_l, corners_l, _ = find_chessboard(img_l, pattern_size)
        found_r, corners_r, _ = find_chessboard(img_r, pattern_size)

        if found_l and found_r:
            objpoints.append(objp)
            imgpoints_left.append(corners_l)
            imgpoints_right.append(corners_r)
            valid_pairs.append(pair_id)

            # Lưu ảnh debug để xem lại khi cần
            dbg_l = draw_chessboard(img_l, pattern_size, corners_l, found_l)
            dbg_r = draw_chessboard(img_r, pattern_size, corners_r, found_r)
            cv2.imwrite(os.path.join(config.DIR_DEBUG, f"extrinsic_left_{pair_id}.png"), dbg_l)
            cv2.imwrite(os.path.join(config.DIR_DEBUG, f"extrinsic_right_{pair_id}.png"), dbg_r)

            report_data.append({
                'pair_id': pair_id, 'left_image': os.path.basename(l_path), 'right_image': os.path.basename(r_path),
                'left_corner_found': True, 'right_corner_found': True, 'valid': True, 'reason': ''
            })
        else:
            reasons = []
            if not found_l: reasons.append("Left missing")
            if not found_r: reasons.append("Right missing")
            report_data.append({
                'pair_id': pair_id, 'left_image': os.path.basename(l_path), 'right_image': os.path.basename(r_path),
                'left_corner_found': found_l, 'right_corner_found': found_r, 'valid': False, 'reason': " & ".join(reasons)
            })

    logger.info(f"Dò tìm thành công {len(valid_pairs)} / {len(left_images)} cặp ảnh hợp lệ.")

    if len(valid_pairs) < args.min_pairs:
        logger.error(f"Số lượng cặp ảnh hợp lệ ({len(valid_pairs)}) ít hơn mức tối thiểu yêu cầu ({args.min_pairs})!")
        logger.error("Vui lòng chụp thêm một vài cặp ảnh rõ nét hơn.")
        sys.exit(1)

    # 3. CHẠY STEREO CALIBRATION VỚI CỜ CALIB_FIX_INTRINSIC
    logger.info("==================================================================")
    logger.info("  BẮT ĐẦU TỐI ƯU HÓA NGOẠI TẠI (ĐÓNG BĂNG K1, D1, K2, D2)         ")
    logger.info("==================================================================")

    flags_stereo = cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    ret_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        K1, D1, K2, D2, current_image_size,
        criteria=criteria_stereo, flags=flags_stereo
    )

    logger.info(f"Stereo RMS Error: {ret_stereo:.4f} pixels")
    logger.info(f"Ma trận xoay R (Camera Phải so với Camera Trái):\n{R}")
    logger.info(f"Vector tịnh tiến T:\n{T.ravel()}")

    # Đánh giá Baseline đo đạc và thực tế
    baseline = float(np.linalg.norm(T))
    logger.info(f"Baseline tính toán từ ảnh (||T||): {baseline:.4f} m ({baseline*100:.2f} cm)")
    logger.info(f"Baseline kỳ vọng (EXPECTED_BASELINE): {config.EXPECTED_BASELINE:.4f} m ({config.EXPECTED_BASELINE*100:.2f} cm)")

    diff = abs(baseline - config.EXPECTED_BASELINE)
    percent_error = (diff / config.EXPECTED_BASELINE) * 100.0 if config.EXPECTED_BASELINE > 0 else 0.0
    logger.info(f"Độ lệch tuyệt đối: {diff:.4f} m ({diff*100:.2f} cm) | Sai số: {percent_error:.2f}%")

    if percent_error > config.BASELINE_WARNING_THRESHOLD_PERCENT:
        logger.warning("****************************************************************")
        logger.warning(f"CẢNH BÁO: Baseline ước lượng lệch {percent_error:.2f}% so với cấu hình EXPECTED_BASELINE!")
        logger.warning("Vui lòng kiểm tra lại SQUARE_SIZE hoặc đo lại khoảng cách camera tại nơi mới.")
        logger.warning("****************************************************************")

    # 4. TÍNH TOÁN STEREO RECTIFICATION (NẮN PHẲNG)
    logger.info("Đang tính toán các ma trận nắn phẳng Stereo Rectification...")
    R1, R2, P1, P2, Q, ROI1, ROI2 = cv2.stereoRectify(
        K1, D1, K2, D2, current_image_size, R, T, alpha=args.alpha
    )

    logger.info("Đang tính toán bản đồ nắn phẳng (Rectification Remap)...")
    map1_left, map2_left = cv2.initUndistortRectifyMap(K1, D1, R1, P1, current_image_size, cv2.CV_16SC2)
    map1_right, map2_right = cv2.initUndistortRectifyMap(K2, D2, R2, P2, current_image_size, cv2.CV_16SC2)

    # 5. LƯU TOÀN BỘ KẾT QUẢ VÀO ĐÚNG ĐỊNH DẠNG DEMO_RAISE_HAND_STEREO.PY YÊU CẦU
    logger.info("Đang lưu các file ma trận hiệu chuẩn...")

    # Lưu stereo_calibration.npz
    npz_path = os.path.join(config.DIR_CALIB_RESULT, "stereo_calibration.npz")
    np.savez(npz_path,
             K1=K1, D1=D1, K2=K2, D2=D2,
             R=R, T=T, E=E, F=F,
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             ROI1=ROI1, ROI2=ROI2, image_size=current_image_size)
    logger.info(f"-> Đã lưu: {npz_path}")

    # Lưu rectification_maps.npz
    maps_path = os.path.join(config.DIR_CALIB_RESULT, "rectification_maps.npz")
    np.savez(maps_path,
             map1_left=map1_left, map2_left=map2_left,
             map1_right=map1_right, map2_right=map2_right)
    logger.info(f"-> Đã lưu: {maps_path}")

    # Lưu stereo_calibration.yml
    yml_path = os.path.join(config.DIR_CALIB_RESULT, "stereo_calibration.yml")
    fs = cv2.FileStorage(yml_path, cv2.FILE_STORAGE_WRITE)
    fs.write("K1", K1)
    fs.write("D1", D1)
    fs.write("K2", K2)
    fs.write("D2", D2)
    fs.write("R", R)
    fs.write("T", T)
    fs.write("E", E)
    fs.write("F", F)
    fs.write("R1", R1)
    fs.write("R2", R2)
    fs.write("P1", P1)
    fs.write("P2", P2)
    fs.write("Q", Q)
    fs.release()
    logger.info(f"-> Đã lưu: {yml_path}")

    # Lưu báo cáo chi tiết TXT
    report_txt_path = os.path.join(config.DIR_CALIB_RESULT, "calibration_report.txt")
    with open(report_txt_path, "w") as f:
        f.write("Stereo Extrinsic Re-Calibration Report\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Intrinsics Source File: {calib_file}\n")
        f.write(f"OpenCV Version: {cv2.__version__}\n")
        f.write(f"Image Resolution: {current_image_size}\n")
        f.write(f"Pattern Size: {pattern_size} | Square Size: {square_size} m\n")
        f.write(f"Stereo RMS: {ret_stereo:.4f}\n")
        f.write(f"Baseline (Estimated): {baseline:.4f} m\n")
        f.write(f"Measured (EXPECTED): {config.EXPECTED_BASELINE:.4f} m\n")
        f.write(f"Difference: {diff:.4f} m ({percent_error:.2f}%)\n")
        f.write("\n--- EXTRINSIC MATRICES ---\n")
        f.write(f"R:\n{R}\n")
        f.write(f"T:\n{T}\n")
        f.write(f"R1:\n{R1}\n")
        f.write(f"R2:\n{R2}\n")
        f.write(f"P1:\n{P1}\n")
        f.write(f"P2:\n{P2}\n")
        f.write(f"Q:\n{Q}\n")
    logger.info(f"-> Đã lưu: {report_txt_path}")

    # Lưu file CSV báo cáo ảnh
    report_csv_path = os.path.join(config.DIR_CALIB_RESULT, "calibration_report.csv")
    with open(report_csv_path, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'pair_id', 'left_image', 'right_image', 'left_corner_found', 'right_corner_found', 'valid', 'reason'
        ])
        writer.writeheader()
        writer.writerows(report_data)

    # 6. SINH ẢNH TRỰC QUAN KIỂM ĐỊNH RECTIFICATION (ĐƯỜNG DÓNG NGANG)
    if valid_pairs:
        test_id = valid_pairs[0]
        test_l_img = cv2.imread(os.path.join(args.left_dir, f"left_{test_id}.png"))
        test_r_img = cv2.imread(os.path.join(args.right_dir, f"right_{test_id}.png"))

        if test_l_img is not None and test_r_img is not None:
            rect_l = cv2.remap(test_l_img, map1_left, map2_left, cv2.INTER_LINEAR)
            rect_r = cv2.remap(test_r_img, map1_right, map2_right, cv2.INTER_LINEAR)
            combined = cv2.hconcat([rect_l, rect_r])
            test_preview = draw_horizontal_lines(combined, line_gap=40, color=(0, 255, 0))

            preview_path = os.path.join(config.DIR_CALIB_RESULT, "rectification_test.png")
            cv2.imwrite(preview_path, test_preview)
            logger.info(f"-> Đã tạo ảnh kiểm định nắn phẳng: {preview_path}")

    print("\n==================================================================")
    print("           HIỆU CHUẨN NGOẠI TẠI HOÀN TẤT THÀNH CÔNG!              ")
    print(f" - Stereo RMS:          {ret_stereo:.4f} pixels")
    print(f" - Baseline ước lượng:  {baseline*100:.2f} cm")
    print(f" - File lưu kết quả:    {npz_path}")
    print(" - demo_raise_hand_stereo.py đã sẵn sàng chạy với thông số mới!")
    print("==================================================================")


if __name__ == "__main__":
    main()

