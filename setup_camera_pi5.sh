#!/bin/bash
set -e

echo "=========================================================="
echo "  TỰ ĐỘNG CÀI ĐẶT DRIVER PI CAMERA V2 CHO RASPBERRY PI 5  "
echo "=========================================================="

# 1. Gỡ bỏ các gói libcamera cũ của Ubuntu để tránh xung đột
echo "[1/6] Đang dọn dẹp các gói libcamera cũ của Ubuntu..."
sudo apt remove --purge -y libcamera-dev libcamera0.2 libcamera-tools python3-libcamera gstreamer1.0-libcamera || true
sudo apt autoremove -y || true

# 2. Cài đặt các công cụ biên dịch và thư viện phụ thuộc
echo "[2/6] Đang cài đặt các công cụ biên dịch (meson, ninja, cmake, pybind11)..."
sudo apt update
sudo apt install -y build-essential meson cmake ninja-build git pkg-config \
    libgnutls28-dev openssl libtiff-dev pybind11-dev python3-yaml python3-ply \
    libboost-dev libglib2.0-dev libdrm-dev libexif-dev libjpeg-dev libpng-dev \
    libevent-dev libcap-dev python3-jinja2 python3-pip python3-venv

# 3. Tải mã nguồn chính thức của Raspberry Pi
echo "[3/6] Đang tải mã nguồn libcamera chính thức của hãng Raspberry Pi..."
cd ~
rm -rf libcamera
git clone --depth 1 https://github.com/raspberrypi/libcamera.git libcamera
cd libcamera

# 4. Cấu hình biên dịch với PiSP cho Pi 5 và Python bindings
echo "[4/6] Đang cấu hình driver (hỗ trợ phần cứng PiSP của Pi 5 & Python bindings)..."
meson setup build --buildtype=release \
    -Dpipelines=rpi/vc4,rpi/pisp \
    -Dipas=rpi/vc4,rpi/pisp \
    -Dv4l2=true \
    -Dcam=enabled \
    -Dpycamera=enabled \
    -Dtest=false \
    -Dlc-compliance=disabled

# 5. Tiến hành biên dịch (tận dụng toàn bộ 4 nhân CPU của Pi 5)
echo "[5/6] Đang biên dịch driver... (Quá trình này mất khoảng 5-10 phút)"
ninja -C build -j$(nproc)
sudo ninja -C build install

# Cập nhật đường dẫn thư viện hệ thống
echo "/usr/local/lib/aarch64-linux-gnu" | sudo tee /etc/ld.so.conf.d/libcamera.conf
echo "/usr/local/lib" | sudo tee -a /etc/ld.so.conf.d/libcamera.conf
sudo ldconfig

# 6. Liên kết mô-đun Python vào môi trường yolo-env
echo "[6/6] Đang liên kết mô-đun Python libcamera vào môi trường yolo-env..."
SITE_PACKAGES="$HOME/hand_detection/yolo-env/lib/python3.12/site-packages"

# Sao chép module python libcamera vào yolo-env
if [ -d "/usr/local/lib/python3.12/site-packages/libcamera" ]; then
    cp -r /usr/local/lib/python3.12/site-packages/libcamera "$SITE_PACKAGES/"
elif [ -d "$HOME/libcamera/build/src/py/libcamera" ]; then
    cp -r "$HOME/libcamera/build/src/py/libcamera" "$SITE_PACKAGES/"
fi

# Cài đặt / cập nhật lại picamera2 trong yolo-env
source "$HOME/hand_detection/yolo-env/bin/activate"
pip install --no-cache-dir picamera2

# Tự động vá pykms để bỏ qua DRM preview không cần thiết
python3 -c "
import os
file = '$SITE_PACKAGES/picamera2/previews/drm_preview.py'
if os.path.exists(file):
    with open(file, 'r') as f:
        code = f.read()
    old = '    except ImportError:\n        import pykms\n        from pykms import PixelFormat'
    new = '    except ImportError:\n        from unittest.mock import MagicMock\n        pykms = MagicMock()\n        PixelFormat = MagicMock()'
    if old in code:
        code = code.replace(old, new)
        with open(file, 'w') as f:
            f.write(code)
"

echo "=========================================================="
echo "  KIỂM TRA HOÀN TẤT: ĐANG TEST THỬ THƯ VIỆN CAMERA..."
python3 -c "import libcamera; print('>>> [THÀNH CÔNG] Đã nạp được libcamera!')" || echo ">>> [CHÚ Ý] Chưa nạp được trực tiếp libcamera."
python3 -c "from picamera2 import Picamera2; print('>>> [THÀNH CÔNG] Picamera2 đã sẵn sàng hoạt động!')" || true
echo "=========================================================="
echo "CÀI ĐẶT HOÀN TẤT! BÂY GIỜ BẠN CÓ THỂ CHẠY: python3 pi_camera_pose.py"
