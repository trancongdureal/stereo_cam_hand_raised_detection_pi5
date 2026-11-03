import cv2
import logging
import sys
import socket
import struct
import numpy as np

logger = logging.getLogger(__name__)

class CameraManager:
    """
    Manages camera abstraction, supporting Picamera2, OpenCV, and Network Socket (for dual-Pi setups).
    """
    def __init__(self, backend='auto', camera_id=0, width=1280, height=720, fps=30, host=None, port=None):
        self.backend = backend
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.host = host
        self.port = port
        self.picam2 = None
        self.cap = None
        self.sock = None
        
        self.use_picam2 = False
        self.use_network = False

        self._initialize_camera()

    def _initialize_camera(self):
        if self.backend == 'network':
            self.use_network = True
            self._init_network()
            return

        if self.backend == 'auto':
            try:
                import libcamera
                from picamera2 import Picamera2
                self.use_picam2 = True
                logger.info("Picamera2 detected, using picamera2 backend.")
            except ImportError:
                self.use_picam2 = False
                logger.info("Picamera2 not found, falling back to OpenCV backend.")
        elif self.backend == 'picamera2':
            try:
                from picamera2 import Picamera2
                self.use_picam2 = True
            except ImportError:
                logger.error("Picamera2 backend requested but library is not available.")
                sys.exit(1)
        elif self.backend == 'opencv':
            self.use_picam2 = False
        else:
            logger.error(f"Unknown camera backend: {self.backend}")
            sys.exit(1)

        if self.use_picam2:
            self._init_picamera2()
        else:
            self._init_opencv()

    def _init_picamera2(self):
        from picamera2 import Picamera2
        try:
            try:
                self.picam2 = Picamera2(int(self.camera_id))
            except TypeError:
                self.picam2 = Picamera2()
            # Create a configuration
            config = self.picam2.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            # You might need to set fps in the config depending on picamera2 version
            self.picam2.configure(config)
            self.picam2.start()
            logger.info(f"Initialized Picamera2 (Camera {self.camera_id}) at {self.width}x{self.height}")
        except Exception as e:
            logger.error(f"Failed to initialize Picamera2 (Camera {self.camera_id}): {e}")
            sys.exit(1)

    def _init_opencv(self):
        # On Windows, DirectShow is sometimes needed for multiple webcams (cv2.CAP_DSHOW).
        # We will try default backend first.
        try:
            if sys.platform.startswith('win'):
                self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(self.camera_id)

            if not self.cap.isOpened():
                logger.error(f"Failed to open OpenCV camera {self.camera_id}.")
                sys.exit(1)

            # Request specific resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Check actual resolution obtained
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_w != self.width or actual_h != self.height:
                logger.warning(f"Camera {self.camera_id}: Requested {self.width}x{self.height}, but got {actual_w}x{actual_h}.")
                # We update the properties to reflect reality, though strict checking in capture script will catch this
                self.width = actual_w
                self.height = actual_h

            logger.info(f"Initialized OpenCV Camera {self.camera_id} at {self.width}x{self.height}")
        except Exception as e:
            logger.error(f"Exception initializing OpenCV Camera {self.camera_id}: {e}")
            sys.exit(1)

    def _init_network(self):
        try:
            logger.info(f"Connecting to Network Camera at {self.host}:{self.port}...")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5.0)
            self.sock.connect((self.host, self.port))
            logger.info(f"Connected successfully to Network Camera at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Network Camera at {self.host}:{self.port}: {e}")
            logger.error("Please ensure the camera worker server is running on the second Pi.")
            sys.exit(1)

    def _recv_all(self, num_bytes):
        data = bytearray()
        while len(data) < num_bytes:
            packet = self.sock.recv(num_bytes - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def read(self):
        """
        Reads a frame.
        Returns:
            ret (bool): True if successful
            frame (numpy.ndarray): The BGR frame (OpenCV format)
        """
        if self.use_network:
            try:
                header = self._recv_all(4)
                if not header:
                    return False, None
                msg_len = struct.unpack(">I", header)[0]
                frame_data = self._recv_all(msg_len)
                if not frame_data:
                    return False, None
                
                np_arr = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None:
                    return False, None
                return True, frame
            except Exception as e:
                logger.error(f"Network Camera read error: {e}")
                return False, None
        elif self.use_picam2:
            try:
                # Trong Picamera2/libcamera với format 'RGB888', thứ tự byte thực tế trong bộ nhớ
                # đã là [B, G, R] (chuẩn BGR của OpenCV).
                # Không chuyển cv2.cvtColor(..., cv2.COLOR_RGB2BGR) để tránh đảo ngược Red và Blue (gây ám xanh).
                frame_bgr = self.picam2.capture_array("main")
                return True, frame_bgr
            except Exception as e:
                logger.error(f"Picamera2 Camera {self.camera_id} read error: {e}")
                return False, None
        else:
            if self.cap is None:
                return False, None
            ret, frame = self.cap.read()
            if not ret:
                logger.error(f"OpenCV Camera {self.camera_id} read returned False.")
            return ret, frame

    def close(self):
        if self.use_network and self.sock:
            try:
                self.sock.close()
            except:
                pass
        elif self.use_picam2 and self.picam2:
            self.picam2.stop()
            self.picam2.close()
        elif self.cap:
            self.cap.release()
