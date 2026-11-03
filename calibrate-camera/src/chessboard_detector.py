import cv2
import logging

logger = logging.getLogger(__name__)

def find_chessboard(image, pattern_size):
    """
    Finds chessboard corners in an image.
    Args:
        image: BGR or grayscale image
        pattern_size: (columns, rows) of inner corners
    Returns:
        found: bool, True if found
        corners: numpy array of corner coordinates if found, else None
        gray: grayscale version of the image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    
    if found:
        # Refine corner locations to subpixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
    return found, corners, gray

def draw_chessboard(image, pattern_size, corners, found):
    """
    Draws chessboard corners on the image (modifies in-place or returns a copy).
    """
    img_draw = image.copy()
    cv2.drawChessboardCorners(img_draw, pattern_size, corners, found)
    return img_draw
