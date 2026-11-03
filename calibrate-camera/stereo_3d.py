import cv2
import numpy as np
import os
import config

class Stereo3D:
    def __init__(self):
        calib_file = os.path.join(config.DIR_CALIB_RESULT, "stereo_calibration.npz")
        if not os.path.exists(calib_file):
            raise FileNotFoundError(f"Calibration file not found at {calib_file}")
            
        data = np.load(calib_file)
        self.Q = data['Q']
        self.P1 = data['P1']
        self.P2 = data['P2']
        self.R1 = data['R1']
        self.R2 = data['R2']
        self.K1 = data['K1']
        self.D1 = data['D1']
        self.K2 = data['K2']
        self.D2 = data['D2']
        self.T = data['T']
        
        # Calculate baseline from translation vector
        self.baseline = np.linalg.norm(self.T)
        
    def triangulate_point(self, pt_left, pt_right):
        """
        Triangulates a single point given its coordinates in the left and right ORIGINAL images.
        pt_left: (x, y) in left image
        pt_right: (x, y) in right image
        Returns: (X, Y, Z) in the left camera's coordinate frame (meters)
        """
        # Convert to numpy arrays of shape (1, 1, 2)
        pl = np.array([[[pt_left[0], pt_left[1]]]], dtype=np.float32)
        pr = np.array([[[pt_right[0], pt_right[1]]]], dtype=np.float32)
        
        # Undistort points (since inputs are from original distorted images)
        pl_undistorted = cv2.undistortPoints(pl, self.K1, self.D1, R=self.R1, P=self.P1)
        pr_undistorted = cv2.undistortPoints(pr, self.K2, self.D2, R=self.R2, P=self.P2)
        
        # Reshape for triangulation
        pl_u = pl_undistorted.reshape(2, 1)
        pr_u = pr_undistorted.reshape(2, 1)
        
        # Triangulate
        point_4d = cv2.triangulatePoints(self.P1, self.P2, pl_u, pr_u)
        
        # Normalize by w
        point_3d = point_4d[:3, :] / point_4d[3, :]
        return point_3d.flatten()

def main():
    print("===========================================")
    print("         STEREO 3D RECONSTRUCTION          ")
    print("===========================================")
    
    try:
        s3d = Stereo3D()
        print("Calibration loaded successfully.")
        print(f"Measured Baseline: {config.EXPECTED_BASELINE} m")
        print(f"Estimated Baseline: {s3d.baseline:.4f} m")
        print(f"Coordinate Unit: meters")
        
        print("\nTesting triangulation with dummy points (center of image)...")
        # Example points
        cx = config.IMAGE_WIDTH / 2
        cy = config.IMAGE_HEIGHT / 2
        
        # Assume an object is at center of left camera, and slightly shifted left in right camera
        pt_left = (cx, cy)
        pt_right = (cx - 50, cy) # 50 pixels disparity
        
        point_3d = s3d.triangulate_point(pt_left, pt_right)
        
        print(f"Left Pixel:  {pt_left}")
        print(f"Right Pixel: {pt_right}")
        print(f"3D Point (X, Y, Z): ({point_3d[0]:.4f}, {point_3d[1]:.4f}, {point_3d[2]:.4f}) m")
        print(f"Distance (Depth Z): {point_3d[2]:.4f} m")
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("You must run calibration before using Stereo3D.")

if __name__ == "__main__":
    main()
