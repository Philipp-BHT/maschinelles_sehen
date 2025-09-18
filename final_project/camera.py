import cv2 as cv
import numpy as np
import glob, os

class Camera:
    def __init__(self, calib_path="calib_fisheye_charuco.npz",
                 images_glob="calib_imgs/*.jpg",
                 squaresX=7, squaresY=5,
                 squareLength=0.030, markerLength=0.022,
                 dict_name=cv.aruco.DICT_5X5_1000,
                 balance=0.0):
        self.calib_path = calib_path
        self.images_glob = images_glob
        self.squaresX, self.squaresY = squaresX, squaresY
        self.squareLength, self.markerLength = squareLength, markerLength
        self.dict_name = dict_name
        self.balance = balance

        self.K = None
        self.D = None
        self.img_size = None  # (w,h) of calibration images
        self.newK = None
        self.R = np.eye(3)
        self.map1 = None
        self.map2 = None
        self.maps_size = None  # (w,h) target size for current maps

        self.load_parameters()

    def load_parameters(self):
        if os.path.exists(self.calib_path):
            data = np.load(self.calib_path, allow_pickle=False)
            self.K = data["K"]
            self.D = data["D"]
            self.img_size = tuple(data["img_size"])
        else:
            self.calibrate_camera_fisheye()

    def calibrate_camera_fisheye(self):
        aruco = cv.aruco
        dict_id = aruco.getPredefinedDictionary(self.dict_name)
        board = aruco.CharucoBoard(
            (self.squaresX, self.squaresY),
            self.squareLength, self.markerLength, dict_id
        )

        objpoints, imgpoints = [], []
        img_size = None

        for fn in sorted(glob.glob(self.images_glob)):
            img = cv.imread(fn)
            if img is None:
                continue
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            if img_size is None:
                img_size = (gray.shape[1], gray.shape[0])

            corners, ids, _ = aruco.detectMarkers(gray, dict_id)
            if ids is None or len(ids) == 0:
                continue

            aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedCorners=[])

            ok, ch_corners, ch_ids = aruco.interpolateCornersCharuco(
                markerCorners=corners, markerIds=ids, image=gray, board=board
            )
            if not ok or ch_corners is None or len(ch_corners) < 6:
                continue

            # Build per-image 3D/2D lists (fisheye prefers (N,1,3) and (N,1,2))
            obj = board.chessboardCorners[np.int32(ch_ids).ravel()]  # (N,3)
            obj = obj.reshape(-1,1,3).astype(np.float32)
            img_pts = ch_corners.reshape(-1,1,2).astype(np.float32)

            objpoints.append(obj)
            imgpoints.append(img_pts)

        if len(objpoints) < 5:
            raise RuntimeError("Not enough valid ChArUco detections for calibration.")

        K = np.eye(3, dtype=np.float64)
        D = np.zeros((4,1), dtype=np.float64)

        flags = (cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
                 cv.fisheye.CALIB_CHECK_COND |
                 cv.fisheye.CALIB_FIX_SKEW)

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

        rms, K, D, rvecs, tvecs = cv.fisheye.calibrate(
            objectPoints=objpoints,
            imagePoints=imgpoints,
            image_size=img_size,
            K=K, D=D, rvecs=None, tvecs=None,
            flags=flags, criteria=criteria
        )

        print(f"Fisheye RMS reprojection error: {rms:.4f}")
        print("K:\n", K)
        print("D (k1..k4):\n", D.ravel())

        np.savez(self.calib_path, K=K, D=D, img_size=img_size, rms=rms)
        print(f"Saved to {self.calib_path}")

        self.K, self.D, self.img_size = K, D, img_size
        self._build_rectification_maps(self.img_size)

    def _build_rectification_maps(self, frame_size):
        """Create/refresh rectification maps for the given frame size (w,h)."""
        self.newK = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.D, frame_size, self.R, balance=self.balance, fov_scale=1.0
        )
        self.map1, self.map2 = cv.fisheye.initUndistortRectifyMap(
            self.K, self.D, self.R, self.newK, frame_size, m1type=cv.CV_16SC2
        )
        self.maps_size = frame_size

    def undistort_image(self, image):
        h, w = image.shape[:2]
        size = (w, h)
        if self.map1 is None or self.maps_size != size:
            self._build_rectification_maps(size)
        # Remap
        return cv.remap(image, self.map1, self.map2, interpolation=cv.INTER_LINEAR)

    # optional: undistort 2D points only (avoid remapping the whole image)
    def undistort_points(self, pts):
        """
        pts: (N,2) pixel coordinates from distorted frame of size 'maps_size'
        returns (N,2) rectified pixel coords in same rectified space (cameraMatrix=self.newK).
        """
        if self.map1 is None:
            raise RuntimeError("Call undistort_image at least once to build maps/newK, or call _build_rectification_maps().")
        pts = np.asarray(pts, dtype=np.float32).reshape(-1,1,2)
        # Convert to normalized with fisheye model, then back to pixels with newK
        norm = cv.fisheye.undistortPoints(pts, self.K, self.D, R=self.R, P=self.newK)
        return norm.reshape(-1,2)
