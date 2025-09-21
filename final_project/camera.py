import cv2 as cv
import numpy as np
import glob, os, time

class Camera:
    def __init__(self, calib_path="calib_fisheye_charuco.npz",
                 squaresX=11, squaresY=8,
                 squareLength=0.020, markerLength=0.0145,
                 dict_name=cv.aruco.DICT_4X4_1000,
                 balance=0.8, init=False):
        self.calib_path = calib_path
        self.image_path = "calib_imgs"
        self.images_glob = os.path.join(self.image_path, "*.jpg")
        self.squaresX, self.squaresY = squaresX, squaresY
        self.squareLength, self.markerLength = squareLength, markerLength
        self.dict_name = dict_name
        self.balance = balance

        self.K = None
        self.D = None
        self.img_size = None
        self.newK = None
        self.R = np.eye(3)
        self.map1 = None
        self.map2 = None
        self.maps_size = None

        if not init:
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
        start_id = 2

        # Create a temp board
        tmp_board = aruco.CharucoBoard((self.squaresX, self.squaresY), self.squareLength, self.markerLength, dict_id)
        num_markers = len(tmp_board.getIds()) if hasattr(tmp_board, "getIds") else len(tmp_board.ids)

        new_ids = (start_id + np.arange(num_markers)).astype(np.int32).reshape(-1, 1)
        board = aruco.CharucoBoard(
            (self.squaresX, self.squaresY),
            self.squareLength, self.markerLength,
            dict_id,
            ids=new_ids  # <-- key bit: pass ids at construction
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

            corners3d = board.getChessboardCorners()
            obj = corners3d[np.int32(ch_ids).ravel()].astype(np.float32).reshape(-1, 1, 3)
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
        assert self.K is not None and self.D is not None, "Intrinsics not loaded"
        w, h = frame_size  # (w,h)

        # Compute newK with a reasonable balance
        bal = float(getattr(self, "balance", 0.5))
        bal = max(0.0, min(1.0, bal))
        R = np.eye(3, dtype=np.float64)
        newK = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.D, (w, h), R, balance=bal, fov_scale=1.0
        )

        def _looks_bad(Km):
            ok = np.isfinite(Km).all()
            fx, fy, cx, cy = Km[0, 0], Km[1, 1], Km[0, 2], Km[1, 2]
            return (not ok) or fx < 10 or fy < 10 or cx < 0 or cy < 0 or cx > 2 * w or cy > 2 * h

        if _looks_bad(newK):
            newK = self.K.copy()
            R = np.eye(3, dtype=np.float64)

        # Build float maps (two maps: x and y)
        map1, map2 = cv.fisheye.initUndistortRectifyMap(
            self.K, self.D, R, newK, (w, h), m1type=cv.CV_32F
        )

        mx, my = map1, map2
        valid = np.mean((mx >= 0) & (mx < w) & (my >= 0) & (my < h))

        if valid < 0.05:
            newK_try = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(
                self.K, self.D, (w, h), np.eye(3), balance=0.0, fov_scale=1.0
            )
            if _looks_bad(newK_try):
                newK_try = self.K.copy()
            map1, map2 = cv.fisheye.initUndistortRectifyMap(
                self.K, self.D, np.eye(3), newK_try, (w, h), m1type=cv.CV_32F
            )
            mx, my = map1, map2
            valid = np.mean((mx >= 0) & (mx < w) & (my >= 0) & (my < h))

            if valid < 0.05:
                newK_try = self.K.copy()
                map1, map2 = cv.fisheye.initUndistortRectifyMap(
                    self.K, self.D, np.eye(3), newK_try, (w, h), m1type=cv.CV_32F
                )

        self.R = R
        self.newK = newK
        self.map1, self.map2 = map1, map2
        self.maps_size = (w, h)

    def undistort_image(self, image):
        if image is None:
            raise ValueError("undistort_image got image=None (bad path or read failure)")

        h, w = image.shape[:2]
        size = (w, h)
        if self.map1 is None or self.maps_size != size:
            self._build_rectification_maps(size)

        rect = cv.remap(image, self.map1, self.map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
        mx = self.map1[..., 0]
        my = self.map1[..., 1]
        valid = np.mean((mx >= 0) & (mx < w) & (my >= 0) & (my < h))
        # print(f"valid mapped pixels: {valid*100:.1f}%")
        return rect

    def show_image(self, image):
        undistorted = self.undistort_image(cv.imread(image))
        win = "Undistorted Image"
        cv.namedWindow(win, cv.WINDOW_NORMAL)
        while True:
            cv.imshow(win, undistorted)

            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        cv.destroyAllWindows()

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

    def record_charuco_images(self,
            num_target=50,
            min_delay_s=0.4,
            min_corners=20,
            cam_index=2,
            width=1280, height=720,
    ):
        os.makedirs(self.image_path, exist_ok=True)

        aruco = cv.aruco
        dict_id = aruco.getPredefinedDictionary(self.dict_name)
        board = aruco.CharucoBoard((self.squaresX, self.squaresY), self.squareLength, self.markerLength, dict_id)
        params = aruco.DetectorParameters()

        cap = cv.VideoCapture(cam_index)  # CAP_DSHOW on Windows; drop on Linux/mac
        if not cap.isOpened():
            print("ERROR: Cannot open camera.")
            return

        # Set endoscope resolution
        cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

        # Try to lock auto-exposure
        try:
            cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)
            cap.set(cv.CAP_PROP_AUTOFOCUS, 0)
        except Exception:
            pass

        saved = 0
        last_save_t = 0.0
        last_pose = None  # (rvec, tvec) for diversity check
        win = "Calib capture (q=quit, s=save)"
        cv.namedWindow(win, cv.WINDOW_NORMAL)

        while saved < num_target:
            ok, frame = cap.read()
            if not ok:
                continue

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(gray, dict_id, parameters=params)

            overlay = frame.copy()
            status = f"{saved}/{num_target} saved | corners: 0"

            if ids is not None and len(ids) > 0:
                aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedCorners=[])
                cv.aruco.drawDetectedMarkers(overlay, corners, ids)

                ok_interp, ch_corners, ch_ids = aruco.interpolateCornersCharuco(
                    markerCorners=corners, markerIds=ids, image=gray, board=board
                )

                nchar = 0 if (not ok_interp or ch_corners is None) else len(ch_corners)
                status = f"{saved}/{num_target} saved | corners: {nchar}"

                if ok_interp and nchar >= min_corners:
                    corners3d = board.getChessboardCorners()
                    obj = corners3d[np.int32(ch_ids).ravel()].astype(np.float32).reshape(-1, 1, 3)
                    imgp = ch_corners.reshape(-1, 1, 2).astype(np.float32)

                    fx = fy = 0.8 * width
                    cx, cy = width / 2, height / 2
                    K_guess = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

                    ok_pnp, rvec, tvec = cv.solvePnP(obj, imgp, K_guess, None, flags=cv.SOLVEPNP_ITERATIVE)
                    keep = ok_pnp

                    # Ensure angle or distance changed enough
                    if ok_pnp and last_pose is not None:
                        r_last, t_last = last_pose
                        R1, _ = cv.Rodrigues(r_last);
                        R2, _ = cv.Rodrigues(rvec)
                        dR = R2 @ R1.T
                        angle = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
                        dt = np.linalg.norm(tvec - t_last)
                        # thresholds (tune): >= 10° or translation change
                        if angle < 10.0 and dt < 0.15:
                            keep = False
                            status += " | duplicate pose"
                    else:
                        keep = ok_pnp

                    # Rate limiting
                    if keep and (time.time() - last_save_t) < min_delay_s:
                        keep = False

                    # Save if good
                    if keep:
                        fname = os.path.join(self.image_path, f"calib_{saved:03d}.jpg")
                        cv.imwrite(fname, frame)
                        saved += 1
                        last_save_t = time.time()
                        last_pose = (rvec, tvec)
                        status += " | SAVED"

            cv.putText(overlay, status, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)
            cv.imshow(win, overlay)

            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                fname = os.path.join(self.image_path, f"calib_{saved:03d}.jpg")
                cv.imwrite(fname, frame)
                saved += 1
                last_save_t = time.time()
                status += " | MANUAL SAVE"

        cap.release()
        cv.destroyAllWindows()
        print(f"Done. Saved {saved} images to {self.image_path}")

    def capture_images(self, save_dir="test_images", cam_index=2):
        os.makedirs(save_dir, exist_ok=True)
        cap = cv.VideoCapture(cam_index)

        if not cap.isOpened():
            print(f"Cannot open camera index {cam_index}")
            return

        print("Press ENTER to save a frame, 'q' to quit.")

        counter = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            cv.imshow("Camera", frame)
            key = cv.waitKey(1) & 0xFF

            if key == 13:  # Enter
                filename = os.path.join(save_dir, f"test_{counter:03d}.jpg")
                cv.imwrite(filename, frame)
                print(f"Saved {filename}")
                counter += 1
            elif key == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    # Camera(init=True).record_charuco_images()
    camera = Camera()
    # camera.show_image("calib_imgs/calib_001.jpg")
    camera.capture_images()