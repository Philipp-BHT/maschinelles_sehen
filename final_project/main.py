from camera import Camera
import cv2 as cv
import numpy as np
import glob, os
from patchmask import TemplateMasks


class NeedleDetect:
    def __init__(self,
                 dict_name=cv.aruco.DICT_4X4_1000,
                 marker_length_m=0.018,
                 draw_axes_len=0.05,
                 needle_spec=None):
        self.camera = Camera()
        self.camera_pos = None       # 3D position of camera in marker/world coords (x,y,z) [m]
        self.camera_R_world = None   # 3x3 rotation of camera in world coords (R_wc)
        self.aruco_pose = None       # dict with rvec, tvec, T_mc, T_cw
        self.needle_pos = None

        self.template_detector = TemplateMasks()

        self.marker_length = marker_length_m
        self.draw_axes_len = draw_axes_len

        self.aruco_dict = cv.aruco.getPredefinedDictionary(dict_name)
        self.aruco_params = cv.aruco.DetectorParameters()

        self._plane_n = None
        self._plane_p0 = None

    @staticmethod
    def rvec_tvec_to_T(rvec, tvec):
        """Marker->Camera 4x4 transform."""
        R, _ = cv.Rodrigues(rvec)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = tvec.reshape(3)
        return T

    @staticmethod
    def invert_T(T):
        """Invert a 4x4 rigid transform."""
        R = T[:3, :3]
        t = T[:3, 3]
        Ti = np.eye(4, dtype=np.float64)
        Ti[:3, :3] = R.T
        Ti[:3, 3] = -R.T @ t
        return Ti

    def aruco_reproj_rms(self, rvec, tvec, detected_corners_rect):
        # 3D marker corners in marker/world coords (origin at center)
        L = self.marker_length * 0.5
        obj = np.array([[-L, L, 0],
                        [L, L, 0],
                        [L, -L, 0],
                        [-L, -L, 0]], np.float32).reshape(-1, 1, 3)
        proj, _ = cv.projectPoints(obj, rvec, tvec, self.camera.newK, None)  # rectified => no dist
        proj = proj.reshape(-1, 2)
        det = detected_corners_rect.reshape(-1, 2).astype(np.float32)
        err = np.linalg.norm(proj - det, axis=1)
        return float(err.mean()), float(err.std())


    def detect_aruco(self, frame_bgr, draw=True):
        """
        Detect a single ArUco marker in the frame and compute camera pose
        relative to that marker (treated as world).
        Returns a dict or None if not found.
        """
        rect = self.camera.undistort_image(frame_bgr)
        gray = cv.cvtColor(rect, cv.COLOR_BGR2GRAY)

        corners, ids, _ = cv.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        if ids is None or len(ids) == 0:
            self.aruco_pose = None
            self.camera_pos = None
            self.camera_R_world = None
            return None

        rvecs, tvecs, _objPts = cv.aruco.estimatePoseSingleMarkers(
            corners, self.marker_length, self.camera.newK, None
        )

        rvec = rvecs[0].reshape(3, 1)
        tvec = tvecs[0].reshape(3, 1)
        T_mc = self.rvec_tvec_to_T(rvec, tvec)    # marker->camera
        T_cw = self.invert_T(T_mc)                # camera->marker(world)

        cam_pos_w = T_cw[:3, 3]
        R_wc = T_cw[:3, :3]

        if draw:
            cv.aruco.drawDetectedMarkers(rect, corners, ids)
            cv.drawFrameAxes(rect, self.camera.newK, None, rvec, tvec, self.draw_axes_len)

        proj_error = self.aruco_reproj_rms(rvec, tvec, np.array(corners))

        self.camera_pos = cam_pos_w.copy()
        self.camera_R_world = R_wc.copy()
        self.aruco_pose = {
            "rvec_m2c": rvec,           # marker->camera rotation vector
            "tvec_m2c": tvec,           # marker->camera translation (m)
            "T_marker_to_cam": T_mc,    # 4x4
            "T_cam_to_world": T_cw,     # 4x4 (world == marker frame)
            "camera_pos_world": cam_pos_w,  # (x,y,z) [m]
            "camera_R_world": R_wc,
            "error": proj_error
        }

        return {"image": rect, "pose": self.aruco_pose}

    @staticmethod
    def auto_canny(img, sigma=0.33):
        v = np.median(img)
        lo = int(max(0, (1.0 - sigma) * v))
        hi = int(min(255, (1.0 + sigma) * v))
        return cv.Canny(img, lo, hi, L2gradient=True)

    @staticmethod
    def fit_circle_least_squares(points):
        """
        Algebraic circle fit (Taubin-ish). points: (N,2) float32
        Returns (ok, center(x,y), radius)
        """
        if len(points) < 3:
            return False, None, None
        x = points[:, 0]
        y = points[:, 1]
        x_m = x.mean()
        y_m = y.mean()
        u = x - x_m
        v = y - y_m
        Suu = (u * u).sum()
        Suv = (u * v).sum()
        Svv = (v * v).sum()
        Suuu = (u * u * u).sum()
        Svvv = (v * v * v).sum()
        Suvv = (u * v * v).sum()
        Svuu = (v * u * u).sum()

        A = np.array([[Suu, Suv], [Suv, Svv]], dtype=np.float64)
        b = 0.5 * np.array([Suuu + Suvv, Svvv + Svuu], dtype=np.float64)
        try:
            uc, vc = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return False, None, None
        xc = x_m + uc
        yc = y_m + vc
        r = np.sqrt((x - xc) ** 2 + (y - yc) ** 2).mean()
        return True, (float(xc), float(yc)), float(r)

    def pose_from_T_cam_to_world(self, T_cw):
        """Extract rotation R_wc and camera center C_w from camera->world 4x4."""
        R_wc = T_cw[:3, :3]
        C_w = T_cw[:3, 3]  # camera origin in world coords
        return R_wc, C_w

    def pixel_to_plane_3d(self, px, newK, T_cam_to_world, plane=None, plane_n=None, plane_d=None, plane_p0=None):
        """
        Intersect a pixel ray with a world plane.

        Args:
          px: (x,y) pixel in the RECTIFIED image.
          newK: 3x3 intrinsics used for rectified image.
          T_cam_to_world: 4x4 transform (camera -> world).
          plane: optional tuple ('nd', n, d) or ('np0', n, p0)
          plane_n, plane_d, plane_p0: alternative ways to pass the plane.

        Returns:
          X_w: 3D point in world coords (np.array shape (3,)), or None if parallel.
        """
        K = newK.astype(np.float64)
        Kinv = np.linalg.inv(K)

        x, y = float(px[0]), float(px[1])
        p_cam_dir = Kinv @ np.array([x, y, 1.0], dtype=np.float64)  # direction (not normalized)
        p_cam_dir = p_cam_dir / np.linalg.norm(p_cam_dir)

        # Camera->World pose
        R_wc, C_w = self.pose_from_T_cam_to_world(T_cam_to_world)
        # Ray in world: X_w(t) = C_w + t * d_w
        d_w = (R_wc @ p_cam_dir)  # rotate direction into world

        if plane is not None:
            key, a, b = plane
            if key == 'nd':
                n_w, d = np.array(a, dtype=np.float64), float(b)
            elif key == 'np0':
                n_w, p0_w = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
                d = -float(n_w @ p0_w)
            else:
                raise ValueError("plane must be ('nd', n, d) or ('np0', n, p0)")
        else:
            if plane_n is None:
                raise ValueError("Provide plane as ('nd', n, d) or ('np0', n, p0) or plane_n/plane_d.")
            n_w = np.array(plane_n, dtype=np.float64)
            if plane_p0 is not None:
                d = -float(n_w @ np.array(plane_p0, dtype=np.float64))
            elif plane_d is not None:
                d = float(plane_d)
            else:
                raise ValueError("If plane_n is given, also give plane_d or plane_p0.")

        denom = float(n_w @ d_w)
        if abs(denom) < 1e-9:
            return None  # Ray is (nearly) parallel to plane

        t = - (n_w @ C_w + d) / denom
        X_w = C_w + t * d_w
        return X_w

    def set_plane_parallel_to_marker(self, offset_m: float):
        """
        Define the intersection plane as parallel to the ArUco marker plane (Z=0 in marker/world),
        offset by 'offset_m' meters along +Z (marker normal).
        Use a negative value to go 'behind' the tag (along -Z).
        """
        self._plane_n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self._plane_p0 = np.array([0.0, 0.0, float(offset_m)], dtype=np.float64)

    def tip_px_to_world(self, tip_px_rect):
        """
        Intersect pixel ray with the currently selected plane.
        Falls back to the marker plane (Z=0) if no plane was set.
        """
        if (self.aruco_pose is None) or (self.camera.newK is None):
            return None, False

        T_cw = self.aruco_pose["T_cam_to_world"]
        if hasattr(self, "_plane_n") and hasattr(self, "_plane_p0"):
            X_w = self.pixel_to_plane_3d(
                px=tip_px_rect, newK=self.camera.newK, T_cam_to_world=T_cw,
                plane=('np0', self._plane_n, self._plane_p0)
            )
        else:
            X_w = self.pixel_to_plane_3d(
                px=tip_px_rect, newK=self.camera.newK, T_cam_to_world=T_cw,
                plane=('np0', np.array([0, 0, 1.0]), np.array([0, 0, 0]))
            )
        return X_w, (X_w is not None)

    @staticmethod
    def project_world_point_to_rect_px(X_w, rvec, tvec, K):
        """
        X_w: (3,) in marker/world coords (meters)
        rvec,tvec: marker->camera pose from your ArUco (same as you already have)
        K: rectified intrinsics (nd.camera.newK)
        Returns integer pixel tuple (x,y) on the rectified image.
        """
        X = np.asarray(X_w, dtype=np.float32).reshape(1,1,3)
        pts2d, _ = cv.projectPoints(X, rvec, tvec, K, distCoeffs=None)
        x, y = pts2d.reshape(2)
        return int(round(x)), int(round(y))

    @staticmethod
    def draw_vector_center_to_tip(rect_img, center_px, tip_px, color=(0,255,255), thickness=1):
        """
        Draws a thin arrow from ArUco center pixel -> needle tip pixel on the rectified image.
        """
        vis = rect_img.copy()
        cx, cy = map(int, center_px)
        tx, ty = map(int, tip_px)

        cv.circle(vis, (cx, cy), 3, (0,255,255), -1, cv.LINE_AA)
        cv.circle(vis, (tx, ty), 4, (0,0,255),   -1, cv.LINE_AA)

        cv.arrowedLine(vis, (cx, cy), (tx, ty), color, thickness, cv.LINE_AA, tipLength=0.02)
        return vis

    def run_needle_detection(self,
                            static=True,
                            image_glob="test_images/*.jpg",
                            cam_index=0,
                            match_every=1,  # run heavy template match every N frames (1 = every frame)
                            plane_offset_m=0.029):

        templates = self.template_detector.load_templates("templates")
        if not templates:
            print("No templates found in ./templates")
            return
        tpl_feats = self.template_detector.load_template_features(templates)

        def frames_from_images(pattern):
            for path in sorted(glob.glob(pattern)):
                img = cv.imread(path)
                if img is None: continue
                yield img, os.path.basename(path)

        def frames_from_camera(index):
            cap = cv.VideoCapture(index)
            if not cap.isOpened():
                print(f"Could not open camera {index}")
                return
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok: break
                    yield frame, None
            finally:
                cap.release()

        source = frames_from_images(image_glob) if static else frames_from_camera(cam_index)

        win = "ArUco center → Needle tip"
        cv.namedWindow(win, cv.WINDOW_NORMAL)

        last_tip = None
        frame_id = 0
        try:
            for img, name in source:
                ar = self.detect_aruco(img, draw=True)
                if ar is not None:
                    rect = ar["image"]
                    cam_pos = ar["pose"]["camera_pos_world"]
                    print(f"Camera pos (m): {cam_pos}")
                else:
                    rect = self.camera.undistort_image(img)

                roi = self.template_detector.build_roi_mask(rect)
                need_match = (frame_id % max(1, int(match_every)) == 0) or (last_tip is None)
                best = None
                if need_match:
                    best = self.template_detector.match_templates_feature(rect, tpl_feats, roi_mask=roi, use_roi=True)
                    if best is None:
                        print("No match with ROI; retrying without ROI…")
                        best = self.template_detector.match_templates_feature(rect, tpl_feats, roi_mask=None,
                                                                              use_roi=False)
                    if best is not None and best.get("tip_px_scene") is not None:
                        last_tip = best["tip_px_scene"]

                if last_tip is None:
                    cv.imshow(win, rect)
                    key = cv.waitKey(0 if static else 1) & 0xFF
                    if key in (27, ord('q')): break
                    frame_id += 1
                    continue

                tip_px_rect = last_tip

                self.set_plane_parallel_to_marker(offset_m=plane_offset_m)
                X_w, ok = self.tip_px_to_world(tip_px_rect)
                dist_m = None
                if ok:
                    dist_m = float(np.linalg.norm(X_w - np.array([0.0, 0.0, 0.0])))
                    print(f"{name or '[live]'} | Tip world: {X_w} | dist: {dist_m * 1000:.1f} mm")
                else:
                    print("Ray parallel to plane — no intersection")

                vis = rect.copy()
                if ar is not None:
                    rvec = self.aruco_pose["rvec_m2c"]
                    tvec = self.aruco_pose["tvec_m2c"]
                    center_px = self.project_world_point_to_rect_px([0.0, 0.0, 0.0], rvec, tvec, self.camera.newK)
                    vis = self.draw_vector_center_to_tip(vis, center_px, tip_px_rect, color=(0, 255, 255), thickness=1)
                    if dist_m is not None:
                        midx = int((center_px[0] + tip_px_rect[0]) / 2)
                        midy = int((center_px[1] + tip_px_rect[1]) / 2)
                        cv.putText(vis, f"{dist_m * 1000:.1f} mm", (midx + 6, midy - 6),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv.LINE_AA)
                else:
                    tx, ty = map(int, tip_px_rect)
                    cv.circle(vis, (tx, ty), 5, (0, 0, 255), -1, cv.LINE_AA)
                    cv.putText(vis, "No ArUco", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv.LINE_AA)

                if name:
                    cv.putText(vis, name, (10, vis.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                               cv.LINE_AA)
                cv.imshow(win, vis)

                key = cv.waitKey(0 if static else 1) & 0xFF
                if key in (27, ord('q')):
                    break
                elif not static and key == ord('p'):
                    while True:
                        k2 = cv.waitKey(20) & 0xFF
                        if k2 in (27, ord('q'), ord('p')):
                            if k2 in (27, ord('q')):
                                raise SystemExit
                            break

                frame_id += 1
        finally:
            cv.destroyAllWindows()


if __name__ == "__main__":
    nd = NeedleDetect(marker_length_m=0.018)
    nd.run_needle_detection()
