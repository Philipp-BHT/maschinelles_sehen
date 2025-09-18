# needle_detect.py
import cv2 as cv
import numpy as np
from camera import Camera

def rvec_tvec_to_T(rvec, tvec):
    """Marker->Camera 4x4 transform."""
    R, _ = cv.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3]  = tvec.reshape(3)
    return T

def invert_T(T):
    """Invert a 4x4 rigid transform."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3]  = -R.T @ t
    return Ti

class NeedleDetect:
    def __init__(self,
                 dict_name=cv.aruco.DICT_5X5_1000,
                 marker_length_m=0.040,     # <-- set this to your printed marker side (meters)
                 draw_axes_len=0.05,
                 needle_spec=None):
        self.camera = Camera()
        self.camera_pos = None       # 3D position of camera in marker/world coords (x,y,z) [m]
        self.camera_R_world = None   # 3x3 rotation of camera in world coords (R_wc)
        self.aruco_pose = None       # dict with rvec, tvec, T_mc, T_cw
        self.needle_pos = None

        self.marker_length = marker_length_m
        self.draw_axes_len = draw_axes_len

        self.aruco_dict = cv.aruco.getPredefinedDictionary(dict_name)
        self.aruco_params = cv.aruco.DetectorParameters_create()


    def detect_aruco(self, frame_bgr, draw=True):
        """
        Detect a single ArUco marker in the frame and compute camera pose
        relative to that marker (treated as world).
        Returns a dict or None if not found.
        """
        # 1) Rectify (fisheye -> pinhole-like) using Camera; this builds self.camera.newK
        rect = self.camera.undistort_image(frame_bgr)
        gray = cv.cvtColor(rect, cv.COLOR_BGR2GRAY)

        # 2) Detect markers
        corners, ids, _ = cv.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        if ids is None or len(ids) == 0:
            self.aruco_pose = None
            self.camera_pos = None
            self.camera_R_world = None
            return None

        # 3) Pose for single markers (marker->camera), using rectified intrinsics
        rvecs, tvecs, _objPts = cv.aruco.estimatePoseSingleMarkers(
            corners, self.marker_length, self.camera.newK, None  # distortion is None after rectification
        )

        # (If multiple markers are visible, pick the first or apply your own selection strategy)
        rvec = rvecs[0].reshape(3, 1)
        tvec = tvecs[0].reshape(3, 1)
        T_mc = rvec_tvec_to_T(rvec, tvec)    # marker->camera
        T_cw = invert_T(T_mc)                # camera->marker(world)

        # Camera position in world (marker) coords: the translation of T_cw
        cam_pos_w = T_cw[:3, 3]
        # Camera rotation in world coords: R_wc
        R_wc = T_cw[:3, :3]

        # 4) Draw overlays
        if draw:
            cv.aruco.drawDetectedMarkers(rect, corners, ids)
            cv.drawFrameAxes(rect, self.camera.newK, None, rvec, tvec, self.draw_axes_len)

        # 5) Save & return
        self.camera_pos = cam_pos_w.copy()
        self.camera_R_world = R_wc.copy()
        self.aruco_pose = {
            "rvec_m2c": rvec,           # marker->camera rotation vector
            "tvec_m2c": tvec,           # marker->camera translation (m)
            "T_marker_to_cam": T_mc,    # 4x4
            "T_cam_to_world": T_cw,     # 4x4 (world == marker frame)
            "camera_pos_world": cam_pos_w,  # (x,y,z) [m]
            "camera_R_world": R_wc
        }

        return {"image": rect, "pose": self.aruco_pose}


    @staticmethod
    def auto_canny(img, sigma=0.33):
        # automatic thresholds based on median
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


    def detect_needle(self, frame_bgr, draw=True, curved=True):
        """
        Returns:
          {
            "image": rectified BGR with optional overlays,
            "mask": binary mask of candidate needle pixels,
            "centerline_pts": Nx2 float32 (possibly sparse sample),
            "model": {"type": "line" or "arc", "params": ...},
            "tip_px": (x,y) or None
          }
        """
        rect = self.camera.undistort_image(frame_bgr)

        # --- 1) Contrast boost ---
        gray = cv.cvtColor(rect, cv.COLOR_BGR2GRAY)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        g = clahe.apply(gray)

        # --- 2) Specular mask (bright + low saturation) ---
        hsv = cv.cvtColor(rect, cv.COLOR_BGR2HSV)
        H,S,V = cv.split(hsv)
        # Thresholds to tune:
        V_thr = 200   # high value
        S_thr =  60   # low saturation (reflective)
        spec = cv.inRange(hsv, (0,0,V_thr), (179,S_thr,255))

        # --- 3) Edge/ridge cues ---
        edges = self.auto_canny(g, sigma=0.33)
        # White top-hat to enhance thin bright ridges
        se = cv.getStructuringElement(cv.MORPH_RECT, (7,7))
        tophat = cv.morphologyEx(g, cv.MORPH_TOPHAT, se)
        _, th_bin = cv.threshold(tophat, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

        # --- 4) Combine and clean ---
        combo = cv.bitwise_or(spec, th_bin)
        combo = cv.bitwise_or(combo, edges)
        combo = cv.morphologyEx(combo, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3)), iterations=1)
        combo = cv.morphologyEx(combo, cv.MORPH_OPEN,  cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3)), iterations=1)

        # --- 5) Keep only the largest thin component (likely the needle) ---
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(combo, connectivity=8)
        if num_labels <= 1:
            return {"image": rect, "mask": combo, "centerline_pts": None, "model": None, "tip_px": None}

        # choose largest non-background
        areas = stats[1:, cv.CC_STAT_AREA]
        best_idx = 1 + int(np.argmax(areas))
        mask = (labels == best_idx).astype(np.uint8)*255

        # optional thinning to get centerline
        # (ximgproc.thinning is great if available; fallback: distance ridge sampling)
        try:
            import cv2.ximgproc as xip
            skel = xip.thinning(mask, thinningType=xip.THINNING_ZHANGSUEN)
        except Exception:
            # simple fallback: use edges intersected with mask
            skel = cv.bitwise_and(edges, mask)

        ys, xs = np.where(skel > 0)
        centerline_pts = np.stack([xs, ys], axis=1).astype(np.float32) if len(xs)>0 else None

        model = None
        tip = None

        if centerline_pts is None or len(centerline_pts) < 10:
            return {"image": rect, "mask": mask, "centerline_pts": centerline_pts, "model": None, "tip_px": None}

        if not curved:
            # --- Straight model: fit a line ---
            # returns unit direction vec and a point on line
            [vx, vy, x0, y0] = cv.fitLine(centerline_pts, cv.DIST_L2, 0, 0.01, 0.01).flatten()
            model = {"type": "line", "params": {"vx":float(vx), "vy":float(vy), "x0":float(x0), "y0":float(y0)}}

            # project points to line and pick endpoint with max projection
            v = np.array([vx, vy], dtype=np.float64)
            p0 = np.array([x0, y0], dtype=np.float64)
            projs = ((centerline_pts - p0) @ v).ravel()
            end_idx = int(np.argmax(np.abs(projs)))
            tip = tuple(centerline_pts[end_idx].astype(int))
            if draw:
                p1 = (int(x0 - 500*vx), int(y0 - 500*vy))
                p2 = (int(x0 + 500*vx), int(y0 + 500*vy))
                cv.line(rect, p1, p2, (0,255,0), 1, cv.LINE_AA)
        else:
            # --- Curved model: fit a circle (arc) ---
            # optionally subsample to speed up / denoise
            if len(centerline_pts) > 2000:
                centerline_pts = centerline_pts[::int(len(centerline_pts)/2000)]
            ok, c, r = self.fit_circle_least_squares(centerline_pts)
            if ok:
                model = {"type": "arc", "params": {"cx":c[0], "cy":c[1], "r":r}}
                if draw:
                    cv.circle(rect, (int(c[0]), int(c[1])), int(r), (0,255,0), 1, cv.LINE_AA)
                # pick tip: furthest point along arc boundary closest to image bright spot
                # simple heuristic: choose endpoint of skeleton farthest from circle center
                dists = np.linalg.norm(centerline_pts - np.array(c, dtype=np.float32), axis=1)
                idx = int(np.argmax(dists))
                tip = tuple(centerline_pts[idx].astype(int))

        # draw overlays
        if draw:
            # mask contour
            cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(rect, cnts, -1, (255,0,0), 1, cv.LINE_AA)
            if tip is not None:
                cv.circle(rect, tip, 4, (0,0,255), -1, cv.LINE_AA)

        return {
            "image": rect,
            "mask": mask,
            "centerline_pts": centerline_pts,
            "model": model,
            "tip_px": tip
        }

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
        # Unpack intrinsics
        K = newK.astype(np.float64)
        Kinv = np.linalg.inv(K)

        # Build normalized ray in camera coords
        x, y = float(px[0]), float(px[1])
        p_cam_dir = Kinv @ np.array([x, y, 1.0], dtype=np.float64)  # direction (not normalized)
        p_cam_dir = p_cam_dir / np.linalg.norm(p_cam_dir)

        # Camera->World pose
        R_wc, C_w = self.pose_from_T_cam_to_world(T_cam_to_world)
        # Ray in world: X_w(t) = C_w + t * d_w
        d_w = (R_wc @ p_cam_dir)  # rotate direction into world

        # Plane parsing
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

if __name__ == "__main__":
    needle_spec = {
        "type": "arc",  # "line" or "arc"
        "radius_range": (8e-3, 25e-3),  # meters, if you’ll lift to 3D (or pixels if staying 2D)
        "length_range_px": (80, 600),
        "thickness_px": 3,  # expected visual thickness after rectification
        "specular": {"V_thr": 200, "S_thr": 60},
        "tophat_kernel": 7,  # odd size in px
        "morph_open": 1,  # iterations
        "morph_close": 1,  # iterations
        "curvature_sign": None,  # +1, -1, or None if unknown
        "prefer_tip_brightest": True
    }

    nd = NeedleDetect(marker_length_m=0.050, needle_spec=needle_spec)

    img = cv.imread("placeholder_image.png")
    ar = nd.detect_aruco(img, draw=True)
    dn = nd.detect_needle(img, draw=True)
    cv.imshow("result", dn["image"])
    cv.waitKey(0)