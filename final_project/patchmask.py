import cv2 as cv
import numpy as np
import os, glob
import json


class TemplateMasks:
    ROI_PARAMS = dict(H_lo=0, S_lo=36, V_lo=44, H_hi=56, S_hi=147, V_hi=255, k=11, it_open=5, it_close=5, min_area=0)

    @staticmethod
    def _norm_for_matcher(name):
        return cv.NORM_L2 if name in ("SIFT", "AKAZE") else cv.NORM_HAMMING

    @staticmethod
    def _detect_desc(det, img_u8, mask=None):
        if mask is not None:
            mask = (mask > 0).astype(np.uint8)
            if mask.shape != img_u8.shape:
                mask = cv.resize(mask, (img_u8.shape[1], img_u8.shape[0]), interpolation=cv.INTER_NEAREST)
        kps, des = det.detectAndCompute(img_u8, mask)
        return kps or [], des

    @staticmethod
    def load_templates(folder="templates"):
        pairs = []
        for img_path in sorted(glob.glob(os.path.join(folder, "*_patch.png"))):
            base = os.path.splitext(os.path.basename(img_path))[0].replace("_patch","")
            mask_path = os.path.join(folder, f"{base}_mask.png")
            meta_path = os.path.join(folder, f"{base}_meta.json")
            t = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            m = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
            tip = None
            if os.path.exists(meta_path):
                try:
                    tip = json.load(open(meta_path))["tip_px"]  # [x,y] or None
                    if tip is not None: tip = tuple(tip)
                except Exception: pass
            if t is None or m is None:
                print("skip template:", img_path); continue
            _, m = cv.threshold(m, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
            pairs.append((t, m, tip))
        return pairs

    @staticmethod
    def _prep_gray_u8(img):
        if img.ndim == 3:
            g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            g = img
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(g)

    @staticmethod
    def _make_detector_chain():
        dets = []
        if hasattr(cv, "SIFT_create"):
            dets.append(("SIFT", cv.SIFT_create(nfeatures=2000)))
        if hasattr(cv, "AKAZE_create"):
            dets.append(("AKAZE", cv.AKAZE_create()))
        dets.append(("ORB", cv.ORB_create(
            nfeatures=4000, scaleFactor=1.2, nlevels=8,
            edgeThreshold=8, patchSize=31, fastThreshold=7, WTA_K=4)))
        return dets

    def detect_template_descriptors(self, tpl_u8, mask_u8=None):
        """
        Returns: (keypoints, descriptors, detector_name)
        - tpl_u8: uint8 gray (or BGR; will be converted)
        - mask_u8: optional uint8 binary mask (0/255); will be resized if needed
        Tries SIFT -> AKAZE -> ORB and returns the first that yields descriptors.
        """
        img_u8 = self._prep_gray_u8(tpl_u8)
        m = None
        if mask_u8 is not None:
            m = (mask_u8 > 0).astype(np.uint8)
            if m.shape != img_u8.shape:
                m = cv.resize(m, (img_u8.shape[1], img_u8.shape[0]), interpolation=cv.INTER_NEAREST)
            m = cv.dilate(m, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3)), 1)

        for name, det in self._make_detector_chain():
            kps, des = det.detectAndCompute(img_u8, m)
            if des is not None and len(kps) > 0:
                return kps, des, name
        return [], None, "none"

    def load_template_features(self, templates):
        """
        templates: list of (tpl_u8, mask_u8_or_None, tip_px_or_None)
        Returns: list of dicts with kps, des, box, tip_px, det_name
        """
        feats = []
        for idx, (tpl_u8, mask_u8, tip_px) in enumerate(templates):
            kps, des, used = self.detect_template_descriptors(tpl_u8, mask_u8)
            h, w = tpl_u8.shape[:2]
            box = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
            feats.append({
                "kps": kps, "des": des, "box": box,
                "tip_px": tip_px, "det_name": used
            })
        return feats

    def match_templates_feature(self, scene_bgr, tpl_feats, roi_mask=None,
                                use_roi=True,
                                ratio=0.9,
                                xcheck=True,
                                ransac_thr=6.0,
                                min_good=6,
                                min_inliers=6,
                                scene_scales=(0.8, 1.0, 1.25)):  # <<< NEW
        """
        Multi-scale version: tries the scene at several scales and returns the best match
        mapped back into the original-resolution coordinate system.
        """

        def _to_u8gray(img):
            return self._prep_gray_u8(img)

        def _resize(img, s):
            h, w = img.shape[:2]
            return cv.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv.INTER_LINEAR)

        def _compose_scale_back(A_2x3, s):
            """Map affine (template->scaled_scene) back to original scene coords."""
            A = np.vstack([A_2x3, [0, 0, 1]])  # 3x3
            S_inv = np.array([[1 / s, 0, 0],
                              [0, 1 / s, 0],
                              [0, 0, 1]], dtype=np.float64)
            A_orig = S_inv @ A
            return A_orig[:2, :]

        def _affine_rms(src_inl, A, dst_inl):
            """RMS reprojection (px) for inliers."""
            src2 = cv.transform(src_inl[None, ...], A)[0]  # (N,2)
            err = np.linalg.norm(src2 - dst_inl, axis=1)
            return float(np.sqrt(np.mean(err * err))) if len(err) else 1e9

        # --- Precompute scene pyramids (gray + ROI) ---
        scene_u8_full = _to_u8gray(scene_bgr)
        roi_full = None
        if use_roi and roi_mask is not None:
            roi_full = (roi_mask > 0).astype(np.uint8)
            if roi_full.shape != scene_u8_full.shape:
                roi_full = cv.resize(roi_full, (scene_u8_full.shape[1], scene_u8_full.shape[0]),
                                     interpolation=cv.INTER_NEAREST)

        pyramids = []
        for s in scene_scales:
            g = scene_u8_full if s == 1.0 else _resize(scene_u8_full, s)
            r = None if roi_full is None else (roi_full if s == 1.0 else _resize(roi_full, s))
            pyramids.append((s, g, r))

        # --- Build detector chain once ---
        dets = self._make_detector_chain()

        best_overall = None
        # Iterate scales
        for s, scene_u8, scene_mask in pyramids:
            # Scene features at this scale
            kpi = []
            dei = None
            used_scene = "none"
            norm_scene = None
            for name, det in dets:
                kpi, dei = self._detect_desc(det, scene_u8, scene_mask)
                if dei is not None and len(kpi) > 0:
                    used_scene = name
                    norm_scene = self._norm_for_matcher(name)
                    break
            print(
                f"[SCENE scale={s:.2f}] det={used_scene} kps={len(kpi)} (ROI on={use_roi and scene_mask is not None})")
            if dei is None or len(kpi) == 0:
                continue

            # Try all templates against this scale
            for i, T in enumerate(tpl_feats):
                if T["des"] is None or len(T["kps"]) == 0:
                    continue

                bf = cv.BFMatcher(norm_scene, crossCheck=False) # BruteForce Matcher
                knn = bf.knnMatch(T["des"], dei, k=2)
                good = [m for m, n in knn if n is not None and m.distance < ratio * n.distance]

                if xcheck:
                    bf_x = cv.BFMatcher(norm_scene, crossCheck=True)
                    xmatches = bf_x.match(T["des"], dei)
                    idx_pairs = {(m.queryIdx, m.trainIdx) for m in good}
                    for xm in xmatches:
                        idx_pairs.add((xm.queryIdx, xm.trainIdx))
                    good = [cv.DMatch(_queryIdx=q, _trainIdx=t, _imgIdx=0, _distance=0) for (q, t) in idx_pairs]

                if len(good) < min_good:
                    continue

                src = np.float32([T["kps"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)  # template
                dst = np.float32([kpi[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)  # scaled scene

                A_s, inliers = cv.estimateAffinePartial2D(
                    src, dst, method=cv.RANSAC, ransacReprojThreshold=ransac_thr,
                    maxIters=4000, confidence=0.995, refineIters=20
                )
                if A_s is None or inliers is None:
                    continue
                inl_mask = inliers.ravel().astype(bool)
                inl = int(inl_mask.sum())
                if inl < min_inliers:
                    continue

                # Score with reprojection RMS for quality
                rms = _affine_rms(src[inl_mask].reshape(-1, 2), A_s, dst[inl_mask].reshape(-1, 2))
                score = inl / (rms + 1e-6)

                # Map affine back to original scene coords
                A_orig = _compose_scale_back(A_s, s)

                # Project box & tip in original coords
                box = T["box"].reshape(-1, 2).astype(np.float32)
                box_proj = cv.transform(box[None, ...], A_orig)[0].astype(int).reshape(-1, 1, 2)

                tip_scene = None
                if T.get("tip_px") is not None:
                    tip_tpl = np.array([[T["tip_px"]]], dtype=np.float32)
                    tip_scene_f = cv.transform(tip_tpl, A_orig)[0, 0]
                    tip_scene = (float(tip_scene_f[0]), float(tip_scene_f[1]))

                cand = {
                    "A": A_orig,
                    "inliers": inl,
                    "rms": rms,
                    "score": score,
                    "tpl_idx": i,
                    "box": box_proj,
                    "scene_det": used_scene,
                    "tip_px_scene": tip_scene
                }
                if (best_overall is None) or (cand["score"] > best_overall["score"]):
                    best_overall = cand

        return best_overall

    @staticmethod
    def draw_feature_match(vis, best):
        if best is None:
            cv.putText(vis, "No match", (10,25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv.LINE_AA)
            return vis
        cv.polylines(vis, [best["box"].reshape(-1,2)], True, (0,255,0), 2, cv.LINE_AA)
        if best.get("tip_px_scene") is not None:
            tx, ty = map(int, best["tip_px_scene"])
            cv.circle(vis, (tx, ty), 5, (0,0,255), -1, cv.LINE_AA)
            cv.putText(vis, "TIP", (tx+6, ty-6), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv.LINE_AA)
        cv.putText(vis, f"inliers={best['inliers']} tpl#{best['tpl_idx']} ({best['scene_det']})",
                   (10,25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv.LINE_AA)
        return vis

    def build_roi_mask(self, img_bgr):
        p = self.ROI_PARAMS
        hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
        k = p["k"] if p["k"] % 2 == 1 else p["k"]+1
        hsv_blur = hsv.copy()
        if k > 1: hsv_blur[:,:,2] = cv.GaussianBlur(hsv[:,:,2], (k,k), 0)
        lower = np.array([p["H_lo"], p["S_lo"], p["V_lo"]], np.uint8)
        upper = np.array([p["H_hi"], p["S_hi"], p["V_hi"]], np.uint8)
        mask = cv.inRange(hsv_blur, lower, upper)
        g = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
        _, bright = cv.threshold(g, max(5, int(np.percentile(g, 2))), 255, cv.THRESH_BINARY)
        mask = cv.bitwise_and(mask, bright)
        se = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
        if p["it_close"]: mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, se, p["it_close"])
        if p["it_open"] : mask = cv.morphologyEx(mask,  cv.MORPH_OPEN,  se, p["it_open"])
        num, lab, stats, _ = cv.connectedComponentsWithStats(mask, 8)
        if num > 1:
            idx = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA]))
            mask = (lab==idx).astype(np.uint8)*255
        return mask

    def build_template_masks(self):
        templates = []
        for img_path in sorted(glob.glob(os.path.join("templates", "*_patch.png"))):
            t = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            if t is None:
                continue
            templates.append((t, None))

        if not templates:
            print("No templates found in ./templates")
            raise SystemExit

        templates = self.load_templates("templates")
        tpl_feats = self.load_template_features(templates)

        # Iterate test images
        image_paths = sorted(glob.glob(os.path.join("test_images", "*.jpg")))
        for path in image_paths:
            img = cv.imread(path)
            if img is None:
                continue

            roi = self.build_roi_mask(img)

            # Try with ROI ON first; if that yields no scene kps, try ROI OFF automatically
            best = self.match_templates_feature(img, tpl_feats, roi_mask=roi, use_roi=True)
            if best is None:
                print("No match with ROI; retrying without ROI…")
                best = self.match_templates_feature(img, tpl_feats, roi_mask=None, use_roi=False)

            vis = self.draw_feature_match(img.copy(), best)

            left = img.copy()
            left[roi == 0] = 0
            stacked = np.hstack([left, vis])
            cv.imshow("feature match (ROI | result)", stacked)
            key = cv.waitKey(0) & 0xFF
            if key in (27, ord('q')):
                break

        cv.destroyAllWindows()


if __name__ == "__main__":
    masks = TemplateMasks()
    masks.build_template_masks()


