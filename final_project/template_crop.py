import cv2 as cv
import numpy as np
import glob, os
import json

def _click_tip(patch):
    win = "Click needle TIP (Enter=save, c=cancel)"
    tip = [None, None]
    def on_mouse(e,x,y,flags,param):
        if e == cv.EVENT_LBUTTONDOWN:
            tip[0], tip[1] = int(x), int(y)
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    cv.setMouseCallback(win, on_mouse)
    while True:
        vis = patch.copy()
        if tip[0] is not None:
            cv.circle(vis, tuple(tip), 4, (0,0,255), -1, cv.LINE_AA)
        cv.imshow(win, vis)
        k = cv.waitKey(10) & 0xFF
        if k in (13,10): break   # Enter
        if k in (27, ord('c')): tip = [None, None]; break
    cv.destroyWindow(win)
    return None if tip[0] is None else (tip[0], tip[1])


def crop_templates(image_paths, out_dir="templates"):
    os.makedirs(out_dir, exist_ok=True)
    for path in image_paths:
        print(path)
        img = cv.imread(path)
        if img is None:
            print("Failed:", path)
            continue

        win = "crop: drag box, Enter to save, c to cancel, q to quit"
        r = cv.selectROI(win, img, showCrosshair=True, fromCenter=False)
        cv.destroyWindow(win)
        if r == (0,0,0,0):
            continue
        x,y,w,h = map(int, r)
        patch = img[y:y+h, x:x+w].copy()

        # soft edge mask (helps matching)
        g = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
        e = cv.Canny(g, 50, 150, L2gradient=True)
        e = cv.dilate(e, cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3)), 1)
        mask = e  # 0/255

        tip_xy = _click_tip(patch)
        base = os.path.splitext(os.path.basename(path))[0]
        out_img = os.path.join(out_dir, f"{base}_patch.png")
        out_mask = os.path.join(out_dir, f"{base}_mask.png")
        out_meta = os.path.join(out_dir, f"{base}_meta.json")
        cv.imwrite(out_img, patch)
        cv.imwrite(out_mask, mask)
        meta = {"tip_px": tip_xy}  # (x,y) in template patch coords
        with open(out_meta, "w") as f:
            json.dump(meta, f)
        print("Saved:", out_img, out_mask, out_meta)

# ---------------------------
# ROI mask (yours)
# ---------------------------
ROI_PARAMS = dict(H_lo=0, S_lo=36, V_lo=44, H_hi=56, S_hi=147, V_hi=255, k=11, it_open=5, it_close=5, min_area=0)

def build_roi_mask(img_bgr, p=ROI_PARAMS):
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

# ---------------------------
# Templates loader (yours)
# ---------------------------
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
                import json
                tip = json.load(open(meta_path))["tip_px"]  # [x,y] or None
                if tip is not None: tip = tuple(tip)
            except Exception: pass
        if t is None or m is None:
            print("skip template:", img_path); continue
        _, m = cv.threshold(m, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        pairs.append((t, m, tip))
    return pairs



# ---------------------------
# Matching helpers (NEW)
# ---------------------------
def prep_gray(img_bgr_or_gray):
    """CLAHE + standardize; returns float32."""
    if img_bgr_or_gray.ndim == 3:
        g = cv.cvtColor(img_bgr_or_gray, cv.COLOR_BGR2GRAY)
    else:
        g = img_bgr_or_gray
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(g)
    g = g.astype(np.float32)
    g = (g - g.mean()) / (g.std() + 1e-6)
    return g

def grad_mag(img_bgr_or_gray):
    """Sobel magnitude + standardize; returns float32."""
    if img_bgr_or_gray.ndim == 3:
        g = cv.cvtColor(img_bgr_or_gray, cv.COLOR_BGR2GRAY)
    else:
        g = img_bgr_or_gray
    g = cv.GaussianBlur(g, (3,3), 0)
    gx = cv.Sobel(g, cv.CV_32F, 1,0,ksize=3)
    gy = cv.Sobel(g, cv.CV_32F, 0,1,ksize=3)
    mag = cv.magnitude(gx, gy)
    mag = (mag - mag.mean()) / (mag.std() + 1e-6)
    return mag.astype(np.float32)

def rotate_image_and_mask(tpl, msk, angle_deg):
    h, w = tpl.shape[:2]
    M = cv.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    tpl_r = cv.warpAffine(tpl, M, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    msk_r = cv.warpAffine(msk, M, (w, h), flags=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT, borderValue=0)
    return tpl_r, msk_r

# ---------------------------
# Interactive matcher (UPDATED)
# ---------------------------
def interactive_match_one(image_path, templates):
    img = cv.imread(image_path)
    if img is None:
        print("Missing:", image_path)
        return

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h, w = gray.shape
    roi = build_roi_mask(img, ROI_PARAMS)

    # neutralize outside ROI
    med = int(np.median(gray[roi>0])) if np.any(roi) else int(np.median(gray))
    scene_gray = gray.copy()
    scene_gray[roi==0] = med

    win = f"Match Tuner - {os.path.basename(image_path)} (q next)"
    cv.namedWindow(win, cv.WINDOW_NORMAL)

    # trackbars
    cv.createTrackbar("template idx",            win, 0, max(0, len(templates)-1), lambda v: None)
    cv.createTrackbar("s_min x100",              win, 60, 300, lambda v: None)  # 0.60
    cv.createTrackbar("s_max x100",              win, 140, 300, lambda v: None) # 1.40
    cv.createTrackbar("n_scales",                win, 5, 10, lambda v: None)
    cv.createTrackbar("ang_min (-30..0)",        win, 18, 90, lambda v: None)   # maps to -value
    cv.createTrackbar("ang_max (0..30)",         win, 18, 90, lambda v: None)   # maps to +value
    cv.createTrackbar("n_angles",                win, 7, 19, lambda v: None)    # odd count is nice
    cv.createTrackbar("method 0=CCORR 1=CCOEFF", win, 0, 1,  lambda v: None)
    cv.createTrackbar("use gradient",            win, 1, 1,  lambda v: None)
    cv.createTrackbar("mask dilate",             win, 1, 5,  lambda v: None)
    cv.createTrackbar("show heat",               win, 1, 1,  lambda v: None)

    def compute():
        ti   = cv.getTrackbarPos("template idx", win)
        smin = max(10, cv.getTrackbarPos("s_min x100", win)) / 100.0
        smax = max(smin, cv.getTrackbarPos("s_max x100", win)) / 100.0
        nsc  = max(1, cv.getTrackbarPos("n_scales", win))
        a_min_deg = -cv.getTrackbarPos("ang_min (-30..0)", win)  # negative
        a_max_deg =  cv.getTrackbarPos("ang_max (0..30)",  win)  # positive
        nang = max(1, cv.getTrackbarPos("n_angles", win))
        method_idx = cv.getTrackbarPos("method 0=CCORR 1=CCOEFF", win)
        method = cv.TM_CCORR_NORMED if method_idx == 0 else cv.TM_CCOEFF_NORMED
        use_grad = (cv.getTrackbarPos("use gradient", win) == 1)
        dil_it = cv.getTrackbarPos("mask dilate", win)

        tpl_u8, msk_u8 = templates[ti]
        if dil_it > 0:
            msk_u8 = cv.dilate(msk_u8, cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3)), dil_it)

        # scene representation
        sceneF = grad_mag(scene_gray) if use_grad else prep_gray(scene_gray)

        # scales & angles
        scales  = np.linspace(smin, smax, nsc).tolist()
        angles  = np.linspace(a_min_deg, a_max_deg, nang).tolist()

        heat = np.full((h, w), -1.0, np.float32)
        best = (-1.0, (0,0,0,0), ti, 1.0, 0.0)

        for ang in angles:
            tpl_r, msk_r = rotate_image_and_mask(tpl_u8, msk_u8, ang)
            for s in scales:
                th = max(8, int(round(tpl_r.shape[0]*s)))
                tw = max(8, int(round(tpl_r.shape[1]*s)))
                if th >= h or tw >= w:
                    continue
                tplS = cv.resize(tpl_r, (tw, th), interpolation=cv.INTER_LINEAR)
                mskS = cv.resize(msk_r, (tw, th), interpolation=cv.INTER_NEAREST)

                tplF = grad_mag(tplS) if use_grad else prep_gray(tplS)

                res = cv.matchTemplate(sceneF, tplF, method, mask=mskS)

                minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(res)
                x0, y0 = maxLoc
                x1, y1 = x0 + tw, y0 + th
                if maxVal > best[0]:
                    best = (float(maxVal), (x0,y0,x1,y1), ti, s, ang)

                # crude heat: fill the found window with its score (fast)
                heat[y0:y1, x0:x1] = np.maximum(heat[y0:y1, x0:x1], float(maxVal))

        valid = heat >= 0
        if np.any(valid):
            hsub = heat[valid]
            mn, mx = float(hsub.min()), float(hsub.max())
            if mx > mn:
                heat[valid] = (hsub - mn) / (mx - mn)
            else:
                heat[valid] = 0.0
            heat[~valid] = 0.0
        else:
            heat[:] = 0.0

        return best, heat, (ti, smin, smax, nsc, a_min_deg, a_max_deg, nang, method_idx, int(use_grad), dil_it)

    last_params = None
    best, heat, last_params = compute()

    while True:
        params = (
            cv.getTrackbarPos("template idx", win),
            cv.getTrackbarPos("s_min x100", win),
            cv.getTrackbarPos("s_max x100", win),
            cv.getTrackbarPos("n_scales", win),
            -cv.getTrackbarPos("ang_min (-30..0)", win),
            cv.getTrackbarPos("ang_max (0..30)", win),
            cv.getTrackbarPos("n_angles", win),
            cv.getTrackbarPos("method 0=CCORR 1=CCOEFF", win),
            cv.getTrackbarPos("use gradient", win),
            cv.getTrackbarPos("mask dilate", win),
            cv.getTrackbarPos("show heat", win),
        )

        if params[:-1] != last_params:
            best, heat, last_params = compute()

        score, (x0,y0,x1,y1), ti, s, ang = best
        vis = img.copy()
        cv.rectangle(vis, (x0,y0), (x1,y1), (0,255,0), 2, cv.LINE_AA)
        cv.putText(vis, f"templ#{ti} scale={s:.2f} ang={ang:+.1f} score={score:.3f}",
                   (10,25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv.LINE_AA)

        left = img.copy(); left[roi==0] = 0
        show_heat = (params[-1] == 0)
        if show_heat:
            heat_color = cv.applyColorMap((heat*255).astype(np.uint8), cv.COLORMAP_JET)
            heat_blend = cv.addWeighted(heat_color, 0.5, img, 0.5, 0)
            right = heat_blend
        else:
            right = vis

        stack = np.hstack([left, right])
        scale = min(1.0, 1600.0 / stack.shape[1])
        stack = cv.resize(stack, (int(stack.shape[1]*scale), int(stack.shape[0]*scale)))
        cv.imshow(win, stack)

        key = cv.waitKey(10) & 0xFF
        if key == ord('q') or key == 27:
            break

    cv.destroyWindow(win)

# ---------------------------
# Driver
# ---------------------------
if __name__ == "__main__":
    create_templates = True
    if create_templates:
        images = [os.path.join("test_images", image) for image in os.listdir("test_images")]
        crop_templates(images)
    quit()

    templates = load_templates("templates")
    if not templates:
        print("No templates found. Run the cropper first.")
        raise SystemExit

    for img_path in sorted(glob.glob("test_images/*.jpg")):
        interactive_match_one(img_path, templates)
