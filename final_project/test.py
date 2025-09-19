# --- needle_lowS_thin.py ---
import cv2 as cv
import numpy as np
import glob, os

def _auto_canny(img, sigma=0.33):
    v = np.median(img); lo = int(max(0, (1-sigma)*v)); hi = int(min(255,(1+sigma)*v))
    return cv.Canny(img, lo, hi, L2gradient=True)

def _fit_circle(points):
    if len(points) < 12: return False, None, None
    x = points[:,0].astype(np.float64); y = points[:,1].astype(np.float64)
    x_m, y_m = x.mean(), y.mean(); u, v = x-x_m, y-y_m
    Suu, Suv, Svv = np.dot(u,u), np.dot(u,v), np.dot(v,v)
    Suuu, Svvv = np.dot(u,u*u), np.dot(v,v*v)
    Suvv, Svuu = np.dot(u,v*v), np.dot(v,u*u)
    A = np.array([[Suu, Suv],[Suv, Svv]]); b = 0.5*np.array([Suuu+Suvv, Svvv+Svuu])
    try: uc, vc = np.linalg.solve(A,b)
    except np.linalg.LinAlgError: return False, None, None
    cx, cy = x_m+uc, y_m+vc
    r = np.mean(np.sqrt((x-cx)**2+(y-cy)**2))
    return True, (float(cx), float(cy)), float(r)

def detect_needle_v2(frame_bgr, curved=True, needle_spec=None, draw=True):
    """
    needle_spec keys (all optional):
      type: "line"|"arc"  (overrides 'curved')
      sat_max: 40         # low saturation
      v_min:  60          # ignore dark
      v_max: 230          # ignore glaring highlights
      tophat_k: 9         # ridge size (odd)
      open_iter: 1
      close_iter: 1
      min_len_px: 60      # min skeleton pixels
      radius_range: (30, 1000)  # in px (2D)
    """
    S = needle_spec or {}
    if "type" in S: curved = (S["type"]=="arc")
    sat_max  = int(S.get("sat_max", 40))
    v_min    = int(S.get("v_min",  60))
    v_max    = int(S.get("v_max", 230))
    k        = int(S.get("tophat_k", 9));  k += (k%2==0)  # make odd
    it_open  = int(S.get("open_iter", 1))
    it_close = int(S.get("close_iter", 1))
    min_len  = int(S.get("min_len_px", 60))
    rr       = S.get("radius_range", None)

    img = frame_bgr.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 1) Color cue: low saturation metal (ignore very dark + very bright)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    H,Sv,V = cv.split(hsv)
    lowS = cv.inRange(hsv, (0,0,v_min), (179,sat_max,v_max))

    # 2) Geometry cue: thin bright-ish ridges (top-hat) + edges
    se = cv.getStructuringElement(cv.MORPH_RECT, (k,1))
    toph = cv.morphologyEx(gray, cv.MORPH_TOPHAT, se)
    # also rotate SE by 45 and 90 to be less directional
    se2 = cv.getStructuringElement(cv.MORPH_RECT, (1,k))
    toph = cv.max(toph, cv.morphologyEx(gray, cv.MORPH_TOPHAT, se2))
    edges = _auto_canny(gray, 0.33)

    # 3) Combine + clean
    _, th = cv.threshold(toph, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    combo = cv.bitwise_and(th, lowS)
    combo = cv.bitwise_or(combo, edges)  # allow faint needles via edges
    el = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    if it_close: combo = cv.morphologyEx(combo, cv.MORPH_CLOSE, el, iterations=it_close)
    if it_open:  combo = cv.morphologyEx(combo,  cv.MORPH_OPEN,  el, iterations=it_open)

    # 4) Keep the largest elongated component
    num, lab, stats, _ = cv.connectedComponentsWithStats(combo, 8)
    if num <= 1:
        return {"image": img, "mask": combo, "centerline": None, "model": None, "tip_px": None}
    areas = stats[1:, cv.CC_STAT_AREA]; idx = 1+int(np.argmax(areas))
    mask = (lab==idx).astype(np.uint8)*255

    # 5) Skeleton (use thinning if present)
    try:
        import cv2.ximgproc as xip
        skel = xip.thinning(mask, thinningType=xip.THINNING_ZHANGSUEN)
    except Exception:
        skel = cv.bitwise_and(mask, edges)
    ys, xs = np.where(skel>0)
    if len(xs) < min_len:
        return {"image": img, "mask": mask, "centerline": None, "model": None, "tip_px": None}
    pts = np.stack([xs,ys], axis=1).astype(np.float32)

    model = None; tip = None

    if not curved:
        # Fit line
        [vx,vy,x0,y0] = cv.fitLine(pts, cv.DIST_L2, 0, 0.01, 0.01).flatten()
        v = np.array([vx,vy], np.float64); p0 = np.array([x0,y0], np.float64)
        projs = ((pts - p0) @ v).ravel()
        end = int(np.argmax(np.abs(projs))); tip = tuple(pts[end].astype(int))
        model = {"type":"line","params":{"vx":float(vx),"vy":float(vy),"x0":float(x0),"y0":float(y0)}}
        if draw:
            p1 = (int(x0-800*vx), int(y0-800*vy))
            p2 = (int(x0+800*vx), int(y0+800*vy))
            cv.line(img, p1, p2, (0,255,0), 1, cv.LINE_AA)
    else:
        # Fit circle (arc)
        if len(pts) > 2000: pts = pts[::max(1,len(pts)//2000)]
        ok, c, r = _fit_circle(pts)
        if ok and rr is not None:
            rmin, rmax = rr; ok = (rmin <= r <= rmax)
        if ok:
            model = {"type":"arc","params":{"cx":c[0],"cy":c[1],"r":r}}
            if draw: cv.circle(img, (int(c[0]),int(c[1])), int(r), (0,255,0), 1, cv.LINE_AA)
            d = np.linalg.norm(pts - np.array(c, np.float32), axis=1)
            tip = tuple(pts[int(np.argmax(d))].astype(int))

    if draw:
        cnts,_ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(img, cnts, -1, (255,0,0), 1, cv.LINE_AA)
        if tip is not None: cv.circle(img, tip, 4, (0,0,255), -1, cv.LINE_AA)

    return {"image": img, "mask": mask, "centerline": pts, "model": model, "tip_px": tip}


def demo_batch(input_glob="test_images/*.jpg", curved=True):
    for fn in sorted(glob.glob(input_glob)):
        img = cv.imread(fn)
        out = detect_needle_v2(img, curved=curved, needle_spec={
            "sat_max": 50, "v_min": 40, "v_max": 230,
            "tophat_k": 9, "open_iter": 1, "close_iter": 1,
            "min_len_px": 80, "radius_range": (30, 1e6)
        }, draw=True)
        vis = out["image"]
        cv.imshow("needle", vis)
        print(fn, "tip_px:", out["tip_px"], "model:", out["model"])
        if cv.waitKey(0) & 0xFF == 27: break
    cv.destroyAllWindows()

if __name__ == "__main__":
    demo_batch()
