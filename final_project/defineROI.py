import cv2 as cv
import numpy as np
import os

def roi_tuner(image_path,
              H_lo=0, S_lo=66, V_lo=21,
              H_hi=50, S_hi=150, V_hi=255,
              k=11, it_open=5, it_close=5,
              min_area=0, out_prefix="roi"):

    img = cv.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]

    win = "ROI Tuner (q=quit, s=save)"
    cv.namedWindow(win, cv.WINDOW_NORMAL)

    # --- Trackbars (create once, then set starting positions) ---
    cv.createTrackbar("H_low",  win, 0,   179, lambda v: None)
    cv.createTrackbar("S_low",  win, 0,   255, lambda v: None)
    cv.createTrackbar("V_low",  win, 0,   255, lambda v: None)
    cv.createTrackbar("H_high", win, 0,   179, lambda v: None)
    cv.createTrackbar("S_high", win, 0,   255, lambda v: None)
    cv.createTrackbar("V_high", win, 0,   255, lambda v: None)

    cv.createTrackbar("Blur k", win, 1,  31,  lambda v: None)   # allow larger kernels too
    cv.createTrackbar("Open",   win, 0,   10,  lambda v: None)
    cv.createTrackbar("Close",  win, 0,   10,  lambda v: None)
    cv.createTrackbar("MinArea",win, 0,  (w*h), lambda v: None)

    # ---- set initial positions from args (clamped) ----
    H_lo = int(np.clip(H_lo, 0, 179)); S_lo = int(np.clip(S_lo, 0, 255)); V_lo = int(np.clip(V_lo, 0, 255))
    H_hi = int(np.clip(H_hi, 0, 179)); S_hi = int(np.clip(S_hi, 0, 255)); V_hi = int(np.clip(V_hi, 0, 255))
    k    = int(max(1, min(k, 31)));         # >0
    if k % 2 == 0: k += 1                    # make odd
    it_open  = int(np.clip(it_open,  0, 10))
    it_close = int(np.clip(it_close, 0, 10))
    min_area = int(np.clip(min_area, 0, w*h))

    cv.setTrackbarPos("H_low",  win, H_lo)
    cv.setTrackbarPos("S_low",  win, S_lo)
    cv.setTrackbarPos("V_low",  win, V_lo)
    cv.setTrackbarPos("H_high", win, H_hi)
    cv.setTrackbarPos("S_high", win, S_hi)
    cv.setTrackbarPos("V_high", win, V_hi)
    cv.setTrackbarPos("Blur k", win, k)
    cv.setTrackbarPos("Open",   win, it_open)
    cv.setTrackbarPos("Close",  win, it_close)
    cv.setTrackbarPos("MinArea",win, min_area)

    # ... (rest of your loop unchanged)

    # Optional: vignette mask to ignore black circular border (assume center crop)
    vignette_mask = np.ones((h, w), np.uint8)*255
    # detect dark border automatically (simple threshold)
    g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    thr = max(5, int(np.percentile(g, 2)))
    _, dark = cv.threshold(g, thr, 255, cv.THRESH_BINARY)
    # Keep bright region (not border)
    vignette_mask = dark

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    while True:
        # Read sliders
        H_lo = cv.getTrackbarPos("H_low",  win)
        S_lo = cv.getTrackbarPos("S_low",  win)
        V_lo = cv.getTrackbarPos("V_low",  win)
        H_hi = cv.getTrackbarPos("H_high", win)
        S_hi = cv.getTrackbarPos("S_high", win)
        V_hi = cv.getTrackbarPos("V_high", win)

        k = cv.getTrackbarPos("Blur k", win)
        k = k if k % 2 == 1 else max(1, k-1)
        k = max(1, k)

        it_open  = cv.getTrackbarPos("Open",  win)
        it_close = cv.getTrackbarPos("Close", win)
        min_area = max(0, cv.getTrackbarPos("MinArea", win))


        # Pre-blur (works on HSV’s V channel visually)
        if k > 1:
            hsv_blur = hsv.copy()
            hsv_blur[:,:,2] = cv.GaussianBlur(hsv[:,:,2], (k, k), 0)
        else:
            hsv_blur = hsv

        # Threshold
        lower = np.array([H_lo, S_lo, V_lo], np.uint8)
        upper = np.array([H_hi, S_hi, V_hi], np.uint8)
        mask = cv.inRange(hsv_blur, lower, upper)

        # Ignore vignette/black border
        mask = cv.bitwise_and(mask, vignette_mask)

        # Morphology
        se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
        if it_close > 0:
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, se, iterations=it_close)
        if it_open > 0:
            mask = cv.morphologyEx(mask,  cv.MORPH_OPEN,  se, iterations=it_open)

        # Keep only the largest area (likely the silicone pad)
        num, lab, stats, _ = cv.connectedComponentsWithStats(mask, 8)
        if num > 1:
            areas = stats[1:, cv.CC_STAT_AREA]
            idx = 1 + int(np.argmax(areas))
            big = (lab == idx).astype(np.uint8)*255
            if areas.max() >= min_area:
                mask = big
            else:
                mask = np.zeros_like(mask)

        # Overlay
        overlay = img.copy()
        cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv.drawContours(overlay, cnts, -1, (0,255,0), 2, cv.LINE_AA)

        # Stack preview: original | mask | overlay
        mask_bgr = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        vis_top  = np.hstack([img, mask_bgr, overlay])
        scale = min(1.0, 1200.0 / vis_top.shape[1])
        vis = cv.resize(vis_top, (int(vis_top.shape[1]*scale), int(vis_top.shape[0]*scale)))

        cv.imshow(win, vis)
        key = cv.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            base = os.path.splitext(os.path.basename(image_path))[0]
            out_mask = f"{out_prefix}_{base}_mask.png"
            out_over = f"{out_prefix}_{base}_overlay.png"
            cv.imwrite(out_mask, mask)
            cv.imwrite(out_over, overlay)
            print(f"Saved: {out_mask}, {out_over}")

    cv.destroyAllWindows()
    return H_lo, S_lo, V_lo, H_hi, S_hi, V_hi, k, it_open, it_close, min_area

if __name__ == "__main__":
    H_lo = 0
    S_lo = 66
    V_lo = 21
    H_hi = 50
    S_hi = 150
    V_hi = 255
    k = 11
    it_open = 5
    it_close = 5
    min_area = 0
    for image in os.listdir("test_images"):
        H_lo, S_lo, V_lo, H_hi, S_hi, V_hi, k, it_open, it_close, min_area  = roi_tuner(f"test_images/{image}", H_lo, S_lo, V_lo, H_hi, S_hi, V_hi, k, it_open, it_close, min_area)
        print(f"{image}\n"
              f"H_lo {H_lo}\n"
              f"S_lo {S_lo}\n"
              f"V_lo {V_lo}\n "
              f"H_hi {H_hi}\n"
              f"S_hi {S_hi}\n"
              f"V_hi {V_hi}\n"
              f"k {k}\n"
              f"it_open {it_open}\n"
              f"it_close {it_close}\n"
              f"min_area {min_area}\n\n")
