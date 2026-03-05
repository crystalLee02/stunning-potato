import glob
import math
import os

import cv2
import numpy as np

from ultralytics import YOLO


# ==========================
# 显示工具（防止太大）
# ==========================
def show_resized(winname, img, max_size=900):
    h, w = img.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale)
    cv2.imshow(winname, img)


# ==========================
# 端子能量曲线可视化
# ==========================
def plot_energy_curve_cv(e_s, xBodyL=None, xBodyR=None, xL_sub=None, xR_sub=None, xL_step=None, xR_step=None, h=240):
    """把 1D 能量曲线 e_s 画成 BGR 图，方便 cv2.imshow - 白色：能量曲线 - 青色：xBodyL/xBodyR - 绿色：xL_sub/xR_sub - 紫色：xL_step/xR_step（可选）.
    """
    e = np.asarray(e_s, dtype=np.float32).reshape(-1)
    n = int(e.shape[0])
    w = n

    # 归一化到 [0,1]
    e_min, e_max = float(np.min(e)), float(np.max(e))
    if e_max - e_min < 1e-6:
        en = np.zeros_like(e)
    else:
        en = (e - e_min) / (e_max - e_min)

    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    xs = np.arange(w, dtype=np.int32)
    ys = (h - 1 - (en * (h - 1))).astype(np.int32)
    pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], False, (255, 255, 255), 1)

    def vline(x, color, thick=1):
        if x is None:
            return
        xi = int(np.clip(round(float(x)), 0, w - 1))
        cv2.line(canvas, (xi, 0), (xi, h - 1), color, thick)

    # 主体边界：青色
    vline(xBodyL, (255, 255, 0), 1)
    vline(xBodyR, (255, 255, 0), 1)

    # 亮度台阶：紫色（可选）
    vline(xL_step, (255, 0, 255), 1)
    vline(xR_step, (255, 0, 255), 1)

    # 端点：绿色
    vline(xL_sub, (0, 255, 0), 2)
    vline(xR_sub, (0, 255, 0), 2)

    cv2.putText(
        canvas,
        f"Energy curve (n={n}) min={e_min:.1f} max={e_max:.1f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (180, 180, 180),
        2,
    )
    cv2.putText(
        canvas, "Cyan: body | Green: xL/xR | Magenta: step", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2
    )

    return canvas


# ==========================
# 你的 PCA 姿态归一化（原样保留）
# ==========================
def normalize_pose_pca(roi_bgr, pad_ratio=0.3):
    h, w = roi_bgr.shape[:2]

    pad_h = int(h * pad_ratio)
    pad_w = int(w * pad_ratio)

    roi_pad = cv2.copyMakeBorder(roi_bgr, pad_h, pad_h, pad_w, pad_w, borderType=cv2.BORDER_REPLICATE)

    gray = cv2.cvtColor(roi_pad, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 80, 180)

    ys, xs = np.where(edges > 0)
    if len(xs) < 200:
        return roi_bgr, 0.0

    pts = np.column_stack((xs, ys)).astype(np.float32)
    _, eigenvectors = cv2.PCACompute(pts, mean=None)
    vx, vy = eigenvectors[0]
    angle = math.degrees(math.atan2(vy, vx))

    H, W = roi_pad.shape[:2]
    center = (W // 2, H // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        roi_pad,
        M,
        (W, H),
        flags=cv2.INTER_LINEAR,
        # 你原来用常数填充：保留
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=int(np.median(gray)),
    )

    y1 = pad_h
    y2 = pad_h + h
    x1 = pad_w
    x2 = pad_w + w
    roi_norm = rotated[y1:y2, x1:x2]
    return roi_norm, angle


# ==========================
# 收集图片
# ==========================
def collect_images(input_path):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    input_path = os.path.normpath(input_path)

    if os.path.isfile(input_path):
        return [input_path] if input_path.lower().endswith(exts) else []

    if os.path.isdir(input_path):
        files = []
        for e in exts:
            files += glob.glob(os.path.join(input_path, f"*{e}"))
        files.sort()
        return files

    files = glob.glob(input_path)
    files = [p for p in files if os.path.isfile(p) and p.lower().endswith(exts)]
    files.sort()
    return files


# ==========================
# YOLO：按类别取置信度最高的框
# ==========================
def get_best_box_by_class(r, cls_id: int):
    """返回 (xyxy_int, conf_float)；如果不存在该类返回 (None, 0.0)."""
    if r.boxes is None or len(r.boxes) == 0:
        return None, 0.0

    xyxy = r.boxes.xyxy.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    conf = r.boxes.conf.cpu().numpy().astype(float)

    m = cls == cls_id
    if m.sum() == 0:
        return None, 0.0

    idxs = np.where(m)[0]
    best_i = idxs[int(np.argmax(conf[idxs]))]
    x1, y1, x2, y2 = xyxy[best_i]
    return (int(x1), int(y1), int(x2), int(y2)), float(conf[best_i])


def clamp_box(x1, y1, x2, y2, W, H, pad=0):
    x1 = max(0, min(W - 1, x1 - pad))
    y1 = max(0, min(H - 1, y1 - pad))
    x2 = max(0, min(W, x2 + pad))
    y2 = max(0, min(H, y2 + pad))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None
    return x1, y1, x2, y2


# ==========================
# 过长裁剪
# ==========================
def center_crop_to_width(img, target_w=580):
    """若图像宽度 > target_w，则左右等量裁剪到 target_w; 返回 (cropped_img, x_offset)."""
    _H, W = img.shape[:2]
    if W <= target_w:
        return img, 0
    x0 = (W - target_w) // 2
    x1 = x0 + target_w
    cropped = img[:, x0:x1].copy()
    return cropped, x0


# ==========================
# 1D 平滑/投影/段提取（为测量服务）
# ==========================
def smooth_1d(x, sigma=2.5):
    if sigma <= 0:
        return x
    k = int(max(3, sigma * 6)) | 1
    g = cv2.GaussianBlur(x.reshape(-1, 1).astype(np.float32), (k, 1), sigmaX=sigma, sigmaY=0)
    return g.reshape(-1)


def morph_close_1d(mask01, k=51):
    k = int(k) | 1
    kernel = np.ones((k,), np.uint8)
    dil = np.convolve(mask01.astype(np.uint8), kernel, mode="same")
    dil = (dil > 0).astype(np.uint8)
    ero = np.convolve(dil, kernel, mode="same")
    ero = (ero >= k).astype(np.uint8)
    return ero


def find_longest_true_segment(mask01, x_min=0, x_max=None, min_len=10):
    n = len(mask01)
    if x_max is None:
        x_max = n
    x_min = max(0, int(x_min))
    x_max = min(n, int(x_max))
    if x_max <= x_min:
        return None

    m = mask01[x_min:x_max]
    if m.sum() == 0:
        return None

    best = None
    best_len = -1
    start = None
    for i, v in enumerate(m):
        if v and start is None:
            start = i
        if (not v) and start is not None:
            end = i
            L = end - start
            if L > best_len and L >= min_len:
                best_len = L
                best = (start + x_min, end + x_min)
            start = None
    if start is not None:
        end = len(m)
        L = end - start
        if L > best_len and L >= min_len:
            best = (start + x_min, end + x_min)

    return best


def energy_profile_x(gray_enh, y1, y2):
    g = cv2.GaussianBlur(gray_enh, (3, 3), 0)
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    e = np.abs(sx)[y1:y2, :].sum(axis=0)
    e_s = smooth_1d(e, sigma=2.5)
    return e, e_s


# ==========================
# ✅ 亚像素峰值细化（核心）
# ==========================
def refine_peak_subpixel_1d(s, i):
    """三点抛物线拟合，把整数峰 i 细化成亚像素 float."""
    n = len(s)
    i = int(i)
    if i <= 0 or i >= n - 1:
        return float(i)
    y0, y1, y2 = float(s[i - 1]), float(s[i]), float(s[i + 1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return float(i)
    delta = 0.5 * (y0 - y2) / denom  # 通常 [-0.5, 0.5]
    delta = float(np.clip(delta, -0.5, 0.5))
    return float(i) + delta


def snap_peak_in_window(e_s, x_lo, x_hi):
    """只在窗口[x_lo,x_hi]里找最大峰（整数）."""
    n = len(e_s)
    x_lo = int(np.clip(x_lo, 0, n - 1))
    x_hi = int(np.clip(x_hi, 0, n - 1))
    if x_hi <= x_lo:
        return x_lo
    j = int(np.argmax(e_s[x_lo : x_hi + 1]))
    return x_lo + j


def snap_peak_subpix_in_window(e_s, x_lo, x_hi):
    """窗口内找峰（整数）+ 亚像素细化，返回 float."""
    i = snap_peak_in_window(e_s, x_lo, x_hi)
    return refine_peak_subpixel_1d(e_s, i)


def snap_to_local_peak_1d(e_s, x, radius):
    """以 x 为中心，在 [x-radius, x+radius] 内找整数峰."""
    n = len(e_s)
    l = max(0, int(x) - int(radius))
    r = min(n, int(x) + int(radius) + 1)
    if r <= l:
        return int(x)
    return int(np.argmax(e_s[l:r])) + l


def snap_to_local_peak_subpix_1d(e_s, x, radius):
    """以 x 为中心，局部窗口找峰 + 亚像素细化，返回 float."""
    i = snap_to_local_peak_1d(e_s, x, radius)
    return refine_peak_subpixel_1d(e_s, i)


def pick_segment_by_threshold(e_s, xL_body, xR_body, thr_percentile=75, close_k=61, min_len_ratio=0.20):
    body = e_s[xL_body : xR_body + 1]
    if body.size < 20:
        return None, None, None

    thr = float(np.percentile(body, thr_percentile))
    mask01 = (e_s >= thr).astype(np.uint8)

    mask01[:xL_body] = 0
    mask01[xR_body + 1 :] = 0

    mask01c = morph_close_1d(mask01, k=close_k)

    min_len = int(max(10, min_len_ratio * (xR_body - xL_body)))
    seg = find_longest_true_segment(mask01c, x_min=xL_body, x_max=xR_body + 1, min_len=min_len)
    return seg, thr, mask01c


# ==========================
# body mask（两套算法共用）
# ==========================
def build_body_mask_from_edges(gray):
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    med = np.median(g)
    lo = int(max(0, 0.66 * med))
    hi = int(min(255, 1.33 * med))
    edges = cv2.Canny(g, lo, hi)

    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)), iterations=2)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    H, W = gray.shape[:2]
    cx0, cy0 = W * 0.5, H * 0.5

    best = None
    best_score = -1
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.02 * H * W:
            continue
        m = cv2.moments(c)
        if m["m00"] <= 1e-6:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        dist = np.hypot(cx - cx0, cy - cy0)
        score = area - 2.0 * dist
        if score > best_score:
            best_score = score
            best = c

    if best is None:
        best = max(cnts, key=cv2.contourArea)

    mask = np.zeros((H, W), np.uint8)
    cv2.drawContours(mask, [best], -1, 255, thickness=-1)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21)), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=1)
    return mask


def build_body_mask_robust(gray):
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (th > 0).mean() > 0.65:
        th = cv2.bitwise_not(th)

    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k1, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k1, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k2, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    if num <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    mask = np.zeros_like(th)
    mask[labels == best] = 255
    return mask


def body_extent_in_band_robust(mask, y1, y2, occ_thr=0.06, close_k=31):
    _H, _W = mask.shape[:2]
    band = (mask[y1:y2, :] > 0).astype(np.uint8)
    occ = band.mean(axis=0)

    m = (occ >= occ_thr).astype(np.uint8)

    k = int(close_k) | 1
    kernel = np.ones((k,), np.uint8)
    dil = np.convolve(m, kernel, mode="same") > 0
    ero = np.convolve(dil.astype(np.uint8), kernel, mode="same") >= k
    m2 = ero.astype(np.uint8)

    best = None
    best_len = -1
    start = None
    for i, v in enumerate(m2):
        if v and start is None:
            start = i
        if (not v) and start is not None:
            L = i - start
            if L > best_len:
                best_len = L
                best = (start, i)
            start = None
    if start is not None:
        L = len(m2) - start
        if L > best_len:
            best = (start, len(m2))

    if best is None or best_len < 20:
        return None, None, occ

    xL, xR = best[0], best[1] - 1
    return int(xL), int(xR), occ


def edges_by_intensity_step(gray_enh, y1, y2, margin_ratio=0.06, smooth_sigma=6.0):
    _H, W = gray_enh.shape[:2]
    s = gray_enh[y1:y2, :].mean(axis=0).astype(np.float32)
    s_s = smooth_1d(s, sigma=smooth_sigma)
    g = np.abs(np.gradient(s_s))

    m = int(max(2, margin_ratio * W))
    L0, L1 = m, int(0.45 * W)
    R0, R1 = int(0.55 * W), W - 1 - m
    if L1 <= L0 or R1 <= R0:
        return None, None, s_s, g

    xL = int(np.argmax(g[L0 : L1 + 1])) + L0
    xR = int(np.argmax(g[R0 : R1 + 1])) + R0
    return xL, xR, s_s, g


# ============================================================
# 算法 B：压接不足测量（升级为亚像素）
# ============================================================
def measure_undercrimp_terminal(
    roi_norm_bgr,
    mm_per_px=0.0,
    band_top=0.28,
    band_bottom=0.72,
    thr_percentile=55,
    close_k=101,
    min_len_ratio=0.35,
    snap_ratio=0.06,
):
    gray = cv2.cvtColor(roi_norm_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enh = clahe.apply(gray)

    H, W = gray.shape[:2]
    y1 = int(band_top * H)
    y2 = int(band_bottom * H)
    y1 = max(0, min(y1, H - 2))
    y2 = max(y1 + 1, min(y2, H))

    # 亮度台阶先粗估（可失败）
    xL_step, xR_step, _, _ = edges_by_intensity_step(gray_enh, y1, y2, margin_ratio=0.06, smooth_sigma=6.0)
    if xL_step is None or xR_step is None or (xR_step - xL_step) < 0.35 * W:
        xL_step = int(0.05 * W)
        xR_step = int(0.95 * W)

    mask = build_body_mask_robust(gray_enh)
    if mask is None:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return None, None, None, None, vis, {"reason": "no_mask"}

    xBodyL, xBodyR, _ = body_extent_in_band_robust(mask, y1, y2, occ_thr=0.06, close_k=41)
    if xBodyL is None:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return None, None, None, None, vis, {"reason": "no_extent", "mask": mask}

    _, e_s = energy_profile_x(gray_enh, y1, y2)
    seg, thr, _ = pick_segment_by_threshold(
        e_s, xBodyL, xBodyR, thr_percentile=thr_percentile, close_k=close_k, min_len_ratio=min_len_ratio
    )

    body_w = float(xBodyR - xBodyL)
    used = "segment"
    if seg is None:
        xL0, xR0 = xBodyL, xBodyR
        used = "fallback_body"
    else:
        xL0, xR0 = seg[0], seg[1] - 1
        if (xR0 - xL0) < 0.35 * body_w:
            xL0, xR0 = xBodyL, xBodyR
            used = "fallback_short_seg"

    # ✅ 原来：整数峰
    # snap_r = int(max(8, snap_ratio * W))
    # xL = snap_to_local_peak_1d(e_s, xL0, snap_r)
    # xR = snap_to_local_peak_1d(e_s, xR0, snap_r)

    # ✅ 现在：整数峰 + 亚像素细化（float）
    snap_r = int(max(8, snap_ratio * W))
    xL_sub = snap_to_local_peak_subpix_1d(e_s, xL0, snap_r)
    xR_sub = snap_to_local_peak_subpix_1d(e_s, xR0, snap_r)

    # clamp（float）
    xL_sub = float(np.clip(xL_sub, xBodyL, xBodyR))
    xR_sub = float(np.clip(xR_sub, xBodyL, xBodyR))

    if xR_sub <= xL_sub:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return None, None, None, None, vis, {"reason": "xR<=xL", "mask": mask}

    # 硬兜底：防止两线挤一起（逻辑仍保留，但用 float）
    seg_w = float(xR_sub - xL_sub)
    if seg_w < 0.35 * body_w or seg_w < 0.25 * float(W):
        # 兜底回 body 边界（仍可亚像素：用整数即可）
        xL_sub = float(xBodyL)
        xR_sub = float(xBodyR)
        used = "fallback_too_short"

    length_px = float(xR_sub - xL_sub)
    length_mm = length_px * float(mm_per_px) if (mm_per_px and mm_per_px > 0) else None

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(vis, (0, y1), (W - 1, y2), (255, 255, 0), 1)

    # ✅ 绘制必须 int
    xL_i = int(np.clip(round(xL_sub), 0, W - 1))
    xR_i = int(np.clip(round(xR_sub), 0, W - 1))
    cv2.line(vis, (xL_i, 0), (xL_i, H - 1), (0, 255, 0), 2)
    cv2.line(vis, (xR_i, 0), (xR_i, H - 1), (0, 255, 0), 2)

    # 本体边界（蓝）
    cv2.line(vis, (xBodyL, 0), (xBodyL, H - 1), (255, 0, 0), 2)
    cv2.line(vis, (xBodyR, 0), (xBodyR, H - 1), (255, 0, 0), 2)

    txt = (
        f"len={length_px:.2f}px ({used})" if length_mm is None else f"len={length_px:.2f}px={length_mm:.3f}mm ({used})"
    )
    cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    dbg = dict(
        mask=mask,
        band_y=(y1, y2),
        xBodyL=xBodyL,
        xBodyR=xBodyR,
        thr=thr,
        used=used,
        xL_step=xL_step,
        xR_step=xR_step,
        xL_sub=xL_sub,
        xR_sub=xR_sub,
        snap_r=snap_r,
        e_s=e_s,
    )

    # ✅ 返回亚像素端点（float）
    return length_px, length_mm, xL_sub, xR_sub, vis, dbg


# ==========================
# 主流程：只处理 UNDERCRIMP（class2 >= 0.6）
# ==========================
if __name__ == "__main__":
    weights_path = r"D:\code\yolo_rephoto\v8n_blam_best.pt"
    input_path = r"D:\code\yolo_rephoto\measure_pics"
    save_dir = r"D:\code\yolo_rephoto\measure_code\measure_out_under_only"
    os.makedirs(save_dir, exist_ok=True)

    mm_per_px = 0.07329

    CLASS_UNDER = 2  # 压接不足
    CLASS_CRIMP = 0  # 压接区域（ROI）
    UNDER_THR = 0.6

    model = YOLO(weights_path)

    img_list = collect_images(input_path)
    if not img_list:
        print("[ERROR] No images found:", input_path)
        raise SystemExit(0)

    print("Keys: A/Left prev | D/Right next | S save | Q/Esc quit")
    print(f"Only process UNDERCRIMP (class={CLASS_UNDER}) with conf >= {UNDER_THR}")

    idx = 0
    cache = {}

    while True:
        idx = max(0, min(idx, len(img_list) - 1))
        img_path = img_list[idx]
        base = os.path.splitext(os.path.basename(img_path))[0]

        if idx not in cache:
            img = cv2.imread(img_path)
            if img is None:
                cache[idx] = {"ok": False, "reason": "read_fail", "img": None}
            else:
                H, W = img.shape[:2]
                r = model(img, conf=0.25, verbose=False)[0]

                # 1) 先拿 class2（压接不足）最高置信度
                under_xyxy, under_conf = get_best_box_by_class(r, CLASS_UNDER)

                # 只处理 under_conf >= 0.6 的图
                if under_conf < UNDER_THR:
                    cache[idx] = {"ok": False, "reason": f"skip_under_conf={under_conf:.2f}", "img": img}
                else:
                    # 2) under 达标后，必须拿 class0 框做长度测量
                    crimp_xyxy, crimp_conf = get_best_box_by_class(r, CLASS_CRIMP)
                    if crimp_xyxy is None:
                        cache[idx] = {
                            "ok": False,
                            "reason": "under_ok_but_no_class0",
                            "img": img,
                            "under_conf": under_conf,
                        }
                    else:
                        pad = int(0.02 * max(W, H))
                        cb = clamp_box(*crimp_xyxy, W=W, H=H, pad=pad)
                        if cb is None:
                            cache[idx] = {"ok": False, "reason": "bad_class0_box", "img": img, "under_conf": under_conf}
                        else:
                            x1, y1, x2, y2 = cb
                            roi = img[y1:y2, x1:x2].copy()

                            # PCA 旋转
                            roi_norm, angle = normalize_pose_pca(roi, pad_ratio=0.3)

                            # 过长裁剪
                            roi_norm, crop_off_x = center_crop_to_width(roi_norm, target_w=580)

                            # 跑压接不足测量（亚像素版本）
                            alg_name = "UNDERCRIMP_MEAS_SUBPIX"
                            len_px, len_mm, xL_sub, xR_sub, meas_vis, dbg = measure_undercrimp_terminal(
                                roi_norm, mm_per_px=mm_per_px
                            )

                            vis = img.copy()
                            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(
                                vis,
                                f"UNDER c2={under_conf:.2f} | ROI=class0 c0={crimp_conf:.2f} | rot={angle:.2f}",
                                (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0),
                                2,
                            )

                            cache[idx] = dict(
                                ok=True,
                                img=img,
                                vis=vis,
                                roi=roi,
                                roi_norm=roi_norm,
                                angle=angle,
                                under_conf=under_conf,
                                crimp_conf=crimp_conf,
                                alg=alg_name,
                                len_px=len_px,
                                len_mm=len_mm,
                                xL_sub=xL_sub,
                                xR_sub=xR_sub,
                                meas_vis=meas_vis,
                                dbg=dbg,
                            )

        data = cache[idx]

        # === 显示 ===
        if not data.get("ok", False):
            show_resized("Original Image + ROI", data.get("img", np.zeros((480, 640, 3), np.uint8)))
            blank = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(
                blank, f"{data.get('reason', 'skip')}", (30, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
            )
            show_resized("Pose Normalized ROI (PCA)", blank)
            show_resized("Measurement (UNDERCRIMP)", blank)
            print(
                f"\r[{idx + 1}/{len(img_list)}] {os.path.basename(img_path)}  {data.get('reason', 'skip')}        ",
                end="",
            )
        else:
            show_resized("Original Image + ROI", data["vis"])
            show_resized("Pose Normalized ROI (PCA)", data["roi_norm"])
            show_resized("Measurement (UNDERCRIMP)", data["meas_vis"])
            msg = "len=FAIL" if data["len_px"] is None else f"len={data['len_px']:.2f}px"
            print(
                f"\r[{idx + 1}/{len(img_list)}] {os.path.basename(img_path)}  under_conf2={data['under_conf']:.2f}  {msg}    ",
                end="",
            )

        # === 键盘 ===
        key = cv2.waitKey(0) & 0xFF

        # === 查看能量曲线：E ===
        if key in (ord("e"), ord("E")) and data.get("ok", False):
            dbg = data.get("dbg", {})
            e_s = dbg.get("e_s", None)
            if e_s is not None:
                curve_img = plot_energy_curve_cv(
                    e_s,
                    xBodyL=dbg.get("xBodyL"),
                    xBodyR=dbg.get("xBodyR"),
                    xL_sub=dbg.get("xL_sub"),
                    xR_sub=dbg.get("xR_sub"),
                    xL_step=dbg.get("xL_step"),
                    xR_step=dbg.get("xR_step"),
                    h=240,
                )
                show_resized("Energy Curve (UNDERCRIMP)", curve_img, max_size=1300)
            else:
                print("\n[WARN] dbg.e_s not found")
            continue

        if key in [27, ord("q"), ord("Q")]:
            break
        if key in [ord("a"), 81]:  # A / ←
            idx -= 1
            continue
        if key in [ord("d"), 83]:  # D / →
            idx += 1
            continue

        # === 保存：只保存 ok 的 ===
        if key in [ord("s"), ord("S")] and data.get("ok", False):
            cv2.imwrite(os.path.join(save_dir, f"{base}_vis.png"), data["vis"])
            cv2.imwrite(os.path.join(save_dir, f"{base}_roi.png"), data["roi"])
            cv2.imwrite(os.path.join(save_dir, f"{base}_roi_norm.png"), data["roi_norm"])
            cv2.imwrite(os.path.join(save_dir, f"{base}_meas.png"), data["meas_vis"])

            with open(os.path.join(save_dir, f"{base}_result.txt"), "w", encoding="utf-8") as f:
                f.write(f"img={img_path}\n")
                f.write(f"under_conf2={data['under_conf']}\n")
                f.write(f"class0_conf={data['crimp_conf']}\n")
                f.write(f"alg={data['alg']}\n")
                f.write(f"rot_angle_deg={data['angle']}\n")
                f.write(f"length_px={data['len_px']}\n")
                f.write(f"length_mm={data['len_mm']}\n")
                f.write(f"xL_subpx={data.get('xL_sub')}\n")
                f.write(f"xR_subpx={data.get('xR_sub')}\n")

            print(f"\n[SAVED] {base} -> {save_dir}")

    cv2.destroyAllWindows()
    print("\n[EXIT]")
