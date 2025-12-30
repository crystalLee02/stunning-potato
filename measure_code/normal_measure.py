import os
import glob
import cv2
import numpy as np
import math
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
# PCA 姿态归一化
# ==========================
def normalize_pose_pca(roi_bgr, pad_ratio=0.3):
    h, w = roi_bgr.shape[:2]
    pad_h = int(h * pad_ratio)
    pad_w = int(w * pad_ratio)

    roi_pad = cv2.copyMakeBorder(
        roi_bgr, pad_h, pad_h, pad_w, pad_w,
        borderType=cv2.BORDER_REPLICATE
    )

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
        roi_pad, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
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
    """返回 (xyxy_int, conf_float)；如果不存在该类返回 (None, 0.0)"""
    if r.boxes is None or len(r.boxes) == 0:
        return None, 0.0

    xyxy = r.boxes.xyxy.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    conf = r.boxes.conf.cpu().numpy().astype(float)

    m = (cls == cls_id)
    if m.sum() == 0:
        return None, 0.0

    idxs = np.where(m)[0]
    best_i = idxs[int(np.argmax(conf[idxs]))]
    x1, y1, x2, y2 = xyxy[best_i]
    return (int(x1), int(y1), int(x2), int(y2)), float(conf[best_i])


def clamp_box(x1, y1, x2, y2, W, H, pad=0):
    x1 = max(0, min(W - 1, x1 - pad))
    y1 = max(0, min(H - 1, y1 - pad))
    x2 = max(0, min(W,     x2 + pad))
    y2 = max(0, min(H,     y2 + pad))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None
    return x1, y1, x2, y2


# ==========================
# ✅ 过长裁剪：左右等比例裁剪到固定宽度
# ==========================
def center_crop_to_width(img, target_w=550):
    """
    若图像宽度 > target_w，则左右等量裁剪到 target_w
    返回: (cropped_img, x_offset, cropped_flag)
    x_offset 表示裁剪后图像在原图中的起始x（用于回映坐标可用）
    """
    H, W = img.shape[:2]
    if W <= target_w:
        return img, 0, False
    x0 = (W - target_w) // 2
    x1 = x0 + target_w
    cropped = img[:, x0:x1].copy()
    return cropped, x0, True


# ==========================
# 1D 工具：平滑/能量/窗口峰值
# ==========================
def smooth_1d(x, sigma=2.5):
    if sigma <= 0:
        return x
    k = int(max(3, sigma * 6)) | 1
    g = cv2.GaussianBlur(x.reshape(-1, 1).astype(np.float32), (k, 1), sigmaX=sigma, sigmaY=0)
    return g.reshape(-1)


def energy_profile_x(gray_enh, y1, y2):
    g = cv2.GaussianBlur(gray_enh, (3, 3), 0)
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    e = np.abs(sx)[y1:y2, :].sum(axis=0)
    e_s = smooth_1d(e, sigma=2.5)
    return e_s


def snap_peak_in_window(e_s, x_lo, x_hi):
    n = len(e_s)
    x_lo = int(np.clip(x_lo, 0, n - 1))
    x_hi = int(np.clip(x_hi, 0, n - 1))
    if x_hi <= x_lo:
        return x_lo
    j = int(np.argmax(e_s[x_lo:x_hi + 1]))
    return x_lo + j


# ==========================
# body mask：正常端子用 edges mask
# ==========================
def build_body_mask_from_edges(gray):
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    med = np.median(g)
    lo = int(max(0, 0.66 * med))
    hi = int(min(255, 1.33 * med))
    edges = cv2.Canny(g, lo, hi)

    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)), iterations=2)

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
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21)), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=1)
    return mask


def body_extent_in_band(mask, y1, y2, occ_thr_ratio=0.16):
    band = mask[y1:y2, :]
    col_occ = (band > 0).sum(axis=0).astype(np.float32)
    thr = occ_thr_ratio * float(y2 - y1)

    idxs = np.where(col_occ >= thr)[0]
    if idxs.size < 10:
        idxs = np.where(col_occ > 0)[0]
    if idxs.size < 10:
        return None, None
    return int(idxs[0]), int(idxs[-1])


# ============================================================
# ✅ 正常端子测量（定义：class2_conf < UNDER_THR）
# ============================================================
def measure_normal_terminal(
    roi_norm_bgr, mm_per_px=0.0,
    band_top=0.35, band_bottom=0.65,
    occ_thr_ratio=0.16,
    end_win_ratio=0.25
):
    gray = cv2.cvtColor(roi_norm_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enh = clahe.apply(gray)

    H, W = gray.shape[:2]
    y1 = int(band_top * H)
    y2 = int(band_bottom * H)
    y1 = max(0, min(y1, H - 2))
    y2 = max(y1 + 1, min(y2, H))

    mask = build_body_mask_from_edges(gray_enh)
    if mask is None:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return None, None, None, None, vis, {"reason": "no_mask"}

    xBodyL, xBodyR = body_extent_in_band(mask, y1, y2, occ_thr_ratio=occ_thr_ratio)
    if xBodyL is None:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return None, None, None, None, vis, {"reason": "no_extent", "mask": mask}

    e_s = energy_profile_x(gray_enh, y1, y2)

    body_w = max(1, xBodyR - xBodyL)
    win = int(end_win_ratio * body_w)
    win = max(20, win)

    xL = snap_peak_in_window(e_s, xBodyL, min(xBodyL + win, xBodyR))
    xR = snap_peak_in_window(e_s, max(xBodyR - win, xBodyL), xBodyR)

    xL = int(np.clip(xL, xBodyL, xBodyR))
    xR = int(np.clip(xR, xBodyL, xBodyR))
    if xR <= xL:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return None, None, None, None, vis, {"reason": "xR<=xL", "mask": mask}

    length_px = float(xR - xL)
    length_mm = length_px * float(mm_per_px) if (mm_per_px and mm_per_px > 0) else None

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(vis, (0, y1), (W - 1, y2), (255, 255, 0), 1)
    cv2.line(vis, (xL, 0), (xL, H - 1), (0, 255, 0), 2)
    cv2.line(vis, (xR, 0), (xR, H - 1), (0, 255, 0), 2)
    txt = f"len={length_px:.1f}px" if length_mm is None else f"len={length_px:.1f}px={length_mm:.2f}mm"
    cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    dbg = {"mask": mask, "band_y": (y1, y2), "xBodyL": xBodyL, "xBodyR": xBodyR}
    return length_px, length_mm, xL, xR, vis, dbg


# ==========================
# 主流程：只处理 under_conf < 0.6 的图片（你定义的“正常端子”）
# ==========================
if __name__ == "__main__":
    weights_path = r"D:\code\yolo_rephoto\v8n_blam_best.pt"
    input_path   = r"D:\code\yolo_rephoto\measure_pics"
    save_dir     = r"D:\code\yolo_rephoto\measure_code\measure_out_normal_by_under"
    os.makedirs(save_dir, exist_ok=True)

    mm_per_px = 0.07329

    CLASS_CRIMP = 0   # 压接区域框（用于测长度）
    CLASS_UNDER = 2   # 压接不足缺陷（用于判定“是否正常”）
    UNDER_THR = 0.6   # <0.6 => 正常端子

    # ✅ 裁剪参数：你要求的“超过就裁剪到固定宽度”
    CROP_TARGET_W = 450

    model = YOLO(weights_path)
    img_list = collect_images(input_path)
    if not img_list:
        print("[ERROR] No images found:", input_path)
        exit(0)

    print("Keys: A/Left prev | D/Right next | S save | Q/Esc quit")
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

                # 先取 class2 best conf：定义“是否正常”
                _, under_conf = get_best_box_by_class(r, CLASS_UNDER)

                if under_conf >= UNDER_THR:
                    # 压接不足：本脚本不测量
                    vis = img.copy()
                    cv2.putText(vis, f"UNDERCRIMP_SKIP: class2_conf={under_conf:.2f} >= {UNDER_THR}",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cache[idx] = {"ok": True, "skip": True, "img": img, "vis": vis, "under_conf": under_conf}
                else:
                    # 正常端子：取 class0 框测长度
                    crimp_xyxy, crimp_conf = get_best_box_by_class(r, CLASS_CRIMP)
                    if crimp_xyxy is None:
                        cache[idx] = {"ok": False, "reason": "no_class0", "img": img, "under_conf": under_conf}
                    else:
                        pad = int(0.02 * max(W, H))
                        cb = clamp_box(*crimp_xyxy, W=W, H=H, pad=pad)
                        if cb is None:
                            cache[idx] = {"ok": False, "reason": "bad_box", "img": img, "under_conf": under_conf}
                        else:
                            x1, y1, x2, y2 = cb
                            roi = img[y1:y2, x1:x2].copy()
                            roi_norm, angle = normalize_pose_pca(roi, pad_ratio=0.3)

                            # ✅ 新增：过长裁剪（左右等比例裁剪到固定宽度）
                            roi_norm_raw_w = roi_norm.shape[1]
                            roi_norm, crop_off_x, cropped = center_crop_to_width(roi_norm, target_w=CROP_TARGET_W)

                            len_px, len_mm, xL, xR, meas_vis, dbg = measure_normal_terminal(
                                roi_norm, mm_per_px=mm_per_px
                            )

                            vis = img.copy()
                            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            crop_tag = f"cropW={roi_norm_raw_w}->{roi_norm.shape[1]} off={crop_off_x}" if cropped else "cropW=NO"
                            cv2.putText(vis,
                                        f"NORMAL: class2_conf={under_conf:.2f} < {UNDER_THR} | class0_conf={crimp_conf:.2f} | rot={angle:.2f} | {crop_tag}",
                                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                            cache[idx] = dict(
                                ok=True, skip=False,
                                img=img, vis=vis,
                                roi=roi, roi_norm=roi_norm, angle=angle,
                                under_conf=under_conf, conf0=crimp_conf,
                                cropped=cropped, crop_off_x=crop_off_x,
                                roi_norm_raw_w=roi_norm_raw_w,
                                len_px=len_px, len_mm=len_mm,
                                meas_vis=meas_vis, dbg=dbg
                            )

        data = cache[idx]

        if not data.get("ok", False):
            show_resized("Original Image + ROI", data.get("img", np.zeros((480, 640, 3), np.uint8)))
            blank = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(blank, f"FAIL: {data.get('reason','unknown')}", (40, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            show_resized("Pose Normalized ROI (PCA)", blank)
            show_resized("Measurement (NORMAL)", blank)

            print(f"\r[{idx+1}/{len(img_list)}] {os.path.basename(img_path)}  FAIL={data.get('reason')}    ", end="")
        else:
            show_resized("Original Image + ROI", data["vis"])
            if data.get("skip", False):
                blank = np.zeros((480, 640, 3), np.uint8)
                cv2.putText(blank, f"SKIPPED (under_conf={data['under_conf']:.2f})", (40, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                show_resized("Pose Normalized ROI (PCA)", blank)
                show_resized("Measurement (NORMAL)", blank)
                print(f"\r[{idx+1}/{len(img_list)}] {os.path.basename(img_path)}  SKIP under_conf2={data['under_conf']:.2f}  ", end="")
            else:
                show_resized("Pose Normalized ROI (PCA)", data["roi_norm"])
                show_resized("Measurement (NORMAL)", data["meas_vis"])
                msg = "len=FAIL" if data["len_px"] is None else f"len={data['len_px']:.1f}px"
                crop_msg = f"crop={data['roi_norm_raw_w']}->{data['roi_norm'].shape[1]}" if data.get("cropped") else "crop=NO"
                print(f"\r[{idx+1}/{len(img_list)}] {os.path.basename(img_path)}  under_conf2={data['under_conf']:.2f}  {crop_msg}  {msg}    ", end="")

        key = cv2.waitKey(0) & 0xFF
        if key in [27, ord('q'), ord('Q')]:
            break
        if key in [ord('a'), 81]:
            idx -= 1
            continue
        if key in [ord('d'), 83]:
            idx += 1
            continue

        if key in [ord('s'), ord('S')] and data.get("ok", False):
            if data.get("skip", False):
                cv2.imwrite(os.path.join(save_dir, f"{base}_skip_vis.png"), data["vis"])
                with open(os.path.join(save_dir, f"{base}_skip.txt"), "w", encoding="utf-8") as f:
                    f.write(f"img={img_path}\n")
                    f.write(f"class2_conf={data['under_conf']}\n")
                    f.write("SKIPPED because under_conf >= UNDER_THR\n")
                print(f"\n[SAVED] SKIP {base} -> {save_dir}")
            else:
                cv2.imwrite(os.path.join(save_dir, f"{base}_vis.png"), data["vis"])
                cv2.imwrite(os.path.join(save_dir, f"{base}_roi.png"), data["roi"])
                cv2.imwrite(os.path.join(save_dir, f"{base}_roi_norm.png"), data["roi_norm"])
                cv2.imwrite(os.path.join(save_dir, f"{base}_meas.png"), data["meas_vis"])
                if isinstance(data["dbg"], dict) and "mask" in data["dbg"] and data["dbg"]["mask"] is not None:
                    cv2.imwrite(os.path.join(save_dir, f"{base}_bodymask.png"), data["dbg"]["mask"])
                with open(os.path.join(save_dir, f"{base}_result.txt"), "w", encoding="utf-8") as f:
                    f.write(f"img={img_path}\n")
                    f.write(f"class2_conf={data['under_conf']}\n")
                    f.write(f"class0_conf={data['conf0']}\n")
                    f.write(f"rot_angle_deg={data['angle']}\n")
                    f.write(f"roi_norm_raw_w={data['roi_norm_raw_w']}\n")
                    f.write(f"roi_norm_final_w={data['roi_norm'].shape[1]}\n")
                    f.write(f"cropped={data['cropped']}\n")
                    f.write(f"crop_off_x={data['crop_off_x']}\n")
                    f.write(f"length_px={data['len_px']}\n")
                    f.write(f"length_mm={data['len_mm']}\n")
                print(f"\n[SAVED] {base} -> {save_dir}")

    cv2.destroyAllWindows()
    print("\n[EXIT]")
