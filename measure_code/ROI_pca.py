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


def normalize_pose_pca(roi_bgr, pad_ratio=0.3):
    """ROI + PCA 姿态归一化（安全版，不截断） pad_ratio: 扩边比例（0.2~0.4 推荐）.
    """
    h, w = roi_bgr.shape[:2]

    # ========= 1️⃣ 扩边 padding =========
    pad_h = int(h * pad_ratio)
    pad_w = int(w * pad_ratio)

    roi_pad = cv2.copyMakeBorder(roi_bgr, pad_h, pad_h, pad_w, pad_w, borderType=cv2.BORDER_REPLICATE)

    # ========= 2️⃣ PCA 方向估计 =========
    gray = cv2.cvtColor(roi_pad, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 80, 180)

    ys, xs = np.where(edges > 0)
    if len(xs) < 200:
        return roi_bgr, 0.0

    pts = np.column_stack((xs, ys)).astype(np.float32)
    _mean, eigenvectors = cv2.PCACompute(pts, mean=None)
    vx, vy = eigenvectors[0]

    angle = math.degrees(math.atan2(vy, vx))

    # ========= 3️⃣ 旋转 =========
    H, W = roi_pad.shape[:2]
    center = (W // 2, H // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(roi_pad, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # ========= 4️⃣ 裁回中心区域 =========
    y1 = pad_h
    y2 = pad_h + h
    x1 = pad_w
    x2 = pad_w + w

    roi_norm = rotated[y1:y2, x1:x2]

    return roi_norm, angle


# ==========================
# ✅ 最小新增：收集图片（单张/文件夹/通配符）
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
# ✅ 最小新增：处理单张（完全沿用你原 boxes[0] 逻辑）
# ==========================
def process_one(model, img_path, conf=0.5, pad_ratio=0.3):
    img = cv2.imread(img_path)
    if img is None:
        return None

    results = model(img, conf=conf, verbose=False)[0]

    if len(results.boxes) == 0:
        vis = img.copy()
        cv2.putText(vis, "No crimp_region detected.", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return {"vis": vis, "roi": None, "roi_norm": None, "angle": None, "bbox": None}

    # ✅ 完全保留你原本：boxes[0]
    box = results.boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    roi = img[y1:y2, x1:x2].copy()
    roi_norm, angle = normalize_pose_pca(roi, pad_ratio=pad_ratio)

    vis = img.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return {"vis": vis, "roi": roi, "roi_norm": roi_norm, "angle": angle, "bbox": (x1, y1, x2, y2)}


# ==========================
# 主流程
# ==========================
if __name__ == "__main__":
    # 1️⃣ 加载模型
    model = YOLO(r"D:\code\yolo_rephoto\v8n_blam_best.pt")  # 或 yolov8_blam.yaml + load 权重

    # 2️⃣ 读取原图（只读一次！）
    input_path = r"D:\code\yolo_rephoto\measure_pics"
    img_list = collect_images(input_path)

    if img_list is None:
        print("[ERROR] Image not found.")
        exit(0)

    # 3️⃣ 保存目录（按 S 保存）
    save_dir = r"D:\code\yolo_rephoto\measure_code\browse_save"
    os.makedirs(save_dir, exist_ok=True)

    idx = 0
    cache = {}
    print("Keys: A/Left prev | D/Right next | S save | Q/Esc quit")

    while True:
        idx = max(0, min(idx, len(img_list) - 1))
        img_path = img_list[idx]
        base = os.path.splitext(os.path.basename(img_path))[0]

        if idx not in cache:
            cache[idx] = process_one(model, img_path, conf=0.5, pad_ratio=0.3)
        data = cache[idx]

        if data is None:
            print("[WARN] Read fail:", img_path)
            idx += 1
            continue

        # 显示三窗口
        show_resized("Original Image + ROI", data["vis"])
        if data["roi"] is not None:
            show_resized("Cropped ROI", data["roi"])
            show_resized("Pose Normalized ROI (PCA)", data["roi_norm"])
        else:
            blank = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(blank, "NO ROI", (200, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            show_resized("Cropped ROI", blank)
            show_resized("Pose Normalized ROI (PCA)", blank)

        # 控制台提示
        print(f"\r[{idx + 1}/{len(img_list)}] {os.path.basename(img_path)}  ", end="")

        key = cv2.waitKey(0) & 0xFF

        # 退出
        if key in [27, ord("q"), ord("Q")]:
            break

        # 上一张
        if key in [ord("a"), 81]:  # A or ←
            idx -= 1
            continue

        # 下一张
        if key in [ord("d"), 83]:  # D or →
            idx += 1
            continue

        # 保存当前结果
        if key in [ord("s"), ord("S")]:
            cv2.imwrite(os.path.join(save_dir, f"{base}_vis.png"), data["vis"])
            if data["roi"] is not None:
                cv2.imwrite(os.path.join(save_dir, f"{base}_roi.png"), data["roi"])
                cv2.imwrite(os.path.join(save_dir, f"{base}_roi_norm.png"), data["roi_norm"])
                with open(os.path.join(save_dir, f"{base}_angle.txt"), "w", encoding="utf-8") as f:
                    f.write(f"{data['angle']}\n")
            print(f"\n[SAVED] {base} -> {save_dir}")
            continue

    # # 3️⃣ YOLO 检测（假设只有一个 crimp_region）
    # results = model(img, conf=0.5, verbose=False)[0]

    # if len(results.boxes) == 0:
    #     print("[WARN] No crimp_region detected.")
    #     exit(0)

    # box = results.boxes[0]
    # x1, y1, x2, y2 = map(int, box.xyxy[0])

    # # 4️⃣ 裁剪 ROI
    # roi = img[y1:y2, x1:x2].copy()

    # # 5️⃣ PCA 姿态归一化（对 ROI）
    # roi_norm, angle = normalize_pose_pca(roi)

    # # 6️⃣ 可视化
    # vis = img.copy()
    # cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # show_resized("Original Image + ROI", vis)
    # show_resized("Cropped ROI", roi)
    # show_resized("Pose Normalized ROI (PCA)", roi_norm)

    # print(f"[INFO] PCA rotation angle = {angle:.2f} deg")
    # print("Press Q / ESC to exit.")

    # while True:
    #     key = cv2.waitKey(30) & 0xFF
    #     if key in [27, ord('q')]:
    #         break

    cv2.destroyAllWindows()
    print("\n[EXIT]")
