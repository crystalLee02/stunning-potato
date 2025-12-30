import csv
import glob
import os

import cv2
import numpy as np

from ultralytics import YOLO


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


def show_resized(winname, img, max_size=1000):
    h, w = img.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale)
    cv2.imshow(winname, img)


def draw_label(img, text, org=(20, 40), color=(0, 255, 0)):
    out = img.copy()
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return out


if __name__ == "__main__":
    # ====== 你只改这里 ======
    weights_path = r"D:\code\yolo_rephoto\v8n_blam_best.pt"
    input_path = r"D:\code\yolo_rephoto\measure_pics"  # 单张/文件夹/通配符
    save_dir = r"D:\code\yolo_rephoto\infer_out"
    conf_thres = 0.25  # 推理阈值（想看更多框就调低）
    # ========================

    os.makedirs(save_dir, exist_ok=True)

    model = YOLO(weights_path)
    names = model.names  # {id: name}

    img_list = collect_images(input_path)
    if not img_list:
        print("[ERROR] No images found:", input_path)
        raise SystemExit(0)

    results_csv = os.path.join(save_dir, "results.csv")
    rows = [["index", "filename", "cls_id", "cls_name", "conf", "x1", "y1", "x2", "y2"]]

    print("Keys: A/Left prev | D/Right next | S save current | Q/Esc quit")
    idx = 0
    cache = {}

    while True:
        idx = max(0, min(idx, len(img_list) - 1))
        img_path = img_list[idx]
        base = os.path.splitext(os.path.basename(img_path))[0]

        if idx not in cache:
            img = cv2.imread(img_path)
            if img is None:
                cache[idx] = dict(ok=False, reason="read_fail", img=None)
            else:
                r = model(img, conf=conf_thres, verbose=False)[0]

                TARGET_CLS = 2  # 压接不足

            if len(r.boxes) == 0:
                cache[idx] = dict(ok=True, img=img, has_box=False, cls_id=None, conf=None, box=None)
            else:
                # ===== 过滤出 cls==2 的框 =====
                cls_all = r.boxes.cls.cpu().numpy().astype(int)
                conf_all = r.boxes.conf.cpu().numpy()
                xyxy_all = r.boxes.xyxy.cpu().numpy()

                keep = np.where(cls_all == TARGET_CLS)[0]

                if keep.size == 0:
                    # 没有压接不足框
                    cache[idx] = dict(ok=True, img=img, has_box=False, cls_id=None, conf=None, box=None)
                else:
                    # 取 conf 最高的那个 cls=2 框
                    best_i = keep[np.argmax(conf_all[keep])]
                    x1, y1, x2, y2 = map(int, xyxy_all[best_i])
                    cls_id = int(cls_all[best_i])
                    conf = float(conf_all[best_i])

                    cache[idx] = dict(ok=True, img=img, has_box=True, cls_id=cls_id, conf=conf, box=(x1, y1, x2, y2))

        data = cache[idx]

        if not data.get("ok", False):
            blank = 255 * (0 * cv2.imread(img_list[0]) if False else None)
            print(f"\n[{idx + 1}/{len(img_list)}] {os.path.basename(img_path)}  FAIL: {data.get('reason')}")
            break

        img = data["img"].copy()

        if not data["has_box"]:
            text = f"[{idx + 1}/{len(img_list)}] {base}  NO BOX"
            vis = draw_label(img, text, color=(0, 0, 255))
            print(f"\r[{idx + 1}/{len(img_list)}] {os.path.basename(img_path)}  NO BOX                    ", end="")
            # 记录 csv
            rows.append([idx, os.path.basename(img_path), "", "", "", "", "", "", ""])
        else:
            x1, y1, x2, y2 = data["box"]
            cls_id = data["cls_id"]
            conf = data["conf"]
            cls_name = names.get(cls_id, str(cls_id))

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"[{idx + 1}/{len(img_list)}] {base}  cls={cls_id}({cls_name})  conf={conf:.2f}"
            vis = draw_label(img, text, color=(0, 255, 0))

            print(
                f"\r[{idx + 1}/{len(img_list)}] {os.path.basename(img_path)}  cls={cls_id}({cls_name}) conf={conf:.2f}        ",
                end="",
            )

            # 记录 csv
            rows.append([idx, os.path.basename(img_path), cls_id, cls_name, f"{conf:.4f}", x1, y1, x2, y2])

        show_resized("Infer Viewer", vis, max_size=1100)

        key = cv2.waitKey(0) & 0xFF
        if key in [27, ord("q"), ord("Q")]:
            break
        if key in [ord("a"), 81]:  # A or Left
            idx -= 1
            continue
        if key in [ord("d"), 83]:  # D or Right
            idx += 1
            continue

        if key in [ord("s"), ord("S")]:
            # 保存当前可视化和 txt
            out_img = os.path.join(save_dir, f"{base}_infer.png")
            cv2.imwrite(out_img, vis)

            out_txt = os.path.join(save_dir, f"{base}_infer.txt")
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(f"img={img_path}\n")
                if not data["has_box"]:
                    f.write("no_box=1\n")
                else:
                    cls_id = data["cls_id"]
                    conf = data["conf"]
                    cls_name = names.get(cls_id, str(cls_id))
                    x1, y1, x2, y2 = data["box"]
                    f.write(f"cls_id={cls_id}\n")
                    f.write(f"cls_name={cls_name}\n")
                    f.write(f"conf={conf}\n")
                    f.write(f"xyxy={x1},{y1},{x2},{y2}\n")

            print(f"\n[SAVED] {base} -> {save_dir}")

    cv2.destroyAllWindows()

    # 写出汇总 CSV（去掉表头重复：上面 rows 每次都 append，所以会多很多行；这里简单去重）
    # 你也可以直接只保留最后一次 rows，本脚本为了“随时退出也能有记录”，就写入全部。
    # 我这里做一次压缩：按 filename 保留最后一条记录。
    last_by_name = {}
    for r in rows[1:]:
        last_by_name[r[1]] = r
    final_rows = [rows[0]] + [last_by_name[k] for k in sorted(last_by_name.keys())]

    with open(results_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerows(final_rows)

    print(f"\n[OK] Saved summary: {results_csv}")
