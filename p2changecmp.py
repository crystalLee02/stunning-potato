# p2changecmp.py
import multiprocessing as mp
from ultralytics import YOLO

def print_per_class_ap(metrics, title=""):
    # Ultralytics 的 metrics.names: {0:'xx', 1:'yy', ...}
    names = metrics.names if hasattr(metrics, "names") else {}
    box = metrics.box

    # mAP50-95 per class（最常用来写论文按类AP）
    ap5095 = getattr(box, "maps", None)  # list[nc]

    print("\n" + "=" * 60)
    print(title)
    print(f"Overall mAP50-95: {box.map:.4f} | mAP50: {box.map50:.4f}")
    print("-" * 60)

    if ap5095 is None:
        print("当前 Ultralytics 版本没找到 box.maps（按类 AP50-95），请把 metrics 对象 repr 发我。")
        return

    for i, ap in enumerate(ap5095):
        cname = names.get(i, str(i))
        print(f"{i:2d} {cname:20s}  AP50-95: {ap:.4f}")

def main():
    data = r"D:\code\yolo_rephoto\dataset_rephoto\data.yaml"

    # 改成你自己的权重路径
    w_baseline = r"D:\code\yolo_rephoto\baseline_best.pt"
    w_p2       = r"D:\code\yolo_rephoto\smallobj_best.pt"

    m0 = YOLO(w_baseline)
    r0 = m0.val(data=data, split="test", imgsz=1024, conf=0.001, workers=0, device=0)
    print_per_class_ap(r0, title="BASELINE")

    m1 = YOLO(w_p2)
    r1 = m1.val(data=data, split="test", imgsz=1024, conf=0.001, workers=0, device=0)
    print_per_class_ap(r1, title="P2 SMALL-OBJECT")

    # 额外：打印每类提升 ΔAP
    ap0 = r0.box.maps
    ap1 = r1.box.maps
    names = r0.names
    print("\n" + "=" * 60)
    print("ΔAP50-95 (P2 - Baseline)")
    print("-" * 60)
    for i in range(len(ap0)):
        cname = names.get(i, str(i))
        print(f"{i:2d} {cname:20s}  {ap0[i]:.4f} -> {ap1[i]:.4f}  Δ{(ap1[i]-ap0[i]):+.4f}")

if __name__ == "__main__":
    mp.freeze_support()   # Windows 必加（尤其你以后可能打包exe时）
    main()
