import os, glob

val_label_dir = r"D:\code\yolo_rephoto\dataset_rephoto\labels\val"
has_def = []
for fp in glob.glob(os.path.join(val_label_dir, "*.txt")):
    with open(fp, "r", encoding="utf-8") as f:
        cls = [int(float(l.split()[0])) for l in f if l.strip()]
    if any(c in (1,2) for c in cls):
        has_def.append(fp)

print("val defect files:", len(has_def))
for p in has_def[:20]:
    print(os.path.basename(p))
