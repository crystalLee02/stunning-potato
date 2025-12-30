import os, glob

label_dir = r"D:\code\yolo_rephoto\dataset_rephoto\labels\val"
bad = []
cnt = [0,0,0]
for fp in glob.glob(os.path.join(label_dir, "*.txt")):
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) != 5: 
                bad.append((fp, "len!=5", line)); 
                continue
            c = int(float(p[0]))
            x,y,w,h = map(float, p[1:])
            if c<0 or c>2:
                bad.append((fp, "cls_out_of_range", line))
                continue
            cnt[c] += 1
            # YOLO 格式必须归一化且 >0
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                bad.append((fp, "bbox_invalid", line))

print("class counts:", cnt)
print("bad samples:", len(bad))
for i in range(min(20, len(bad))):
    print(bad[i])
