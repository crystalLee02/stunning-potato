from ultralytics import YOLO


def main():
    print(">>> Starting YOLOv8n dualhead training script")

    model = YOLO("yolov8_dualhead.yaml")

    model.train(
        data=r"D:\code\yolo_rephoto\dataset_rephoto\data.yaml",
        imgsz=1024,
        epochs=20,
        batch=8,
        device=0,
        patience=0,
        name="yolov8n_dualhead_1225",
    )

    print(">>> Training finished")


if __name__ == "__main__":
    main()
