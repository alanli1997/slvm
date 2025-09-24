from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("runs/detect/DT/weights/best.pt")
    results = model.val(
        data="datasets/DT/DT.yaml",
        # model='ultralytics/models/v8/seg/yolov8n-seg.yaml',
        device='0',
        imgsz=640,
        name='val',
        batch=1,
        #amp=False,
        half=False,
        augment=False,
        save_json=True
    )

