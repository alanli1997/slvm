from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO("runs/segment/DISDt27/weights/best.pt")
    results = model.predict(
        data="datasets/DISDt/DISDt.yaml",
        # model='ultralytics/models/v8/seg/yolov8n-seg.yaml',
        device='0',
        imgsz=640,
        source='inference/000016_10.png',
        #amp=False,
        half=False,
        verbose=True,
        save=True,
        visualize=False,
        line_thickness=3,
        iou=0.3
        # line_width=3,
        # font_size=None


    )

