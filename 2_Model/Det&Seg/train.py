from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("ultralytics/models/darkseg/sniv8n-seg-f.yaml")
    results = model.train(
        data="datasets/DT/DT.yaml",
        name='DT',
        # datasets/DTSeg-test/DTSeg-test.yaml
        device='0',
        epochs=200,
        imgsz=640,
        batch=16,
        workers=16,
        rect=False,
        #amp=False,
        optimizer='SGD',
        cache=True,
        resume=False,
        half=False,
        deterministic=False,
        vid_stride=1,
        v5loader=False,
        augment=False,
        # save_hybrid=True,


    )

