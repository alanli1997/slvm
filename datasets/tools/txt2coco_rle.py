import os
import json
import cv2
from tqdm import tqdm
from pycocotools import mask as maskUtils
import numpy as np

#
categories = [
    # yolo-predictions format
    # {"id": 0, "name": "bicycle"},
    # {"id": 1, "name": "car"},
    # {"id": 2, "name": "motorbike"},
    # {"id": 3, "name": "bus"},
    # {"id": 4, "name": "person"},
    # {"id": 5, "name": "traffic light"}

    # coco format
    {"id": 1, "name": "bicycle"},
    {"id": 2, "name": "car"},
    {"id": 3, "name": "motorbike"},
    {"id": 4, "name": "bus"},
    {"id": 5, "name": "person"},
    {"id": 6, "name": "traffic light"}
]

category_map = {i: cat["id"] for i, cat in enumerate(categories)}


def yolo_to_coco_segmentation(label_path, img_w, img_h):
    annotations = []
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for ann_id, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        class_id = int(parts[0])
        norm_coords = list(map(float, parts[1:]))

        # gain bbox
        xc, yc, w, h = norm_coords[:4]
        bbox = [
            round((xc - w / 2) * img_w, 6),
            round((yc - h / 2) * img_h, 6),
            round(w * img_w, 6),
            round(h * img_h, 6)
        ]

        # segmentation points
        segm_pts = norm_coords[4:]
        segm = []
        for i in range(0, len(segm_pts), 2):
            x = round(segm_pts[i] * img_w, 6)
            y = round(segm_pts[i + 1] * img_h, 6)
            segm.extend([x, y])

        # RLE encoding
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        poly = np.array(segm, dtype=np.float32).reshape((-1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [poly], 1)
        rle = maskUtils.encode(np.asfortranarray(mask))
        area = float(maskUtils.area(rle))
        rle["counts"] = rle["counts"].decode("utf-8")  # json

        annotations.append({
            "id": ann_id,
            "image_id": None,  #
            "category_id": category_map[class_id],
            "segmentation": rle,
            "area": round(area, 6),
            "bbox": bbox,
            "iscrowd": 0
        })
    return annotations


def convert_yolo_to_coco(images_dir, labels_dir, output_json):
    images = []
    annotations = []
    ann_id = 0
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])
    for img_idx, img_file in tqdm(enumerate(image_files), total=len(image_files)):
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + ".txt")
        if not os.path.exists(label_path):
            continue

        image = cv2.imread(img_path)
        height, width = image.shape[:2]
        image_id = int(os.path.splitext(img_file)[0])  # 0000000010.jpg → id=10

        images.append({
            "id": image_id,
            "file_name": img_file,
            "width": width,
            "height": height
        })

        anns = yolo_to_coco_segmentation(label_path, width, height)
        for ann in anns:
            ann["id"] = ann_id
            ann["image_id"] = image_id
            annotations.append(ann)
            ann_id += 1

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_json, "w") as f:
        json.dump(coco_dict, f)
    print(f"✅ transfer completed, saving to {output_json}")

# 用法示例（修改路径）
# to modify the path to yours datasets
convert_yolo_to_coco(
    images_dir="images/val",
    labels_dir="labels/val",
    # the validation annotations json filename, for evaluation of coco format
    # 输出你的验证json文件用于coco格式的精度评估
    output_json="val_bbox.json" 
)
