'''
SLVM-tools:
This script will help you convert YOLO formatted txt annotations to coco formatted json annotations
for accuracy evaluation using coco format.
'''
import os
import json
import cv2
from tqdm import tqdm

# 设置类别名与对应 category_id（确保与 predictions.json 对应）
CLASS_NAMES = ["bicycle", "car", "motorbike", "bus", "person", "traffic light"]
CATEGORY_DICT = {name: i for i, name in enumerate(CLASS_NAMES)}

# 参数配置
YOLO_LABEL_PATH = 'labels/val'   # 存放 .txt 的目录
YOLO_IMAGE_PATH = 'images/val'   # 存放 .jpg 的目录
OUTPUT_JSON_PATH = 'val_bbox_1.json'

def create_coco_json(yolo_label_path, yolo_image_path, output_json_path):
    images = []
    annotations = []
    annotation_id = 1  # 每个目标一个唯一 ID
    categories = [{"id": v, "name": k} for k, v in CATEGORY_DICT.items()]

    image_files = sorted([f for f in os.listdir(yolo_image_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    for img_file in tqdm(image_files):
        img_id_str = os.path.splitext(img_file)[0]
        try:
            image_id = int(img_id_str)
        except ValueError:
            print(f"⚠️ 警告：图像名 {img_file} 无法转换为 image_id，跳过。")
            continue

        img_path = os.path.join(yolo_image_path, img_file)
        label_path = os.path.join(yolo_label_path, img_id_str + '.txt')

        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ 警告：图像 {img_file} 无法读取，跳过。")
            continue
        height, width = img.shape[:2]

        images.append({
            "file_name": img_file,
            "height": height,
            "width": width,
            "id": image_id
        })

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # 非法行
            class_id, x_center, y_center, box_w, box_h = map(float, parts)
            class_id = int(class_id)

            # 坐标转换（YOLO → COCO）
            bbox_w = box_w * width
            bbox_h = box_h * height
            x_min = (x_center * width) - (bbox_w / 2)
            y_min = (y_center * height) - (bbox_h / 2)

            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id+1,  # COCO类别ID从1开始
                "bbox": [round(x_min, 2), round(y_min, 2), round(bbox_w, 2), round(bbox_h, 2)],
                "area": round(bbox_w * bbox_h, 2),
                "iscrowd": 0
            })
            annotation_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_json_path, 'w') as f:
        json.dump(coco_format, f, indent=2)
    print(f"✅ 成功写入 COCO 标注 JSON 到 {output_json_path}")

# 执行转换
create_coco_json(YOLO_LABEL_PATH, YOLO_IMAGE_PATH, OUTPUT_JSON_PATH)
