'''
SLVM-tools:
This script will help you visually check the correctness of all dense annotations.
'''
import os
import cv2
import matplotlib.pyplot as plt


def visualize_seg_and_det(
    image_dir, seg_label_dir, det_label_dir, image_id
):
    # read
    img_path = os.path.join(image_dir, f"{image_id}.jpg")
    seg_path = os.path.join(seg_label_dir, f"{image_id}.txt")
    det_path = os.path.join(det_label_dir, f"{image_id}.txt")

    image = cv2.imread(img_path)
    if image is None:
        print(f"read falied: {img_path}")
        return
    h, w, _ = image.shape

    # mask
    if os.path.exists(seg_path):
        with open(seg_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue  # > 3 points
                coords = list(map(float, parts[1:]))
                points = [(int(x * w), int(y * h)) for x, y in zip(coords[0::2], coords[1::2])]
                cv2.polylines(image, [np.array(points)], isClosed=True, color=(255, 0, 0), thickness=2)

    # bbox
    if os.path.exists(det_path):
        with open(det_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, x_center, y_center, box_w, box_h = map(float, parts)
                x1 = int((x_center - box_w / 2) * w)
                y1 = int((y_center - box_h / 2) * h)
                x2 = int((x_center + box_w / 2) * w)
                y2 = int((y_center + box_h / 2) * h)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # results
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(image_rgb)
    plt.title(f"Image {image_id}: Blue=Segmentation | Red=Detection Box")
    plt.axis("off")
    plt.show()

#
import numpy as np
visualize_seg_and_det(
    image_dir="images/train",
    seg_label_dir="seglabels/train",
    det_label_dir="detlabels/train",
    image_id="0000000000"
)
