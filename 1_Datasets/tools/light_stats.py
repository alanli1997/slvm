import os
import cv2
import numpy as np
from collections import defaultdict

# ISD: 0 bicycle, 1 car, 2 motobike, 3 bus
# coco: 0 person, 1 bicycle, 2 car, 3 moto, 5 bus, 9 trafic_light
# 0 bicycle, 1 car, 2 motobike, 3 bus, 4 person, 5 traffic-light


def analyze_dark_dataset(image_dir, label_dir):
    # init
    channel_stats = {'r': [], 'g': [], 'b': []}
    class_dist = defaultdict(int)

    # read images
    for img_name in os.listdir(image_dir):
        if not img_name.endswith(('.jpg', '.png')):
            continue

        # proceeding
        img_path = os.path.join(image_dir, img_name)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        for i, ch in enumerate(['r', 'g', 'b']):
            channel = img[:, :, i].flatten()
            channel_stats[ch].append([np.mean(channel), np.std(channel)])

        # annotations
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + '.txt')
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    class_id = int(line.strip().split()[0])
                    class_dist[class_id] += 1

    # report
    print("=== Channel-feature distribution ===")
    for ch in ['r', 'g', 'b']:
        data = np.array(channel_stats[ch])
        means = data[:, 0]  # mean channel-wise
        stds = data[:, 1]  # std channel-wise
        print(f"{ch.upper()}channel - mean-area: {means.min():.1f}-{means.max():.1f}")
        print(f"{ch.upper()}channel - std-area: {stds.min():.1f}-{stds.max():.1f}")

        print(f"{ch.upper()}channel - median mean: {np.median(means):.1f}")
        print(f"{ch.upper()}channel - median standard deviation: {np.median(stds):.1f}")

    print("\n=== Category distribution ===")
    for cls, count in sorted(class_dist.items()):
        print(f"Category {cls}: {count} instances")


if __name__ == "__main__":
    analyze_dark_dataset(
        image_dir="./DT/images/val",
        label_dir="./DT/labels/val"
    )