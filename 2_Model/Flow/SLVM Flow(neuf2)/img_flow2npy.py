import os
from glob import glob
from PIL import Image
import numpy as np
from data_utils import frame_utils
from tqdm import tqdm


def convert_images_to_npy(image_dir, output_dir):
    image_files = sorted(glob(os.path.join(image_dir, '*.png')))
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Converting {len(image_files)} image(s) to .npy in {output_dir}")
    for image_file in tqdm(image_files):
        try:
            image = np.array(Image.open(image_file)).astype(np.uint8)
            base_name = os.path.splitext(os.path.basename(image_file))[0]
            out_path = os.path.join(output_dir, base_name + '.npy')
            np.save(out_path, image)
        except Exception as e:
            print(f"[ERROR] Failed to process image: {image_file} => {e}")


def convert_flows_to_npy(flow_glob_pattern, output_dir):
    flow_files = sorted(glob(flow_glob_pattern))
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Converting {len(flow_files)} flow file(s) to .npy in {output_dir}")
    for flow_file in tqdm(flow_files):
        try:
            flow = np.array(frame_utils.read_gen(flow_file)).astype(np.float32)
            base_name = os.path.splitext(os.path.basename(flow_file))[0]
            out_path = os.path.join(output_dir, base_name + '.npy')
            np.save(out_path, flow)
        except Exception as e:
            print(f"[ERROR] Failed to process flow: {flow_file} => {e}")


if __name__ == "__main__":
    dataset_root = 'datasets/DarkT'
    image_input_dir = os.path.join(dataset_root, 'training/image_2')
    # flow_glob_pattern = os.path.join(dataset_root, 'optical_flow/*/*/*/*/*/*.pfm')

    image_output_dir = os.path.join(dataset_root, 'training/image_2d_npy')
    # flow_output_dir = os.path.join(dataset_root, 'npy_output/flows')

    convert_images_to_npy(image_input_dir, image_output_dir)
    # convert_flows_to_npy(flow_glob_pattern, flow_output_dir)
