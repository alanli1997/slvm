import torch
import cv2
import os
import argparse
import numpy as np
from glob import glob
from NeuFlow.neuflow import NeuFlow
from NeuFlow.backbone_v7 import ConvBlock
from data_utils import flow_viz


def fuse_conv_and_bn(conv, bn):
    fusedconv = torch.nn.Conv2d(
        conv.in_channels, conv.out_channels,
        kernel_size=conv.kernel_size, stride=conv.stride,
        padding=conv.padding, dilation=conv.dilation,
        groups=conv.groups, bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def preprocess_frame(frame, width, height):
    frame = cv2.resize(frame, (width, height))
    image = torch.from_numpy(frame).permute(2, 0, 1).half()  # CHW
    return image[None].cuda(), frame


def load_model(checkpoint_path, width, height):
    model = NeuFlow().cuda()
    ckpt = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(ckpt['model'], strict=True)

    for m in model.modules():
        if isinstance(m, ConvBlock):
            m.conv1 = fuse_conv_and_bn(m.conv1, m.norm1)
            m.conv2 = fuse_conv_and_bn(m.conv2, m.norm2)
            delattr(m, "norm1")
            delattr(m, "norm2")
            m.forward = m.forward_fuse

    model.eval()
    model.half()
    model.init_bhwd(1, height, width, 'cuda')
    return model


def infer_pairwise(model, img_paths, out_dir, width, height):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for p0, p1 in zip(img_paths[:-1], img_paths[1:]):
        name = os.path.basename(p0)
        frame0 = cv2.imread(p0)
        frame1 = cv2.imread(p1)
        img0, raw0 = preprocess_frame(frame0, width, height)
        img1, _ = preprocess_frame(frame1, width, height)

        with torch.no_grad():
            flow = model(img0, img1)[-1][0].permute(1, 2, 0).cpu().numpy()
            flow_img = flow_viz.flow_to_image(flow)
            stacked = np.vstack([cv2.resize(raw0, (width, height)), flow_img])
            cv2.imwrite(os.path.join(out_dir, name), stacked)


def infer_video(model, video_path, out_dir, width, height, camera=False):
    cap = cv2.VideoCapture(0 if camera else video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video or camera!")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    frame_id = 0
    ret, prev = cap.read()
    while ret:
        ret, curr = cap.read()
        if not ret:
            break

        img0, raw0 = preprocess_frame(prev, width, height)
        img1, _ = preprocess_frame(curr, width, height)

        with torch.no_grad():
            flow = model(img0, img1)[-1][0].permute(1, 2, 0).cpu().numpy()
            flow_img = flow_viz.flow_to_image(flow)
            raw0_resized = cv2.resize(prev, (width, height))
            stacked = np.vstack([raw0_resized, flow_img])

            cv2.imshow("NeuFlow - Raw + Flow", stacked)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            save_path = os.path.join(out_dir, f"{frame_id:06d}.jpg")
            cv2.imwrite(save_path, stacked)

        prev = curr
        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="SLVM Inference Script")
    parser.add_argument("--checkpoint", type=str, default='./checkpoints/step_010000.pth', help="Path to model checkpoint")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--image_dir", type=str, help="Folder with .png/.jpg for image pair inference")
    parser.add_argument("--video", type=str, default='', help="Path to input video file")
    parser.add_argument("--camera", action="store_true", default=1, help="Use webcam as input")
    parser.add_argument("--out_dir", type=str, default="outputs/cemera1", help="Path to output directory")
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.width, args.height)

    if args.image_dir:
        image_paths = sorted(glob(os.path.join(args.image_dir, '*.png')) + glob(os.path.join(args.image_dir, '*.jpg')))
        infer_pairwise(model, image_paths, args.out_dir, args.width, args.height)

    elif args.video:
        infer_video(model, args.video, args.out_dir, args.width, args.height, camera=False)

    elif args.camera:
        infer_video(model, None, args.out_dir, args.width, args.height, camera=True)

    else:
        print("Please specify either --image_dir or --video or --camera.")
        exit(1)


if __name__ == "__main__":
    main()
