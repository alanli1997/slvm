import cv2
import argparse
import os
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Camera Live Preview Tool")
    parser.add_argument("--camera_id", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--width", type=int, default=640, help="Preview width")
    parser.add_argument("--height", type=int, default=640, help="Preview height")
    parser.add_argument("--save_dir", type=str, default="saved_frames", help="Directory to save snapshots")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"❌ Cannot open camera with ID {args.camera_id}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    os.makedirs(args.save_dir, exist_ok=True)
    print("✅ Press [s] to save current frame, [q] to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        cv2.imshow("🔍 Camera Preview", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(args.save_dir, f"frame_{ts}.jpg")
            cv2.imwrite(path, frame)
            print(f"💾 Saved snapshot to {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Preview ended.")


if __name__ == "__main__":
    main()
