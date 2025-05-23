import os
import argparse

import cv2

def extract_frames(video_path, output_folder):
    """Extracts frames from an OGV video and saves them as images."""

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_MSEC,(frame_count*1000)) # just one frame per sec, we don't need more        
        ret, frame = cap.read()
        if not ret:
            break

        output_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(output_path, frame)
        frame_count += 1

    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    extract_frames(args.video_path, args.output_path)
