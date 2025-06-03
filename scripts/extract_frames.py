import os
import cv2
import numpy as np

def extract_frames(video_path, output_dir, total_frames):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    frame_indices = np.linspace(0, total_video_frames - 1, total_frames * 2, dtype=int)  # request extra in case of misses

    saved = 0
    attempted = 0
    i = 0  # actual frame counter for naming

    while saved < total_frames and attempted < len(frame_indices):
        frame_num = frame_indices[attempted]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            frame_name = f"{video_basename}_frame{i:04d}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)
            i += 1
            saved += 1
        else:
            print(f"Failed to read frame {frame_num}")
        attempted += 1

    cap.release()
    print(f"Extracted {saved} frames to {output_dir} (attempted {attempted})")

# Example usage
extract_frames("raw_videos/raw_ig_20250602_01.mov", "frames/raw_ig_20250602_01", total_frames=100)