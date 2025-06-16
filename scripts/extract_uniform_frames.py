import os
import cv2
import numpy as np
from pathlib import Path

'''
frame extraction:
only extracts a specified number of frames, uniformly, from a vid
'''

BASE_DIR = Path(__file__).resolve().parent

def extract_uniform_frames(video_path: Path, output_dir: Path, total_frames: int):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))  
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_basename = video_path.stem

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
            save_path = output_dir / frame_name  
            cv2.imwrite(str(save_path), frame)  # Convert to str when saving
            i += 1
            saved += 1
        else:
            print(f"Failed to read frame {frame_num}")
        attempted += 1

    cap.release()
    print(f"Extracted {saved} frames to {output_dir} (attempted {attempted})")

# Example usage
extract_uniform_frames(
    (BASE_DIR / "../raw_videos/raw_ss_20250614_01.mov").resolve(),
    (BASE_DIR / "../frames/raw_ss_20250614_01").resolve(),
    total_frames=40)