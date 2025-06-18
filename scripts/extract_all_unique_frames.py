import os
import cv2
import imagehash
from PIL import Image
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

'''
frame extraction:
extracts ALL unique frames from a raw video, with an optional perceptual hashing threshold if you want the frames to be different to some degree, 
'''

def extract_all_unique_frames(video_path, output_dir, hash_diff_threshold=0):
    """
    Extracts all unique frames from a video using perceptual hashing (pHash).
    Only frames that are visually different from the last saved frame are written.

    Args:
        video_path (str): Path to the input video.
        output_dir (str): Directory to save unique frames.
        hash_diff_threshold (int): Max hash difference allowed before saving new frame.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return

    video_basename = video_path.stem

    i = 0  # saved frame index
    frame_idx = 0  # all frame index
    last_hash = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for perceptual hashing
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        curr_hash = imagehash.phash(pil_frame)

        # Save frame if it's visually different from the last
        if last_hash is None or abs(curr_hash - last_hash) >= hash_diff_threshold:
            frame_name = f"{video_basename}_frame{i:04d}.jpg"
            save_path = output_dir / frame_name

            # Save JPEG with max quality (or use PNG for lossless)
            cv2.imwrite(save_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            # To use PNG instead, uncomment this line and comment the one above:
            # save_path = save_path.replace(".jpg", ".png")
            # cv2.imwrite(save_path, frame)

            last_hash = curr_hash
            i += 1

        frame_idx += 1

    cap.release()
    print(f"✅ Saved {i} unique frames out of {frame_idx} total frames to: {output_dir}")

# Example usage
extract_all_unique_frames(
    video_path=(BASE_DIR/"../raw_videos/clip_cosports_20250616_06.mov").resolve(),
    output_dir=(BASE_DIR/"../frames/clip_cosports_20250616_06").resolve(),
    hash_diff_threshold=0
)