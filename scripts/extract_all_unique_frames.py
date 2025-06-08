import os
import cv2
import imagehash
from PIL import Image

def extract_all_unique_frames(video_path, output_dir, hash_diff_threshold=5):
    """
    Extracts unique frames from a video using perceptual hashing (pHash).
    Only frames that are visually different from the last saved frame are written.

    Args:
        video_path (str): Path to the input video.
        output_dir (str): Directory to save unique frames.
        hash_diff_threshold (int): Max hash difference allowed before saving new frame.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    i = 0  # saved frame index
    frame_idx = 0  # all frame index
    last_hash = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR (OpenCV) to RGB (PIL) and then to PIL Image
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        curr_hash = imagehash.phash(pil_frame)

        # Save frame if it's visually different from the last
        if last_hash is None or abs(curr_hash - last_hash) >= hash_diff_threshold:
            frame_name = f"{video_basename}_frame{i:04d}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)
            last_hash = curr_hash
            i += 1

        frame_idx += 1

    cap.release()
    print(f"Saved {i} unique frames out of {frame_idx} total frames to: {output_dir}")

# Example usage

extract_all_unique_frames(video_path="raw_videos/clip_yt_20210131_17.mp4",output_dir="frames/unique_clip_yt_20210131_17",hash_diff_threshold=0)