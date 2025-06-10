import torch
import cv2
import os
import argparse
import numpy as np
from pathlib import Path
from math import sqrt

# ------------------------ CONFIG ------------------------
# Hardcoded input video path
input_video_path = '../raw_videos/clip_yt_20210131_02.mp4'

# Hardcoded weights path
weights_path = '../yolov5/runs/train/smashspeed_v3/weights/best.pt'
# --------------------------------------------------------

# Make sure runs/ exists
os.makedirs("runs", exist_ok=True)

def convert_mov_to_mp4(input_path):
    if not input_path.lower().endswith('.mov'):
        return input_path

    print("Converting .mov to .mp4...")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    temp_path = input_path.replace('.mov', '_converted.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"Saved converted video to {temp_path}")
    return temp_path

def draw_boxes_on_frame(frame, boxes, fps, prev_center=None, prev_frame_idx=None, current_frame_idx=0):
    speed_text = ""
    current_center = None

    if len(boxes) > 0:
        # Use first box
        xmin, ymin, xmax, ymax = map(int, boxes[0][:4])
        center_x = (xmin + xmax) // 2
        center_y = (ymin + ymax) // 2
        current_center = (center_x, center_y)

        # Draw bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        if prev_center is not None:
            dx = center_x - prev_center[0]
            dy = center_y - prev_center[1]
            dist = sqrt(dx**2 + dy**2)
            frame_diff = current_frame_idx - prev_frame_idx
            time_sec = frame_diff / fps if frame_diff > 0 else 1e-6
            px_per_frame = dist / frame_diff if frame_diff > 0 else 0
            px_per_sec = dist / time_sec if time_sec > 0 else 0
            speed_text = f"{px_per_frame:.1f} px/frame | {px_per_sec:.1f} px/s"

            # Write on top of box
            cv2.putText(frame, speed_text, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw detected FPS in top-left
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return frame, current_center, speed_text

def run_inference_on_video(video_path, model, fixed_fps=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    detected_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fixed_fps if fixed_fps else detected_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_name = Path(video_path).stem
    out_path = os.path.join("runs", f"boxed_{output_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    prev_center = None
    prev_frame_idx = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)
        boxes = results.xyxy[0].cpu().numpy()

        frame, current_center, speed_text = draw_boxes_on_frame(
            frame, boxes, fps, prev_center, prev_frame_idx, frame_idx
        )

        if current_center is not None:
            prev_center = current_center
            prev_frame_idx = frame_idx

        out_writer.write(frame)
        cv2.imshow("Output", frame)

        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()
    print(f"✅ Saved video to: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="YOLOv5 video analysis with speed overlay")
    parser.add_argument('--fps', type=float, default=None, help='Override FPS manually (default: detect from video)')
    args = parser.parse_args()

    # Convert input if .mov
    processed_path = convert_mov_to_mp4(input_video_path)
    if processed_path:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
        run_inference_on_video(processed_path, model, fixed_fps=args.fps)

if __name__ == "__main__":
    main()