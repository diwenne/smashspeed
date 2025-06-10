import torch
import cv2
import os
from datetime import datetime
from pathlib import Path

# Paths
weights_path = '../yolov5/runs/train/smashspeed_v3/weights/best.pt'  # your weights
input_video_path = '../raw_videos/clip_cosports_20250608_01.mov'     # your input video

# Ensure 'runs' folder exists
os.makedirs("runs", exist_ok=True)

# Convert .mov to .mp4 if necessary
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

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)

# Draw bounding boxes
def draw_boxes_on_frame(frame, boxes):
    for box in boxes:
        xmin, ymin, xmax, ymax = map(int, box[:4])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    return frame

# Run YOLO inference on video
def run_inference_on_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Derive output filename from input
    original_name = Path(video_path).stem  # filename without extension
    out_filename = f"boxed_{original_name}.mp4"
    out_path = os.path.join("runs", out_filename)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)
        boxes = results.xyxy[0].cpu().numpy()
        frame = draw_boxes_on_frame(frame, boxes)

        out.write(frame)
        cv2.imshow('Video with Bounding Boxes', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Saved output video to: {out_path}")

# Convert and process
converted_path = convert_mov_to_mp4(input_video_path)
if converted_path:
    run_inference_on_video(converted_path)