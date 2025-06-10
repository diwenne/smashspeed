import torch
import cv2
import os
import argparse
import numpy as np
from pathlib import Path
from math import sqrt

# example usage 
# python speed_scaled.py --real_length_m 6 --fps 30
# fps will be detected if not set, real_length_m is in meters



# ------------------------ CONFIG ------------------------
input_video_path = '../raw_videos/clip_cosports_20250608_02.mov'
weights_path = '../yolov5/runs/train/smashspeed_v3/weights/best.pt'
# --------------------------------------------------------

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

def save_scale_info(pixel_length, scale_factor, video_path):
    os.makedirs("runs", exist_ok=True)  # Ensure 'runs/' exists
    output_name = Path(video_path).stem
    save_path = os.path.join("runs", f"{output_name}_scale_info.txt")
    with open(save_path, "w") as f:
        f.write(f"Pixel length: {pixel_length:.2f}\n")
        f.write(f"Scale factor (m/pixel): {scale_factor:.6f}\n")
    print(f"📁 Saved scale info to: {save_path}")

def draw_boxes_on_frame(frame, boxes, fps, prev_center=None, prev_frame_idx=None, current_frame_idx=0, scale_factor=None):
    speed_text = ""
    current_center = None
    if len(boxes) > 0:
        xmin, ymin, xmax, ymax = map(int, boxes[0][:4])
        center_x = (xmin + xmax) // 2
        center_y = (ymin + ymax) // 2
        current_center = (center_x, center_y)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        if prev_center is not None:
            dx = center_x - prev_center[0]
            dy = center_y - prev_center[1]
            dist = sqrt(dx**2 + dy**2)
            frame_diff = current_frame_idx - prev_frame_idx
            time_sec = frame_diff / fps if frame_diff > 0 else 1e-6
            px_per_sec = dist / time_sec if time_sec > 0 else 0
            speed_text = f"{px_per_sec:.1f} px/s"

            if scale_factor:
                m_per_sec = px_per_sec * scale_factor
                km_per_hr = m_per_sec * 3.6
                speed_text += f" | {km_per_hr:.1f} km/h"

            cv2.putText(frame, speed_text, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return frame, current_center, speed_text

def run_inference_on_video(video_path, model, fixed_fps=None, scale_factor=None):
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
            frame, boxes, fps, prev_center, prev_frame_idx, frame_idx, scale_factor=scale_factor
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

def get_reference_pixel_length(video_path):
    import cv2
    import math

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to load video.")
        return None

    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            print(f"✅ Point {len(points)} selected: ({x}, {y})")

    cv2.namedWindow("Select Reference")
    cv2.setMouseCallback("Select Reference", click_event)

    while True:
        display = frame.copy()
        for pt in points:
            cv2.circle(display, pt, 6, (0, 255, 0), -1)
        if len(points) == 2:
            cv2.line(display, points[0], points[1], (255, 0, 0), 2)
            cv2.imshow("Select Reference", display)
            print("✅ Both points selected.")
            cv2.waitKey(0)
            break

        cv2.imshow("Select Reference", display)
        if cv2.waitKey(10) & 0xFF == 27:  # ESC to exit early
            break

    cv2.destroyWindow("Select Reference")

    if len(points) == 2:
        dx = points[1][0] - points[0][0]
        dy = points[1][1] - points[0][1]
        return math.sqrt(dx ** 2 + dy ** 2)
    else:
        return None

def main():
    parser = argparse.ArgumentParser(description="YOLOv5 video analysis with speed overlay")
    parser.add_argument('--fps', type=float, default=None, help='Override FPS manually (default: detect from video)')
    parser.add_argument('--real_length_m', type=float, required=True, help='Real-world reference length in meters')
    args = parser.parse_args()

    processed_path = convert_mov_to_mp4(input_video_path)

    scale_factor = None
    if processed_path:
        # Always ask user to draw reference line to get pixel length
        pixel_length = get_reference_pixel_length(processed_path)
        if pixel_length:
            scale_factor = args.real_length_m / pixel_length
            print(f"📏 Detected pixel length: {pixel_length:.2f} px")
            print(f"📏 Scale factor: {scale_factor:.6f} m/pixel")
            save_scale_info(pixel_length, scale_factor, processed_path)

    if processed_path:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
        run_inference_on_video(processed_path, model, fixed_fps=args.fps, scale_factor=scale_factor)

if __name__ == "__main__":
    main()