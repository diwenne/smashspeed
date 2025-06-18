import torch
import cv2
import os
import argparse
import numpy as np
from pathlib import Path
from math import sqrt
import ffmpeg

# ------------------------ CONFIG ------------------------
BASE_DIR = Path(__file__).resolve().parent
input_video_path = (BASE_DIR / "../raw_videos/clip_cosports_20250608_02.mov").resolve()
weights_path = (BASE_DIR / "../yolov5/runs/train/smashspeed_v3/weights/best.pt").resolve()
# --------------------------------------------------------

os.makedirs(BASE_DIR / "runs", exist_ok=True)

def get_video_info(video_path):
    probe = ffmpeg.probe(str(video_path))
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    fps = eval(video_info['r_frame_rate'])
    return width, height, fps

def read_frames_ffmpeg(video_path, width, height):
    process = (
        ffmpeg.input(str(video_path))
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run_async(pipe_stdout=True)
    )
    frame_size = width * height * 3
    while True:
        in_bytes = process.stdout.read(frame_size)
        if not in_bytes:
            break
        frame = (
            np.frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )
        yield frame
    process.wait()

def save_scale_info(pixel_length, scale_factor, video_path):
    output_name = Path(video_path).stem
    save_path = BASE_DIR / "runs" / f"{output_name}_scale_info.txt"
    with open(save_path, "w") as f:
        f.write(f"Pixel length: {pixel_length:.2f}\n")
        f.write(f"Scale factor (m/pixel): {scale_factor:.6f}\n")
    print(f"📁 Saved scale info to: {save_path}")

def draw_boxes_on_frame(frame, boxes, fps, prev_center=None, prev_frame_idx=None, current_frame_idx=0, scale_factor=None):
    speed_text = ""
    current_center = None
    for i in range(len(boxes)):
        xmin, ymin, xmax, ymax = map(int, boxes[i][:4])
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

def get_reference_pixel_length(video_path):
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
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
            cv2.waitKey(0)
            break

        cv2.imshow("Select Reference", display)
        if cv2.waitKey(10) & 0xFF == 27:
            break

    cv2.destroyWindow("Select Reference")

    if len(points) == 2:
        dx = points[1][0] - points[0][0]
        dy = points[1][1] - points[0][1]
        return sqrt(dx ** 2 + dy ** 2)
    else:
        return None

def run_inference_on_video(video_path, model, fixed_fps=None, scale_factor=None):
    width, height, detected_fps = get_video_info(video_path)
    fps = fixed_fps if fixed_fps else detected_fps
    output_name = Path(video_path).stem
    out_path = BASE_DIR / "runs" / f"boxed_{output_name}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    prev_center = None
    prev_frame_idx = None

    for frame_idx, frame_rgb in enumerate(read_frames_ffmpeg(video_path, width, height)):
        results = model(frame_rgb)
        all_boxes = results.xyxy[0].cpu().numpy()
        boxes = []

        if len(all_boxes) > 0:
        # Get box with highest confidence (index 4 is confidence)
            best_box = max(all_boxes, key=lambda b: b[4])
            boxes = [best_box]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_bgr, current_center, _ = draw_boxes_on_frame(
            frame_bgr, boxes, fps, prev_center, prev_frame_idx, frame_idx, scale_factor=scale_factor
        )
        if current_center is not None:
            prev_center = current_center
            prev_frame_idx = frame_idx
        out_writer.write(frame_bgr)
        cv2.imshow("Output", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out_writer.release()
    cv2.destroyAllWindows()
    print(f"✅ Saved video to: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="YOLOv5 video analysis with speed overlay")
    parser.add_argument('--fps', type=float, default=None, help='Override FPS manually (default: detect from video)')
    parser.add_argument('--ref_len', type=float, default=3.91, help='Real-world reference length in meters (default: 3.91)')
    args = parser.parse_args()

    scale_factor = None
    pixel_length = get_reference_pixel_length(input_video_path)
    if pixel_length:
        scale_factor = args.ref_len / pixel_length
        print(f"📏 Detected pixel length: {pixel_length:.2f} px")
        print(f"📏 Scale factor: {scale_factor:.6f} m/pixel")
        save_scale_info(pixel_length, scale_factor, input_video_path)

    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
    run_inference_on_video(input_video_path, model, fixed_fps=args.fps, scale_factor=scale_factor)

if __name__ == "__main__":
    main()