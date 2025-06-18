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
INPUT_VIDEO_PATH = (BASE_DIR / "../raw_videos/clip_cosports_20250608_02.mov").resolve()
WEIGHTS_PATH = (BASE_DIR / "../yolov5/runs/train/smashspeed_v3/weights/best.pt").resolve()
RUNS_DIR = BASE_DIR / "runs"
RUNS_DIR.mkdir(exist_ok=True)
# --------------------------------------------------------

class SmashSpeed:
    def __init__(self, video_path, weights_path, ref_len=3.91, fps_override=None):
        self.video_path = Path(video_path)
        self.weights_path = Path(weights_path)
        self.ref_len = ref_len
        self.fps_override = fps_override
        self.model = self.load_model()
        self.width, self.height, self.fps = self.get_video_info()
        if self.fps_override:
            self.fps = self.fps_override
        self.scale_factor = None
        self.prev_center = None
        self.prev_frame_idx = None

    def load_model(self):
        return torch.hub.load('ultralytics/yolov5', 'custom', path=str(self.weights_path))

    def get_video_info(self):
        probe = ffmpeg.probe(str(self.video_path))
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width = int(video_info['width'])
        height = int(video_info['height'])
        fps = eval(video_info['r_frame_rate'])
        return width, height, fps

    def read_frames(self):
        process = (
            ffmpeg.input(str(self.video_path))
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdout=True)
        )
        frame_size = self.width * self.height * 3
        while True:
            in_bytes = process.stdout.read(frame_size)
            if not in_bytes:
                break
            frame = (
                np.frombuffer(in_bytes, np.uint8)
                .reshape([self.height, self.width, 3])
            )
            yield frame
        process.wait()

    def draw_reference_line(self):
        cap = cv2.VideoCapture(str(self.video_path))
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
            pixel_length = sqrt(dx ** 2 + dy ** 2)
            self.scale_factor = self.ref_len / pixel_length
            self.save_scale_info(pixel_length)

    def save_scale_info(self, pixel_length):
        output_name = self.video_path.stem
        save_path = RUNS_DIR / f"{output_name}_scale_info.txt"
        with open(save_path, "w") as f:
            f.write(f"Pixel length: {pixel_length:.2f}\n")
            f.write(f"Scale factor (m/pixel): {self.scale_factor:.6f}\n")
        print(f"📁 Saved scale info to: {save_path}")

    def draw_boxes_on_frame(self, frame, boxes, frame_idx):
        speed_text = ""
        current_center = None
        for i in range(len(boxes)):
            xmin, ymin, xmax, ymax = map(int, boxes[i][:4])
            center_x = (xmin + xmax) // 2
            center_y = (ymin + ymax) // 2
            current_center = (center_x, center_y)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            if self.prev_center is not None:
                dx = center_x - self.prev_center[0]
                dy = center_y - self.prev_center[1]
                dist = sqrt(dx**2 + dy**2)
                frame_diff = frame_idx - self.prev_frame_idx
                time_sec = frame_diff / self.fps if frame_diff > 0 else 1e-6
                px_per_sec = dist / time_sec if time_sec > 0 else 0
                speed_text = f"{px_per_sec:.1f} px/s"

                if self.scale_factor:
                    m_per_sec = px_per_sec * self.scale_factor
                    km_per_hr = m_per_sec * 3.6
                    speed_text += f" | {km_per_hr:.1f} km/h"

                cv2.putText(frame, speed_text, (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(frame, f"FPS: {self.fps:.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return frame, current_center, speed_text

    def run_inference(self):
        output_name = self.video_path.stem
        out_path = RUNS_DIR / f"boxed_{output_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(str(out_path), fourcc, self.fps, (self.width, self.height))

        for frame_idx, frame_rgb in enumerate(self.read_frames()):
            results = self.model(frame_rgb)
            all_boxes = results.xyxy[0].cpu().numpy()
            boxes = [max(all_boxes, key=lambda b: b[4])] if len(all_boxes) > 0 else []
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_bgr, current_center, _ = self.draw_boxes_on_frame(
                frame_bgr, boxes, frame_idx
            )
            if current_center is not None:
                self.prev_center = current_center
                self.prev_frame_idx = frame_idx
            out_writer.write(frame_bgr)
            cv2.imshow("Output", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out_writer.release()
        cv2.destroyAllWindows()
        print(f"✅ Saved video to: {out_path}")

    def review_mode(self):
        print("🔍 Review mode not yet implemented. Placeholder for manual bounding box adjustments.")


def main():
    parser = argparse.ArgumentParser(description="YOLOv5 video analysis with speed overlay")
    parser.add_argument('--fps', type=float, default=None, help='Override FPS manually')
    parser.add_argument('--ref_len', type=float, default=3.91, help='Real-world reference length in meters')
    parser.add_argument('--review', action='store_true', help='Enter manual bounding box review mode after inference')
    args = parser.parse_args()

    smash = SmashSpeed(INPUT_VIDEO_PATH, WEIGHTS_PATH, ref_len=args.ref_len, fps_override=args.fps)
    smash.draw_reference_line()
    smash.run_inference()

    if args.review:
        smash.review_mode()

if __name__ == "__main__":
    main()