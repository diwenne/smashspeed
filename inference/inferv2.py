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
        self.all_frames = []
        self.all_boxes = []

    def load_model(self):
        # Load YOLOv5 model from custom weights
        return torch.hub.load('ultralytics/yolov5', 'custom', path=str(self.weights_path))

    def get_video_info(self):
        # Extract video width, height, and frame rate using ffmpeg
        probe = ffmpeg.probe(str(self.video_path))
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width = int(video_info['width'])
        height = int(video_info['height'])
        fps = eval(video_info['r_frame_rate'])
        return width, height, fps

    def read_frames(self):
        # Generator to yield each frame as a NumPy array using ffmpeg
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
        # Display one frame for user to draw a real-world distance reference
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
        # Save scale info to file in runs/ directory
        output_name = self.video_path.stem
        save_path = RUNS_DIR / f"{output_name}_scale_info.txt"
        with open(save_path, "w") as f:
            f.write(f"Pixel length: {pixel_length:.2f}\n")
            f.write(f"Scale factor (m/pixel): {self.scale_factor:.6f}\n")
        print(f"📁 Saved scale info to: {save_path}")

    def draw_boxes_on_frame(self, frame, boxes, frame_idx,draw_speed=True):
        # Draw bounding boxes and speed overlay
        speed_text = ""
        current_center = None
        for i in range(len(boxes)):
            xmin, ymin, xmax, ymax = map(int, boxes[i][:4])
            center_x = (xmin + xmax) // 2
            center_y = (ymin + ymax) // 2
            current_center = (center_x, center_y)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            if self.prev_center is not None and draw_speed:
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

        cv2.putText(frame, f"FPS: {self.fps:.2f}", (10, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        return frame, current_center, speed_text

    def run_inference(self):
        """
        Runs YOLOv5 inference on the input video, draws bounding boxes, and stores the frames and boxes in memory.
        After processing, the user is prompted (within the OpenCV window) to either review the results or save them.
        """
        self.all_frames = []
        self.all_boxes = []

        for frame_idx, frame_rgb in enumerate(self.read_frames()):
            # Run YOLOv5 model inference
            results = self.model(frame_rgb)
            all_boxes = results.xyxy[0].cpu().numpy()
            best_box = [max(all_boxes, key=lambda b: b[4])] if len(all_boxes) > 0 else []

            # Convert RGB to BGR and store frame and boxes
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            self.all_frames.append(frame_bgr.copy())
            self.all_boxes.append(best_box)

            # Draw box and compute speed
            frame_bgr, current_center, _ = self.draw_boxes_on_frame(frame_bgr, best_box, frame_idx)
            if current_center is not None:
                self.prev_center = current_center
                self.prev_frame_idx = frame_idx

            # Display frame with bounding box
            cv2.imshow("Output", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Prompt user inside the OpenCV window instead of destroying it
        prompt_frame = self.all_frames[-1].copy()
        cv2.putText(prompt_frame, "Press 'r' to review or 's' to save and exit",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Output", prompt_frame)

        # Wait for review or save key
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('r'):
                self.review_mode()
                break
            elif key == ord('s'):
                self.save_video()
                break

        # Clean up after user choice
        cv2.destroyAllWindows()

    def save_video(self):
        # Save the video using frames and their associated boxes
        output_name = self.video_path.stem
        out_path = RUNS_DIR / f"boxed_{output_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(str(out_path), fourcc, self.fps, (self.width, self.height))

        for frame in self.all_frames:
            out_writer.write(frame)

        out_writer.release()
        print(f"✅ Saved reviewed video to: {out_path}")

    def review_mode(self):
        print("🛠️ Entered review mode. Use 'a'/'d' to browse, 'q' to quit and save")
        current_frame_idx = 0
        cv2.namedWindow("Review Mode")

        while True:
            frame = self.all_frames[current_frame_idx].copy()
            boxes = self.all_boxes[current_frame_idx]
            frame, _, _ = self.draw_boxes_on_frame(frame, boxes, current_frame_idx,draw_speed=False)
            cv2.putText(frame, f"Review Mode - Frame {current_frame_idx+1}/{len(self.all_frames)}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Review Mode", frame)

            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                print("🚪 Exiting review mode.")
                break
            elif key == ord('d'):  # Next frame
                current_frame_idx = min(current_frame_idx + 1, len(self.all_frames) - 1)
            elif key == ord('a'):  # Previous frame
                current_frame_idx = max(current_frame_idx - 1, 0)
            else:
                print(f"Pressed key code: {key}")

        cv2.destroyAllWindows()
        self.save_video()


def main():
    parser = argparse.ArgumentParser(description="YOLOv5 video analysis with speed overlay")
    parser.add_argument('--fps', type=float, default=None, help='Override FPS manually')
    parser.add_argument('--ref_len', type=float, default=3.91, help='Real-world reference length in meters')
    args = parser.parse_args()

    smash = SmashSpeed(INPUT_VIDEO_PATH, WEIGHTS_PATH, ref_len=args.ref_len, fps_override=args.fps)
    smash.draw_reference_line()
    smash.run_inference()

if __name__ == "__main__":
    main()