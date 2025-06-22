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
INPUT_VIDEO_PATH = (BASE_DIR / "videos/hao7.mov").resolve()
WEIGHTS_PATH = (BASE_DIR / "../yolov5/runs/train/smashspeed_v4/weights/best.pt").resolve()
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
        self.prev_timestamp = None
        self.all_frames = []
        self.all_boxes = []
        self.all_timestamps = []

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
        process = (
            ffmpeg.input(str(self.video_path))
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdout=True)
        )
        frame_size = self.width * self.height * 3
        frame_idx = 0
        while True:
            in_bytes = process.stdout.read(frame_size)
            if not in_bytes:
                break
            frame = (
                np.frombuffer(in_bytes, np.uint8)
                .reshape([self.height, self.width, 3])
            )
            timestamp = frame_idx / self.fps
            yield frame, timestamp
            frame_idx += 1
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

    def draw_boxes_on_frame(self, frame, boxes, timestamp, draw_speed=True):
        speed_text = ""
        current_center = None

        for box in boxes:
            xmin, ymin, xmax, ymax = map(int, box[:4])
            box_w = xmax - xmin
            box_h = ymax - ymin
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2

            # Determine direction-aware tip
            if self.prev_center is None:
                tip_x, tip_y = cx, cy
            else:
                dx = cx - self.prev_center[0]
                dy = cy - self.prev_center[1]
                if abs(dx) > abs(dy):
                    tip_x = xmax if dx >= 0 else xmin
                    tip_y = cy
                else:
                    tip_x = cx
                    tip_y = ymax if dy >= 0 else ymin

            current_center = (tip_x, tip_y)

            # Draw bounding box and tip marker
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            cv2.circle(frame, current_center, 4, (0, 255, 0), -1)  # green dot for tip

            # Compute and draw speed
            if self.prev_center and self.prev_timestamp is not None and draw_speed:
                dx = tip_x - self.prev_center[0]
                dy = tip_y - self.prev_center[1]
                dist = sqrt(dx**2 + dy**2)
                dt = max(timestamp - self.prev_timestamp, 1e-6)
                px_per_sec = dist / dt
                speed_text = f"{px_per_sec:.1f} px/s"
                if self.scale_factor:
                    kmph = px_per_sec * self.scale_factor * 3.6
                    speed_text += f" | {kmph:.1f} km/h"
                cv2.putText(frame, speed_text, (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw timestamp regardless of box presence
        cv2.putText(frame, f"Time: {timestamp:.3f}s", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame, current_center, speed_text

    def run_inference(self):
        self.all_frames.clear()
        self.all_boxes.clear()
        self.all_timestamps.clear()

        for frame, timestamp in self.read_frames():
            results = self.model(frame)
            all_boxes = results.xyxy[0].cpu().numpy()
            best_box = [max(all_boxes, key=lambda b: b[4])] if len(all_boxes) > 0 else []

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.all_frames.append(frame_bgr.copy())
            self.all_boxes.append(best_box)
            self.all_timestamps.append(timestamp)

            frame_bgr, current_center, _ = self.draw_boxes_on_frame(frame_bgr, best_box, timestamp)
            if current_center is not None:
                self.prev_center = current_center
                self.prev_timestamp = timestamp

            cv2.imshow("Output", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        prompt_frame = self.all_frames[-1].copy()
        cv2.putText(prompt_frame, "Press 'r' to review or 'q' to save and exit",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Output", prompt_frame)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('r'):
                self.review_mode()
                break
            elif key == ord('q'):
                self.save_video()
                break

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
        current_frame_idx = 0
        self.modified_boxes = {}

        cv2.namedWindow("Review Mode")
        dragging = False
        drag_offset = (0, 0)

        def mouse_callback(event, x, y, flags, param):
            nonlocal dragging, drag_offset
            boxes = self.all_boxes[current_frame_idx]
            if not boxes:
                return
            box = boxes[0]
            xmin, ymin, xmax, ymax = map(int, box[:4])
            cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2

            if event == cv2.EVENT_LBUTTONDOWN:
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    dragging = True
                    drag_offset = (x - cx, y - cy)

            elif event == cv2.EVENT_MOUSEMOVE and dragging:
                dx, dy = drag_offset
                new_cx = x - dx
                new_cy = y - dy
                w = xmax - xmin
                h = ymax - ymin
                new_box = [new_cx - w // 2, new_cy - h // 2, new_cx + w // 2, new_cy + h // 2, box[4], box[5]]
                self.all_boxes[current_frame_idx] = [new_box]

            elif event == cv2.EVENT_LBUTTONUP:
                dragging = False
                self.modified_boxes[current_frame_idx] = self.all_boxes[current_frame_idx]

        cv2.setMouseCallback("Review Mode", mouse_callback)

        while True:
            frame = self.all_frames[current_frame_idx].copy()
            boxes = self.all_boxes[current_frame_idx]
            frame, _, _ = self.draw_boxes_on_frame(frame, boxes, current_frame_idx, draw_speed=False)

            # Frame info
            cv2.putText(frame, f"Review Mode - Frame {current_frame_idx+1}/{len(self.all_frames)}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Instructions overlay
            instructions = [
                "'a': Prev frame",
                "'d': Next frame",
                "Click + drag: Move box",
                "'n': New box",
                "'q': Finish & Recalculate"
            ]
            for i, text in enumerate(instructions):
                cv2.putText(frame, text, (30, 70 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            cv2.imshow("Review Mode", frame)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('q'):
                print("✅ Review complete. Recalculating speed...")
                break
            elif key == ord('d'):
                current_frame_idx = min(current_frame_idx + 1, len(self.all_frames) - 1)
            elif key == ord('a'):
                current_frame_idx = max(current_frame_idx - 1, 0)
            elif key == ord('n'):
                frame_h, frame_w = self.all_frames[current_frame_idx].shape[:2]
                box_w, box_h = 60, 60
                cx, cy = frame_w // 2, frame_h // 2
                new_box = [cx - box_w // 2, cy - box_h // 2, cx + box_w // 2, cy + box_h // 2, 1.0, 0]
                self.all_boxes[current_frame_idx] = [new_box]
                self.modified_boxes[current_frame_idx] = [new_box]

        cv2.destroyAllWindows()
        self.recalculate_speed()
    
    
    def recalculate_speed(self):
        print("🔁 Running updated inference with modified boxes...")

        self.prev_center = None
        self.prev_timestamp = None

        output_name = self.video_path.stem
        out_path = RUNS_DIR / f"reviewed_{output_name}.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height))

        cv2.namedWindow("Final Output", cv2.WINDOW_NORMAL)
        # cv2.waitKey(500)
        for idx, frame in enumerate(self.all_frames):
            boxes = self.all_boxes[idx]
            timestamp = self.all_timestamps[    idx]

            frame_copy = frame.copy()
            frame_out, current_center, _ = self.draw_boxes_on_frame(frame_copy, boxes, timestamp, draw_speed=True)

            if current_center is not None:
                self.prev_center = current_center
                self.prev_timestamp = timestamp

            writer.write(frame_out)
            cv2.imshow("Final Output", frame_out)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("⏹️ Display interrupted by user.")
                break

        writer.release()

        # Show the last frame and wait for a key to close
        cv2.imshow("Final Output", frame_out)
        print("✅ Finished. Press any key to close window.")
        # cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f"💾 Final reviewed video saved: {out_path}")


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