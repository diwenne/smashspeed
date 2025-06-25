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

class Box:
    """
    Represents a bounding box with position, confidence, and class label.

    Attributes:
        xmin, ymin, xmax, ymax (int): Coordinates of the bounding box.
        conf (float): Confidence score of the detection.
        cls (int): Class ID of the detected object.
    """
    def __init__(self, xmin, ymin, xmax, ymax, conf=1.0, cls=0):
        self.xmin = int(xmin)
        self.ymin = int(ymin)
        self.xmax = int(xmax)
        self.ymax = int(ymax)
        self.conf = conf
        self.cls = cls

    def width(self):
        """Returns the width of the bounding box."""
        return self.xmax - self.xmin

    def height(self):
        """Returns the height of the bounding box."""
        return self.ymax - self.ymin

    def center(self):
        """Returns the (x,y) center coordinates of the bounding box."""
        return (self.xmin + self.xmax) // 2, (self.ymin + self.ymax) // 2

    def tip(self, prev_center):
        """
        Calculates a direction-aware 'tip' point of the bounding box.

        Args:
            prev_center (tuple or None): Previous tip coordinates to infer direction.

        Returns:
            (int, int): Coordinates of the tip point, oriented based on movement.
        """
        cx, cy = self.center()
        if not prev_center:
            return cx, cy
        dx = cx - prev_center[0]
        dy = cy - prev_center[1]
        if abs(dx) > abs(dy):
            # Horizontal movement dominant
            return (self.xmax if dx >= 0 else self.xmin, cy)
        else:
            # Vertical movement dominant
            return (cx, self.ymax if dy >= 0 else self.ymin)

    def to_list(self):
        """
        Converts the box attributes to a list format.

        Returns:
            list: [xmin, ymin, xmax, ymax, conf, cls]
        """
        return [self.xmin, self.ymin, self.xmax, self.ymax, self.conf, self.cls]

    def move_to_center(self, new_cx, new_cy):
        """
        Moves the box to a new center coordinate, keeping the same width and height.

        Args:
            new_cx (int): New center x-coordinate.
            new_cy (int): New center y-coordinate.
        """
        w = self.width()
        h = self.height()
        self.xmin = new_cx - w // 2
        self.ymin = new_cy - h // 2
        self.xmax = new_cx + w // 2
        self.ymax = new_cy + h // 2
        
    def update_corners(self, xmin, ymin, xmax, ymax):
        self.xmin, self.xmax = sorted([int(xmin), int(xmax)])
        self.ymin, self.ymax = sorted([int(ymin), int(ymax)])

class SmashSpeed:
    """
    Main class to perform YOLOv5-based video object detection and speed calculation.

    Attributes:
        video_path (Path): Path to the input video file.
        weights_path (Path): Path to the YOLOv5 model weights.
        ref_len (float): Real-world length of reference distance in meters.
        fps_override (float or None): Optional FPS override.
        model: Loaded YOLOv5 model.
        width, height, fps (int, int, float): Video metadata.
        scale_factor (float or None): Pixels-to-meters scale factor.
        prev_center, prev_timestamp: Track last tip position and time for speed calc.
        all_frames (list): List of all video frames as numpy arrays (BGR).
        all_boxes (list): List of lists of Box objects detected per frame.
        all_timestamps (list): List of timestamps per frame.
    """

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
        """Loads the YOLOv5 custom model with provided weights."""
        return torch.hub.load('ultralytics/yolov5', 'custom', path=str(self.weights_path))

    def get_video_info(self):
        """
        Uses ffmpeg to probe video and extract width, height, and FPS.

        Returns:
            tuple: (width, height, fps)
        """
        probe = ffmpeg.probe(str(self.video_path))
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width = int(video_info['width'])
        height = int(video_info['height'])
        fps = eval(video_info['r_frame_rate'])
        return width, height, fps

    def read_frames(self):
        """
        Generator yielding frames as RGB numpy arrays and their timestamps.

        Yields:
            tuple: (frame (H,W,3), timestamp in seconds)
        """
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
        """
        Displays a frame for the user to manually select two points representing
        a known real-world distance, to compute pixel-to-meter scale.
        """
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
        """
        Saves the pixel-to-meter scale factor to a text file.

        Args:
            pixel_length (float): Length between reference points in pixels.
        """
        output_name = self.video_path.stem
        save_path = RUNS_DIR / f"{output_name}_scale_info.txt"
        with open(save_path, "w") as f:
            f.write(f"Pixel length: {pixel_length:.2f}\n")
            f.write(f"Scale factor (m/pixel): {self.scale_factor:.6f}\n")
        print(f"📁 Saved scale info to: {save_path}")

    def draw_boxes_on_frame(self, frame, boxes, timestamp, draw_speed=True):
        """
        Draws bounding boxes, tip markers, and speed information on a video frame.

        Args:
            frame (np.ndarray): The frame to draw on (BGR).
            boxes (list of Box): List of Box objects to draw.
            timestamp (float): Timestamp of the current frame in seconds.
            draw_speed (bool): Whether to calculate and display speed.

        Returns:
            tuple: (frame with drawings, last tip center coordinates, speed text)
        """
        speed_text = ""
        current_center = None

        for box in boxes:
            if not isinstance(box, Box):
                box = Box(*box)

            tip_x, tip_y = box.tip(self.prev_center)
            current_center = (tip_x, tip_y)

            # Draw bounding box (red) and tip marker (green)
            cv2.rectangle(frame, (box.xmin, box.ymin), (box.xmax, box.ymax), (0, 0, 255), 2)
            cv2.circle(frame, current_center, 4, (0, 255, 0), -1)

            # Calculate and draw speed if available
            if self.prev_center and self.prev_timestamp is not None and draw_speed:
                dx = tip_x - self.prev_center[0]
                dy = tip_y - self.prev_center[1]
                dist = sqrt(dx ** 2 + dy ** 2)
                dt = max(timestamp - self.prev_timestamp, 1e-6)
                px_per_sec = dist / dt
                speed_text = f"{px_per_sec:.1f} px/s"
                if self.scale_factor:
                    kmph = px_per_sec * self.scale_factor * 3.6
                    speed_text += f" | {kmph:.1f} km/h"
                cv2.putText(frame, speed_text, (box.xmin, box.ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw timestamp always
        cv2.putText(frame, f"Time: {timestamp:.3f}s", (10, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return frame, current_center, speed_text

    def run_inference(self):
        """
        Runs the object detection inference on each video frame, storing
        frames, boxes, and timestamps, and showing the live output.
        """
        self.all_frames.clear()
        self.all_boxes.clear()
        self.all_timestamps.clear()

        for frame, timestamp in self.read_frames():
            results = self.model(frame)
            all_boxes_raw = results.xyxy[0].cpu().numpy()
            all_boxes = [Box(*b[:6]) for b in all_boxes_raw]  # Convert to Box objects
            best_box = [max(all_boxes, key=lambda b: b.conf)] if all_boxes else []

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

        # Show last frame with prompt to review or save
        prompt_frame = self.all_frames[-1].copy()
        cv2.putText(prompt_frame, "Press 'r' to review or 'q' to save and exit",
                    (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
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
        

    def review_mode(self):
        """
        Interactive review mode allowing user to move, resize, add, and delete bounding boxes.

        Controls:
            - Click + drag inside box: Move box.
            - Click + drag near corner: Resize box.
            - 'a': Previous frame.
            - 'd': Next frame.
            - 'n': Add new box at center.
            - 'x': Delete current box.
            - 'q': Finish review and recalculate speeds.
        """
        current_frame_idx = 0
        dragging = False
        resizing = False
        drag_offset = (0, 0)
        resize_corner = None  # Which corner is being resized
        corner_radius = 10

        def point_near(px, py, cx, cy, radius=corner_radius):
            return abs(px - cx) <= radius and abs(py - cy) <= radius

        def mouse_callback(event, x, y, flags, param):
            nonlocal dragging, resizing, drag_offset, resize_corner
            boxes = self.all_boxes[current_frame_idx]
            if not boxes:
                return

            box = boxes[0]

            # Convert to Box if needed
            if not isinstance(box, Box):
                box = Box(*box)
                self.all_boxes[current_frame_idx][0] = box

            xmin, ymin, xmax, ymax = box.xmin, box.ymin, box.xmax, box.ymax
            cx, cy = box.center()

            # Corner coordinates
            corners = {
                "tl": (xmin, ymin),
                "tr": (xmax, ymin),
                "bl": (xmin, ymax),
                "br": (xmax, ymax),
            }

            if event == cv2.EVENT_LBUTTONDOWN:
                # Check if near corner first - start resizing
                for corner_name, (cxr, cyr) in corners.items():
                    if point_near(x, y, cxr, cyr):
                        resizing = True
                        resize_corner = corner_name
                        break
                else:
                    # Not resizing? Check if inside box for dragging
                    if xmin <= x <= xmax and ymin <= y <= ymax:
                        dragging = True
                        drag_offset = (x - cx, y - cy)

            elif event == cv2.EVENT_MOUSEMOVE:
                if dragging:
                    new_cx = x - drag_offset[0]
                    new_cy = y - drag_offset[1]
                    box.move_to_center(new_cx, new_cy)
                elif resizing and resize_corner:
                    if resize_corner == "tl":
                        box.update_corners(x, y, box.xmax, box.ymax)
                    elif resize_corner == "tr":
                        box.update_corners(box.xmin, y, x, box.ymax)
                    elif resize_corner == "bl":
                        box.update_corners(x, box.ymin, box.xmax, y)
                    elif resize_corner == "br":
                        box.update_corners(box.xmin, box.ymin, x, y)

            elif event == cv2.EVENT_LBUTTONUP:
                dragging = False
                resizing = False
                resize_corner = None

        cv2.namedWindow("Review Mode")
        cv2.setMouseCallback("Review Mode", mouse_callback)

        while True:
            frame = self.all_frames[current_frame_idx].copy()
            boxes = self.all_boxes[current_frame_idx]

            frame, _, _ = self.draw_boxes_on_frame(frame, boxes, self.all_timestamps[current_frame_idx], draw_speed=False)

            # Frame index and instructions
            cv2.putText(frame, f"Review Mode - Frame {current_frame_idx+1}/{len(self.all_frames)}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            instructions = [
                "'a': Prev frame",
                "'d': Next frame",
                "Click + drag inside box: Move box",
                "Click + drag corner: Resize box",
                "'n': New box",
                "'x': Delete box",
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
                new_box = Box(cx - box_w // 2, cy - box_h // 2, cx + box_w // 2, cy + box_h // 2, 1.0, 0)
                self.all_boxes[current_frame_idx] = [new_box]
            elif key == ord('x'):
                self.all_boxes[current_frame_idx] = []

        cv2.destroyAllWindows()
        self.save_video()

    def save_video(self, output_name_suffix="reviewed"):
        """
        Draws speed annotations and saves the final output video.

        Args:
            output_name_suffix (str): Suffix for output video filename.
        """
        print("💾 Exporting video with drawn boxes and speed...")

        self.prev_center = None
        self.prev_timestamp = None

        output_name = self.video_path.stem
        out_path = RUNS_DIR / f"{output_name_suffix}_{output_name}.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height))

        cv2.namedWindow("Final Output", cv2.WINDOW_NORMAL)

        for idx, frame in enumerate(self.all_frames):
            boxes = self.all_boxes[idx]
            timestamp = self.all_timestamps[idx]

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
        cv2.destroyAllWindows()
        print(f"✅ Final video saved: {out_path}")


def main():
    """
    Entry point of the program. Parses command line arguments and runs SmashSpeed.
    """
    parser = argparse.ArgumentParser(description="YOLOv5 video analysis with speed overlay")
    parser.add_argument('--fps', type=float, default=None, help='Override FPS manually')
    parser.add_argument('--ref_len', type=float, default=3.91, help='Real-world reference length in meters')
    args = parser.parse_args()

    smash = SmashSpeed(INPUT_VIDEO_PATH, WEIGHTS_PATH, ref_len=args.ref_len, fps_override=args.fps)
    smash.draw_reference_line()
    smash.run_inference()

if __name__ == "__main__":
    main()