# 🏸 SmashSpeed: Shuttlecock Speed Detection using YOLOv5

**SmashSpeed** is a computer vision project that estimates the speed of a badminton shuttlecock by extracting frames from video and applying object detection with YOLOv5.

## 🧪 Pipeline

1. **Record Videos** of rallies using a fixed side-view camera angle.
2. **Extract Frames** using `extract_frames.py`.
3. **Annotate** shuttlecock positions with [Roboflow](https://roboflow.com).
4. **Export Annotations** in YOLOv5 format, keeping the same folder structure.
5. **Split Dataset** into `train/`, `valid/`, and `test/` folders.
6. **Train YOLOv5** on the prepared dataset.
7. *(Planned)* **Calculate shuttlecock speed** based on frame-to-frame movement analysis.