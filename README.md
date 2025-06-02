# 🏸 SmashSpeed: Shuttlecock Speed Detection using YOLOv5

**SmashSpeed** is a computer vision project that estimates the speed of a badminton shuttlecock using frame extraction and object detection with YOLOv5.

## 🧪 Pipeline

1. **Record Videos** of rallies using a fixed side-view angle.
2. **Extract Frames** with `extract_frames.py`.
3. **Annotate** shuttlecock positions using [Roboflow](https://roboflow.com).
4. **Export Annotations** in YOLOv5 format (same folder structure).
5. **Split Dataset** into `train/`, `valid/`, and `test/`.
6. **Train YOLOv5** on the dataset.
7. *(Planned)* **Calculate shuttlecock speed** using frame-to-frame movement.
