# 🏸 SmashSpeed: Shuttlecock Speed Detection with YOLOv5

**SmashSpeed** is a computer vision project that estimates the speed of a badminton shuttlecock by processing video footage and applying object detection models, primarily using YOLOv5.

---

## 🧪 Overview & Pipeline

1. **Video Recording**  
   Capture badminton rallies using a fixed, side-view camera angle aligned with the court for consistent perspective.

2. **Frame Extraction**  
   Extract unique frames from videos using custom scripts (`extract_frames.py`) with perceptual hashing to avoid duplicates.

3. **Annotation**  
   Label shuttlecock positions in frames using [Roboflow](https://roboflow.com), exporting annotations in YOLOv5-compatible format while preserving folder structure.

4. **Dataset Preparation**  
   Organize the dataset into `train/`, `valid/`, and `test/` subsets for balanced model training and evaluation.

5. **Model Training**  
   Train YOLOv5 models (starting with pretrained weights such as `yolov5s.pt`) using the prepared datasets.

6. **Evaluation & Improvement**  
   Track model performance with metrics like `mAP@0.5` and refine the pipeline for improved accuracy.

7. **Speed Calculation (Planned)**  
   Develop a post-processing step to calculate shuttlecock speed by analyzing detected positions frame-by-frame, converting pixel displacement into real-world speed using court dimensions as reference.