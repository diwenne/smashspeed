import torch
import os
from pathlib import Path
import cv2

#cd inference

# Set paths
weights_path = '../yolov5/runs/train/smashspeed_v3/weights/best.pt'  # change to your path
source = '../frames/clip_cosports_20250608/clip_cosports_20250608_01/clip_cosports_20250608_01_frame0004.jpg'                      # folder or single image

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)

results = model(source)

boxes = results.xyxy[0].int().tolist()

import cv2

def draw_bbox(image_path, boxes):
    """
    image_path: str, path to the image file
    boxes: list of tuples, each tuple is (xmin, ymin, xmax, ymax)
    """
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found or unable to read.")
        return
    boxes = [box[:4] for box in boxes]
    # Draw each bounding box on the image
    for (xmin, ymin, xmax, ymax) in boxes:
        # Draw rectangle in red color with thickness 2
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    
    # Show image with bounding boxes
    cv2.imshow("Image with Bounding Boxes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
draw_bbox(source,boxes)
