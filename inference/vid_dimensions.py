import cv2

'''
helper file that gets dimensions of a video
'''

video_path = '../raw_videos/clip_cosports_20250608_01.mov'  # Replace with your path

cap = cv2.VideoCapture(video_path)
if cap.isOpened():
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Frame dimensions: {width}x{height}")
else:
    print("Error: Could not open video.")

cap.release()