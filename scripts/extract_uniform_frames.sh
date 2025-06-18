#!/bin/bash

# =======================
# CONFIGURATION SECTION
# =======================
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VIDEO_PATH="$BASE_DIR/../raw_videos/clip_cosports_20250616_01.mov"
OUTPUT_DIR="$BASE_DIR/../frames/clip_cosports_20250616_01"
NUM_FRAMES=30  # Number of frames you want to extract uniformly

# =======================
# SCRIPT STARTS HERE
# =======================

mkdir -p "$OUTPUT_DIR"

# Get total frame count of the video
TOTAL_FRAMES=$(ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 "$VIDEO_PATH")

if [ -z "$TOTAL_FRAMES" ]; then
  echo "❌ Could not determine total frames."
  exit 1
fi

echo "Total frames in video: $TOTAL_FRAMES"

# Calculate frame step to extract approx NUM_FRAMES evenly spaced frames
STEP=$((TOTAL_FRAMES / NUM_FRAMES))
if [ "$STEP" -lt 1 ]; then
  STEP=1
fi

echo "Extracting every $STEP frame to get approximately $NUM_FRAMES frames."

# Extract frames using select filter
ffmpeg -i "$VIDEO_PATH" -vf "select='not(mod(n\,$STEP))'" -vsync vfr "$OUTPUT_DIR/frame_%04d.png"

echo "✅ Extracted approximately $NUM_FRAMES frames to: $OUTPUT_DIR"