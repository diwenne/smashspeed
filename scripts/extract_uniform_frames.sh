#!/bin/bash

```
Extracts a specified number of evenly spaced frames from a video (hardcoded), skipping unnecessary frames to save time and space.
```

# =======================
# CONFIGURATION SECTION
# =======================
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VIDEO_PATH="$BASE_DIR/../raw_videos/raw_cosports_20250616_02.mov"
OUTPUT_DIR="$BASE_DIR/../frames/raw_cosports_20250616_02"
NUM_FRAMES=240  # Number of frames to extract

# =======================
# SCRIPT STARTS HERE
# =======================

mkdir -p "$OUTPUT_DIR"

# Get video duration in seconds
DURATION=$(ffprobe -v error -select_streams v:0 -show_entries format=duration \
  -of default=noprint_wrappers=1:nokey=1 "$VIDEO_PATH")

if [ -z "$DURATION" ]; then
  echo "❌ Could not determine video duration."
  exit 1
fi

echo "Video duration: $DURATION seconds"

# Calculate the interval between frames
INTERVAL=$(awk "BEGIN {print $DURATION / $NUM_FRAMES}")

echo "Extracting 1 frame every $INTERVAL seconds"

# Extract frames at calculated timestamps
for i in $(seq 0 $(($NUM_FRAMES - 1))); do
  TIMESTAMP=$(awk "BEGIN {printf \"%.2f\", $i * $INTERVAL}")
  OUTPUT_FILE=$(printf "$OUTPUT_DIR/frame_%04d.jpg" "$i")
  ffmpeg -loglevel error -ss "$TIMESTAMP" -i "$VIDEO_PATH" -frames:v 1 "$OUTPUT_FILE"
  echo "🖼️  Saved frame at $TIMESTAMP sec -> $(basename "$OUTPUT_FILE")"
done

echo "✅ Done! Extracted $NUM_FRAMES frames into: $OUTPUT_DIR"