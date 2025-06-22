#!/bin/bash

# Extracts every single frame from the video (hardcoded). 

# =======================
# CONFIGURATION SECTION
# =======================
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"  # Script directory as base
VIDEO_PATH="$BASE_DIR/../raw_videos/concatclips_cosports_20250618_01.mp4"    # relative to base dir
OUTPUT_DIR="$BASE_DIR/../frames/concatclips_cosports_20250618_01"            # relative to base dir

# =======================
# SCRIPT STARTS HERE
# =======================

mkdir -p "$OUTPUT_DIR"

ffmpeg -i "$VIDEO_PATH" -vsync vfr -qscale:v 1 -frame_pts true "$OUTPUT_DIR/frame_%04d.jpg"

echo "✅ All frames extracted to: $OUTPUT_DIR"