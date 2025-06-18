#!/bin/bash

```
Extracts every single frame from the video (hardcoded). 
```

# =======================
# CONFIGURATION SECTION
# =======================
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"  # Script directory as base
VIDEO_PATH="$BASE_DIR/../raw_videos/clip_cosports_20250616_09.mov"    # relative to base dir
OUTPUT_DIR="$BASE_DIR/../frames/clip_cosports_20250616_09"            # relative to base dir

# =======================
# SCRIPT STARTS HERE
# =======================

mkdir -p "$OUTPUT_DIR"

ffmpeg -i "$VIDEO_PATH" -vsync 0 "$OUTPUT_DIR/frame_%04d.jpg"

echo "✅ All frames extracted to: $OUTPUT_DIR"