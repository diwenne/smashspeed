#!/bin/bash

: '
Helper script that concatenates clips or videos to make frame extraction more meaningful and efficient.
'

# =======================
# CONFIGURATION SECTION
# =======================
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

VIDEOS=(
  "$BASE_DIR/../raw_videos/raw1.mov"
  "$BASE_DIR/../raw_videos/raw2.mov"
  "$BASE_DIR/../raw_videos/raw3.mov"
)

OUTPUT="$BASE_DIR/../raw_videos/raw_cosports_20250616_02.mp4"
LIST_FILE="$BASE_DIR/videos_to_concat.txt"

# =======================
# SCRIPT STARTS HERE
# =======================

> "$LIST_FILE"  # Empty the file

for vid in "${VIDEOS[@]}"; do
  echo "file '$vid'" >> "$LIST_FILE"
done

ffmpeg -f concat -safe 0 -i "$LIST_FILE" -c copy "$OUTPUT"

if [ $? -eq 0 ]; then
  echo "✅ Concat successful: $OUTPUT"
  echo "🧹 Deleting original input files..."

  SEEN=""

  for vid in "${VIDEOS[@]}"; do
    # Check if this file has already been deleted
    if ! echo "$SEEN" | grep -q "$vid"; then
      rm "$vid"
      echo "🗑️ Deleted $vid"
      SEEN="$SEEN$vid "
    fi
  done
else
  echo "❌ Concat failed. Input videos not deleted."
fi

rm "$LIST_FILE"