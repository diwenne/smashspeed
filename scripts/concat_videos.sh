#!/bin/bash

: '
Helper script that concatenates clips or videos to make frame extraction more meaningful and efficient.
'

# =======================
# CONFIGURATION SECTION
# =======================
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

VIDEOS=(
  "$BASE_DIR/../raw_videos/diwen1.mov"
  "$BASE_DIR/../raw_videos/diwen2.mov"
  "$BASE_DIR/../raw_videos/diwen3.mov"
  "$BASE_DIR/../raw_videos/diwen4.mov"
  "$BASE_DIR/../raw_videos/diwen5.mov"
  "$BASE_DIR/../raw_videos/diwen6.mov"
  "$BASE_DIR/../raw_videos/diwen7.mov"
  "$BASE_DIR/../raw_videos/diwen8.mov"
  "$BASE_DIR/../raw_videos/diwen9.mov"
  "$BASE_DIR/../raw_videos/diwen10.mov"
  "$BASE_DIR/../raw_videos/diwen11.mov"
  "$BASE_DIR/../raw_videos/diwen12.mov"
  "$BASE_DIR/../raw_videos/hao1.mov"
  "$BASE_DIR/../raw_videos/hao2.mov"
  "$BASE_DIR/../raw_videos/hao3.mov"
  "$BASE_DIR/../raw_videos/hao4.mov"
  "$BASE_DIR/../raw_videos/hao5.mov"
  "$BASE_DIR/../raw_videos/hao6.mov"
  "$BASE_DIR/../raw_videos/hao7.mov"
  "$BASE_DIR/../raw_videos/hao8.mov"
  "$BASE_DIR/../raw_videos/hao9.mov"
  "$BASE_DIR/../raw_videos/hao10.mov"
  "$BASE_DIR/../raw_videos/hao11.mov"

)

OUTPUT="$BASE_DIR/../raw_videos/concatclips_cosports_20250618_01.mp4"
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