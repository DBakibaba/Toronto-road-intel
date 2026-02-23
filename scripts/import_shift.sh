#!/bin/bash
# ============================================================
# import_shift.sh
# Copies one day's dashcam clips from SD card to project
# Usage: bash scripts/import_shift.sh 20260223
# ============================================================

SD_CARD="/d/Normal/Front"
PROJECT_DATA="data/raw_video"
SAMPLE_DATA="data/sample"

DATE=$1

if [ -z "$DATE" ]; then
    echo "❌ Error: No date provided."
    echo "Usage: bash scripts/import_shift.sh YYYYMMDD"
    exit 1
fi

if [ ! -d "$SD_CARD" ]; then
    echo "❌ SD card not found at $SD_CARD"
    echo "Make sure your SD card is inserted."
    exit 1
fi

DEST="$PROJECT_DATA/$DATE"
mkdir -p "$DEST"

COUNT=$(ls "$SD_CARD"/NO"$DATE"*.MP4 2>/dev/null | wc -l)

if [ "$COUNT" -eq 0 ]; then
    echo "❌ No clips found for date: $DATE"
    exit 1
fi

echo "📁 Found $COUNT clips for $DATE"
echo "📋 Copying to $DEST ..."

cp "$SD_CARD"/NO"$DATE"*.MP4 "$DEST"/

echo "✅ $COUNT clips copied to $DEST"
echo ""
echo "📋 Copying 3 sample clips for testing..."
ls "$SD_CARD"/NO"$DATE"*.MP4 | head -3 | xargs -I{} cp {} "$SAMPLE_DATA"/
echo "✅ Sample clips ready in $SAMPLE_DATA"
echo ""
echo "==============================="
echo "  SHIFT IMPORT COMPLETE"
echo "  Date:   $DATE"
echo "  Clips:  $COUNT"
echo "==============================="