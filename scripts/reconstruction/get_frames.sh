#!/usr/bin/env bash
# Pull TARGET frames, evenly spaced over the whole clip, out of every video:
#   <DATA>/<light>/<date>/*.mp4  ->  <DATA>/<light>/<date>/images/frame_00001.png ...
# Frames are full resolution, lossless PNG.
#
# Usage (on the Mac, needs ffmpeg and bc):
#   bash scripts/reconstruction/get_frames.sh                 # round 2
#   ROUND=1st bash scripts/reconstruction/get_frames.sh
#   TARGET=400 bash scripts/reconstruction/get_frames.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ROUND="${ROUND:-2nd}"
DATA="${DATA:-$ROOT/data/raw/videos/$ROUND}"
TARGET="${TARGET:-600}"          # frames per clip
shopt -s nullglob

for vid in "$DATA"/*/*/*.mp4; do
    datedir=$(dirname "$vid")
    light=$(basename "$(dirname "$datedir")")
    date=$(basename "$datedir")
    outdir="$datedir/images"
    mkdir -p "$outdir"

    # the clip length decides how many frames per second to take to end up with TARGET frames
    dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$vid")

    # the source frame rate comes as a fraction like "30000/1001"; bc turns it into a decimal
    rate=$(ffprobe -v error -select_streams v:0 \
           -show_entries stream=r_frame_rate -of csv=p=0 "$vid")
    srcfps=$(echo "scale=6; $rate" | bc -l)

    # are there enough frames in the clip? if not, some frames would be taken twice
    avail=$(echo "$dur * $srcfps" | bc -l)
    if (( $(echo "$avail < $TARGET" | bc -l) )); then
        echo "  WARNING: $light/$date only has about ${avail%.*} frames (< $TARGET)"
    fi

    # output rate = frames wanted / seconds, so the frames are spread evenly over the clip
    outfps=$(echo "scale=6; $TARGET / $dur" | bc -l)

    # -vsync 0   : don't let ffmpeg add or drop frames on its own, only the fps filter decides
    # -pred mixed: PNG predictor that gives smaller files; pixels stay exactly the same (lossless)
    ffmpeg -v error -y -i "$vid" \
        -vf "fps=$outfps" \
        -vsync 0 \
        -pred mixed \
        "$outdir/frame_%05d.png"

    echo "  $light/$date: ${dur}s, ${srcfps%.*} fps -> $(ls "$outdir"/*.png | wc -l) frames"
done
