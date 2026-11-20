#!/usr/bin/env bash
# Run COLMAP (SIFT features, sequential matching, mapper) on every clip, on the GPU.
# Written for COLMAP >= 4.x (built with build_colmap.sh, or the colmap/colmap Docker image).
#
#   bash colmap.sh bruteforce    # -> <clip>/colmap_bruteforce_out/  guided matching ON  (default, used for both plantings)
#   bash colmap.sh lightglue     # -> <clip>/colmap_lightglue_out/   guided matching OFF (LightGlue aborts with it)
#   FORCE=1 bash colmap.sh ...   # redo clips that already have a model
#   ROUND=1st bash colmap.sh     # data in /workspace/videos/1st
#
# Camera model is SIMPLE_RADIAL: the DJI Osmo Action 5 Pro dewarps in camera, but some distortion is left.
set -uo pipefail

MODE="${1:-bruteforce}"
case "$MODE" in
    lightglue)   MATCH_TYPE="SIFT_LIGHTGLUE";  OUTNAME="colmap_lightglue_out" ;;
    bruteforce)  MATCH_TYPE="SIFT_BRUTEFORCE"; OUTNAME="colmap_bruteforce_out" ;;
    *) echo "Unknown mode '$MODE', use 'lightglue' or 'bruteforce'"; exit 1 ;;
esac
echo ">>> matcher = $MATCH_TYPE   ->   output folder = $OUTNAME   (FORCE=${FORCE:-0})"

# ---- settings ----
ROUND="${ROUND:-2nd}"
DATA="${DATA:-/workspace/videos/$ROUND}"
CAMERA_MODEL="SIMPLE_RADIAL"
MATCHER="sequential_matcher"
# ------------------

command -v colmap >/dev/null || { echo "colmap not found, wrong image?"; exit 1; }

# Keep the exact build, for the parameters table in the thesis.
VERSION_BANNER=$(colmap -h 2>&1 | head -1)
echo ">>> $VERSION_BANNER"

# Stop early if this colmap is older than the 4.x option names used below
# (e.g. an old apt build ending up first on PATH).
if ! colmap "$MATCHER" --help 2>&1 | grep -q -- '--FeatureMatching.type'; then
    echo "ERROR: this COLMAP has no --FeatureMatching.type, the build is too old for this script."
    exit 1
fi

shopt -s nullglob

# Count the clips, so an empty glob says so instead of quietly printing DONE.
found=0
failed=()

for imgdir in "$DATA"/*/*/images; do
    clipdir=$(dirname "$imgdir")                    # .../<light>/<date>
    date=$(basename "$clipdir")
    light=$(basename "$(dirname "$clipdir")")
    out="$clipdir/$OUTNAME"
    db="$out/database.db"
    sparse="$out/sparse"
    found=$((found + 1))

    echo "############  $light/$date  [$MATCH_TYPE]  ############"

    # images/ is uploaded, not made here, so check it instead of creating it.
    nimg=$(find "$imgdir" -maxdepth 1 -type f \
                \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) \
                ! -name '._*' | wc -l)
    if [ "$nimg" -eq 0 ]; then
        echo "  no images in $imgdir, skipping"
        failed+=("$light/$date (no images)")
        continue
    fi
    echo "  $nimg images"

    # COLMAP tries to read EVERY file in image_path, so stray files (.DS_Store, a .json ...) give read errors.
    nstray=$(find "$imgdir" -maxdepth 1 -type f \
                  ! -iname '*.png' ! -iname '*.jpg' ! -iname '*.jpeg' | wc -l)
    [ "$nstray" -gt 0 ] && echo "  note: $nstray non-image file(s) in images/, COLMAP will log read errors for them"

    mkdir -p "$sparse"

    if [ "${FORCE:-0}" != "1" ] && [ -f "$sparse/0/points3D.bin" ]; then
        echo "  already done ($OUTNAME/sparse/0 exists), skipping. FORCE=1 to redo."
        continue
    fi

    # Keep the COLMAP build next to each reconstruction.
    echo "$VERSION_BANNER" > "$out/colmap_version.txt"

    rm -f "$db"
    colmap feature_extractor \
        --database_path "$db" --image_path "$imgdir" \
        --ImageReader.single_camera 1 --ImageReader.camera_model "$CAMERA_MODEL" \
        --FeatureExtraction.type SIFT \
        --FeatureExtraction.use_gpu 1 \
        || { echo "  feature extraction failed"; failed+=("$light/$date (extractor)"); continue; }

    # guided matching helps brute force but makes LightGlue abort, so it depends on the mode
    if [ "$MODE" = "lightglue" ]; then
        colmap "$MATCHER" \
            --database_path "$db" \
            --FeatureMatching.type "$MATCH_TYPE" \
            --FeatureMatching.use_gpu 1 \
            || { echo "  matching failed"; failed+=("$light/$date (matcher)"); continue; }
    else
        colmap "$MATCHER" \
            --database_path "$db" \
            --FeatureMatching.type "$MATCH_TYPE" \
            --FeatureMatching.guided_matching 1 \
            --FeatureMatching.use_gpu 1 \
            || { echo "  matching failed"; failed+=("$light/$date (matcher)"); continue; }
    fi

    colmap mapper \
        --database_path "$db" --image_path "$imgdir" --output_path "$sparse" \
        || { echo "  mapper failed"; failed+=("$light/$date (mapper)"); continue; }

    if [ -d "$sparse/0" ]; then
        echo "-----  $light/$date  [$MATCH_TYPE]  results  -----"
        colmap model_analyzer --path "$sparse/0" 2>&1 | \
            grep -Ei "Registered images|Points:|Mean track length|reprojection error"
        n_models=$(find "$sparse" -mindepth 1 -maxdepth 1 -type d | wc -l)
        [ "$n_models" -gt 1 ] && echo "  note: $n_models sub-models, check with colmap_fix_sparse.py"
    else
        echo "  no model made"
        failed+=("$light/$date (no model)")
    fi
done

if [ "$found" -eq 0 ]; then
    echo ""
    echo "WARNING: nothing matched $DATA/*/*/images"
    echo "         expected: $DATA/<light>/<date>/images/frame_00001.png"
    echo "         look for zip files that were not unpacked, or one folder level too many."
    exit 1
fi

echo ""
if [ ${#failed[@]} -gt 0 ]; then
    echo "FINISHED WITH ${#failed[@]} PROBLEM(S) ($MATCH_TYPE):"
    printf '  %s\n' "${failed[@]}"
    exit 1
fi
echo "DONE ($MATCH_TYPE), no problems. Copy the results back BEFORE terminating the pod."
