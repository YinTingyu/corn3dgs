#!/usr/bin/env bash
# Train splatfacto-big on every clip, on the nerfstudio Docker image (or a pod set up with build_nerfstudio.sh).
# Each clip: colmap_bruteforce_out/sparse/0 + images/ -> ns-process-data -> ns-train -> ns-export.
# Can be re-run: clips whose splat.ply is already exported are skipped (unless FORCE=1).
#
#   bash nerfstudio_run.sh
#   ROUND=1st bash nerfstudio_run.sh
set -uo pipefail   # NOT -e: one clip failing must not stop the whole batch

# torch.compile in splatfacto hits an inductor ImportError on this image, so run eager
export TORCH_COMPILE_DISABLE=1

ROUND="${ROUND:-2nd}"
DATA="${DATA:-/workspace/videos/$ROUND}"
EXPORTS="${EXPORTS:-/workspace/exports/$ROUND}"
TRAIN_OUT="${TRAIN_OUT:-/workspace/outputs/$ROUND}"
ITERS=50000
mkdir -p "$EXPORTS"

command -v ns-train >/dev/null || { echo "ns-train not found, is this the nerfstudio image?"; exit 1; }

shopt -s nullglob
failed=()
done_n=0

for imgdir in "$DATA"/*/*/images; do
    clipdir=$(dirname "$imgdir")
    date=$(basename "$clipdir")
    light=$(basename "$(dirname "$clipdir")")
    tag="${light}_${date}"

    sparse="$clipdir/colmap_bruteforce_out/sparse/0"
    nerf="$clipdir/nerf"
    export_dir="$EXPORTS/$tag"

    echo ""
    echo "############  $light/$date  ############"

    # --- already exported? (FORCE=1 to redo) ---
    if [ "${FORCE:-0}" != "1" ] && [ -f "$export_dir/splat.ply" ]; then
        echo "  already exported ($export_dir/splat.ply), skipping. FORCE=1 to redo."
        continue
    fi

    # --- the clip must have a COLMAP model ---
    if [ ! -f "$sparse/points3D.bin" ]; then
        echo "  no sparse model at $sparse, skipping"
        failed+=("$tag (no COLMAP model)")
        continue
    fi

    # --- [1/3] COLMAP model -> nerfstudio format ---
    if [ ! -f "$nerf/transforms.json" ]; then
        echo "  [1/3] ns-process-data (skip-colmap)"
        ns-process-data images \
            --data "$imgdir" \
            --output-dir "$nerf" \
            --skip-colmap \
            --colmap-model-path "$sparse" \
            || { echo "  ns-process-data failed, skipping clip"; failed+=("$tag (ns-process-data)"); continue; }
    else
        echo "  [1/3] $nerf/transforms.json exists, skipping ns-process-data"
    fi

    # --- [2/3] train splatfacto-big at full resolution ---
    echo "  [2/3] ns-train splatfacto-big ($ITERS iterations, full resolution)"
    ns-train splatfacto-big \
        --data "$nerf" \
        --max-num-iterations "$ITERS" \
        --pipeline.model.num-downscales 0 \
        --optimizers.means.scheduler.max-steps "$ITERS" \
        --pipeline.model.stop-split-at 25000 \
        --vis viewer \
        --viewer.quit-on-train-completion True \
        --output-dir "$TRAIN_OUT/$tag" \
        || { echo "  ns-train failed, skipping clip"; failed+=("$tag (ns-train)"); continue; }

    # --- [3/3] export the Gaussian splat ---
    echo "  [3/3] ns-export"
    config=$(find "$TRAIN_OUT/$tag" -name config.yml -path '*splatfacto*' | sort | tail -1)
    if [ -z "$config" ]; then
        echo "  no config.yml under $TRAIN_OUT/$tag, export skipped"
        failed+=("$tag (no config.yml)")
        continue
    fi
    if ns-export gaussian-splat --load-config "$config" --output-dir "$export_dir" \
            && [ -f "$export_dir/splat.ply" ]; then
        echo "  exported -> $export_dir"
        done_n=$((done_n + 1))
    else
        echo "  ns-export failed for $tag"
        failed+=("$tag (ns-export)")
    fi
done

echo ""
echo "=== $done_n clip(s) trained and exported this run ==="
if [ ${#failed[@]} -gt 0 ]; then
    echo "${#failed[@]} PROBLEM(S):"
    printf '  %s\n' "${failed[@]}"
fi
echo "Exports are in $EXPORTS/<light>_<date>/. Pack them up and pull them back, e.g."
echo "  cd $(dirname "$EXPORTS") && tar -czf splats_$ROUND.tar.gz $(basename "$EXPORTS")"
echo "  then rsync / runpodctl it to the Mac, and STOP the pod."
[ ${#failed[@]} -eq 0 ]
