#!/usr/bin/env bash
# Run the built image over the 4 validation subjects exactly the way the
# organizers do: one container per subject, no network, 5-minute timeout,
# only /data/features (ro) and /data/output mounted.
#
#   ./run_val_subjects.sh                       # -> submission_docker/
#   SUBJECTS="sub-004" ./run_val_subjects.sh    # just one
#   VAL_DIR=/path/to/bic-mac-data/val ./run_val_subjects.sh
#
# GPU access. With the NVIDIA Container Toolkit installed (as on the
# organizers' host) the run uses `--gpus all`. Without it, the script falls
# back to passing the device nodes and the driver's user-space libraries in by
# hand, which is enough for CUDA inside the container.
#
#   GPU_MODE=toolkit   force --gpus (what the organizers do)
#   GPU_MODE=manual    force the device/library passthrough
#   GPU_MODE=none      CPU only (slow: ~30 min/subject, but proves the packaging)
#
# The fallback changes nothing about the image - only how the host hands it a
# GPU - so a successful run is still a valid test of the container.

set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-bic-mac-pseudoct:latest}"
VAL_DIR="${VAL_DIR:-$(pwd)/../data/bic-mac-data/val}"
OUT_ROOT="${OUT_ROOT:-$(pwd)/submission_docker}"
SUBJECTS="${SUBJECTS:-sub-004 sub-009 sub-010 sub-018}"
GPUS="${GPUS:-all}"
GPU_INDEX="${GPU_INDEX:-0}"
DRIVER_LIB_DIR="${DRIVER_LIB_DIR:-/usr/lib64}"

# -- pick a GPU mode ---------------------------------------------------------
if [ -z "${GPU_MODE:-}" ]; then
    if docker run --rm --gpus "$GPUS" --entrypoint true "$IMAGE" >/dev/null 2>&1; then
        GPU_MODE=toolkit
    elif [ -e /dev/nvidiactl ] && [ -e "$DRIVER_LIB_DIR/libcuda.so.1" ]; then
        GPU_MODE=manual
        echo "note: no NVIDIA Container Toolkit here - using manual device/library passthrough"
    else
        GPU_MODE=none
        echo "warning: no GPU available to Docker; falling back to CPU (very slow)" >&2
    fi
fi

GPU_ARGS=()
case "$GPU_MODE" in
    toolkit)
        GPU_ARGS=(--gpus "$GPUS")
        ;;
    manual)
        for node in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools \
                    "/dev/nvidia${GPU_INDEX}"; do
            [ -e "$node" ] && GPU_ARGS+=(--device "$node")
        done
        # Resolve each versioned driver library and mount it under the SONAME
        # the loader looks for. The container's CUDA runtime comes from the
        # image; only these driver-side pieces have to come from the host.
        for pair in "libcuda.so.1:libcuda.so.1" \
                    "libnvidia-ml.so.1:libnvidia-ml.so.1" \
                    "libnvidia-ptxjitcompiler.so.1:libnvidia-ptxjitcompiler.so.1" \
                    "libnvidia-nvvm.so.4:libnvidia-nvvm.so.4"; do
            src="$DRIVER_LIB_DIR/${pair%%:*}"
            [ -e "$src" ] || continue
            GPU_ARGS+=(-v "$(readlink -f "$src")":"/nv/${pair##*:}":ro)
        done
        GPU_ARGS+=(-e LD_LIBRARY_PATH=/nv)
        ;;
    none) ;;
    *) echo "unknown GPU_MODE=$GPU_MODE" >&2; exit 2 ;;
esac

mkdir -p "$OUT_ROOT"
echo "image    : $IMAGE"
echo "gpu mode : $GPU_MODE"
echo "subjects : $SUBJECTS"
echo "output   : $OUT_ROOT"

for name in $SUBJECTS; do
    features="$VAL_DIR/$name/features"
    outdir="$OUT_ROOT/$name"
    [ -d "$features" ] || { echo "missing $features" >&2; exit 1; }
    mkdir -p "$outdir"
    echo "=== $name ==="
    start=$(date +%s)
    # `set -e` would abort before the message below, so failures are caught
    # here instead; exit 124 from `timeout` is the 5-minute limit being hit.
    status=0
    timeout 300 docker run --rm \
        --memory 128g \
        --network none \
        "${GPU_ARGS[@]}" \
        -v "$features":/data/features:ro \
        -v "$outdir":/data/output \
        "$IMAGE" || status=$?
    elapsed=$(( $(date +%s) - start ))
    if [ $status -ne 0 ]; then
        [ $status -eq 124 ] && echo "$name TIMED OUT (>300s, the challenge limit)" >&2
        echo "$name FAILED (exit $status after ${elapsed}s)" >&2
        exit $status
    fi
    [ -f "$outdir/ct.nii.gz" ] || { echo "$name produced no ct.nii.gz" >&2; exit 1; }
    echo "$name ok in ${elapsed}s  ($(du -h "$outdir/ct.nii.gz" | cut -f1))"
done

echo
echo "All subjects done -> $OUT_ROOT"
echo "Score them with the challenge repository's submission.py (Codabench)."

