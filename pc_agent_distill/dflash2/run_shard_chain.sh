#!/bin/bash
# DFlash2 GUI-Owl-32B shard-chain runner: train parts one by one, warm-starting
# each shard from the previous shard's latest checkpoint.
#
# Usage:
#   bash run_shard_chain.sh <start_ckpt_or_NONE> <shard1> [shard2 ...]
#   PARTS_DIR=parts_pass2 TAG=pass2- bash run_shard_chain.sh <ckpt> part-00 ...
#
# Env knobs:
#   PARTS_DIR  dir under $D holding <shard>.jsonl (default: parts)
#   TAG        prefix for run_id/output/state dirs (default: "")
#
# Each shard: precache (idempotent) -> fresh state dirs -> specforge train.
set -uo pipefail

D=/mnt/lisiqi23/internship/zhangyikang/uiagent/pc_agent_distill/dflash2
TARGET=/mnt/lisiqi23/internship/zhangyikang/uiagent/GUI-Owl-1.5-32B-Instruct/
export SGLANG_MM_AVOID_RETOKENIZE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

PARTS_DIR=${PARTS_DIR:-parts}
TAG=${TAG:-}

WARM_START="$1"; shift

latest_ckpt() {
    # newest step-* dir under a run output dir; sort by the numeric suffix of
    # the BASENAME (paths contain many 'p' characters, so field-based sort on
    # the full path is broken)
    ls -d "$1"/*step*/ 2>/dev/null \
        | awk -F'step' '{n=$NF; sub("/","",n); print n+0, $0}' \
        | sort -n | tail -1 | cut -d' ' -f2-
}

for SHARD in "$@"; do
    echo "================ ${TAG}${SHARD} ================"

    echo "[1/4] precache $SHARD (from $PARTS_DIR)"
    python3 "$D/precache.py" --data "$D/$PARTS_DIR/$SHARD.jsonl" \
        --target-model "$TARGET" --cache-dir "$D/cache" \
        --max-length 8192 --num-proc 128 || { echo "PRECACHE FAILED $SHARD"; exit 1; }

    echo "[2/4] make yaml"
    Y="$D/train-${TAG}${SHARD}.yaml"
    if [ "$WARM_START" = "NONE" ]; then
        python3 "$D/make_shard_yaml.py" --shard "$SHARD" --parts-dir "$PARTS_DIR" --tag "$TAG" --out "$Y"
    else
        python3 "$D/make_shard_yaml.py" --shard "$SHARD" --parts-dir "$PARTS_DIR" --tag "$TAG" --warm-start "$WARM_START" --out "$Y"
    fi

    echo "[3/4] clean state & train"
    # guard: /dev/shm symlink farm may get wiped by host cleaners; rebuild cheap
    if ! ls /dev/shm/pc_agent_images/ubuntu/ >/dev/null 2>&1 || \
       [ -z "$(ls /dev/shm/pc_agent_images/ubuntu/ 2>/dev/null | head -1)" ]; then
        echo "symlink farm missing, rebuilding..."
        python3 "$D/make_symlink_farm.py" || { echo "SYMLINK FARM FAILED"; exit 3; }
    fi
    rm -rf "$D/train-${TAG}${SHARD}" "/root/specforge_state/train-${TAG}${SHARD}"
    mkdir -p "$D/train-${TAG}${SHARD}"
    specforge train --config "$Y" 2>&1 | tee -a "$D/train-${TAG}${SHARD}/train.log"

    echo "[4/4] resolve next warm start"
    CKPT=$(latest_ckpt "$D/train-${TAG}${SHARD}")
    if [ -z "$CKPT" ]; then
        echo "NO CHECKPOINT produced for ${TAG}${SHARD}, aborting chain"
        exit 2
    fi
    WARM_START="${CKPT%/}"
    echo "${TAG}${SHARD} done, next warm start: $WARM_START"
done

echo "CHAIN COMPLETE. Final ckpt: $WARM_START"
