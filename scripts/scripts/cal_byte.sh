#!/usr/bin/env bash
set -euo pipefail

RUN_ID=${RUN_ID:-earc_calib_byte}
RECURRENCE_STEPS=${RECURRENCE_STEPS:-8}

BYTE_LEVEL=1 \
VOCAB_SIZE=256 \
DATA_PATH=./data/datasets/fineweb_bytes \
MAX_WALLCLOCK_SECONDS=300 \
VAL_LOSS_EVERY=200 \
GRAD_CLIP_NORM=1.0 \
RUN_ID="$RUN_ID" \
RECURRENCE_STEPS="$RECURRENCE_STEPS" \
python3 records/track_10min_16mb/2026-03-19_EARC/train_gpt.py

