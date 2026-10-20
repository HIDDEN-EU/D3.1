#!/usr/bin/env bash

GPUS=$1
CONFIG=$2
WEIGHT=$3
TMPDIR=${4:-/tmp/opus_val_${USER}_$$}

python3 -m torch.distributed.run --nproc_per_node "$GPUS" val.py \
    --config "$CONFIG" --weights "$WEIGHT" --tmpdir "$TMPDIR" "${@:5}"
