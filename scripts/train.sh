#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/mixed_sc_dmd_trd/self_forcing.yaml}"
LOGDIR="${LOGDIR:-logs/$(basename "${CONFIG%.yaml}")}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  train.py \
  --config_path "${CONFIG}" \
  --logdir "${LOGDIR}" \
  --disable-wandb
