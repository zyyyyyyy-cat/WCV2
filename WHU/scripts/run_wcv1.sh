#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-train1000}"
DATA_ROOT="${DATA_ROOT:-/project/code/data}"
PRETRAIN="${PRETRAIN:-$PWD/jx_vit_base_p16_224-80ecf9dd.pth}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
CONFIG=configs/WCV1.yml
DATASET=WHU-MARS
RUN=logs/WCV1_1000_DDP
case "$MODE" in
  train1000|test1000|testgd) ;;
  train2337|test2337) DATASET=WHU-MARS-2337; RUN=logs/WCV1_2337_DDP ;;
  baseline1000) CONFIG=configs/UAD.yml; RUN=logs/UAD_1000_control ;;
  *) echo 'Usage: bash scripts/run_wcv1.sh {train1000|test1000|testgd|train2337|test2337|baseline1000}' >&2; exit 2 ;;
esac
[[ -d "$DATA_ROOT/$DATASET" ]] || { echo "Missing dataset: $DATA_ROOT/$DATASET" >&2; exit 2; }
[[ -f "$PRETRAIN" ]] || { echo "Missing ImageNet weights: $PRETRAIN" >&2; exit 2; }
if [[ "$MODE" == train* || "$MODE" == baseline1000 ]]; then
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 train.py \
    --config_file "$CONFIG" MODEL.DIST_TRAIN True \
    MODEL.PRETRAIN_PATH "$PRETRAIN" DATASETS.ROOT_DIR "$DATA_ROOT" \
    DATASETS.DATASET_DIR "$DATASET" DATASETS.FILTER_GD False \
    SOLVER.IMS_PER_BATCH 256 SOLVER.BASE_LR 0.032 SOLVER.SEED 1234 \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 10 OUTPUT_DIR "$RUN"
else
  GD=False; OUT="$RUN/test"
  if [[ "$MODE" == testgd ]]; then GD=True; OUT="$RUN/test_GD"; fi
  WEIGHT="${WEIGHT:-$RUN/transformer_120.pth}"
  [[ -f "$WEIGHT" ]] || { echo "Missing checkpoint: $WEIGHT" >&2; exit 2; }
  python test.py --config_file "$CONFIG" MODEL.DIST_TRAIN False \
    MODEL.PRETRAIN_PATH "$PRETRAIN" DATASETS.ROOT_DIR "$DATA_ROOT" \
    DATASETS.DATASET_DIR "$DATASET" DATASETS.FILTER_GD "$GD" \
    TEST.IMS_PER_BATCH 512 TEST.WEIGHT "$WEIGHT" TEST.NECK_FEAT before \
    TEST.FEAT_NORM yes TEST.RE_RANKING False TEST.TOP_K_EVAL 0 \
    OUTPUT_DIR "$OUT"
fi
