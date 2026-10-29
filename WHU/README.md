# UAD-SAM-B（本轮实验运行入口）

以下为新方案完整命令。旧UAD命令保留在文末，不能将其 `configs/UAD.yml` 当作新方案配置。
服务器使用已安装的PyTorch1.12.1/CUDA11.3，不安装原requirements.txt。先进入上传后的代码目录；修改下面两个路径为实际数据及原ImageNet预训练文件。

```bash
export DATA_ROOT=/project/code/data
export PRETRAIN=/project/code/WHU_base/WC/jx_vit_base_p16_224-80ecf9dd.pth

# 合成检查；不读取WHU数据，不要给这些检查额外套torchrun。
python tests/test_sam_training.py
CUDA_VISIBLE_DEVICES=0 python tests/test_sam_training.py --cuda
CUDA_VISIBLE_DEVICES=0,1,2,3 python tests/test_sam_training.py --backend nccl --world-size 4
```

## WHU-MARS-1000：四卡完整训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 torchrun \
  --standalone --nproc_per_node=4 train.py \
  --config_file configs/UAD_SAM_B.yml \
  MODEL.DIST_TRAIN True MODEL.PRETRAIN_PATH "$PRETRAIN" \
  DATASETS.ROOT_DIR "$DATA_ROOT" DATASETS.DATASET_DIR WHU-MARS DATASETS.FILTER_GD False \
  SOLVER.IMS_PER_BATCH 256 SOLVER.BASE_LR 0.032 SOLVER.SEED 1234 \
  SOLVER.MAX_EPOCHS 120 WCV1.ENABLED False SAM.ENABLED True SAM.SAFE_BASELINE True \
  OUTPUT_DIR logs/UAD_SAM_B_1000_DDP
```

## WHU-MARS-1000：总体、九方向和AG

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --config_file configs/UAD_SAM_B.yml \
  MODEL.DIST_TRAIN False MODEL.PRETRAIN_PATH "$PRETRAIN" \
  DATASETS.ROOT_DIR "$DATA_ROOT" DATASETS.DATASET_DIR WHU-MARS DATASETS.FILTER_GD False \
  TEST.IMS_PER_BATCH 512 TEST.NECK_FEAT before TEST.FEAT_NORM yes \
  TEST.RE_RANKING False TEST.TOP_K_EVAL 0 TEST.DETAILED_DOMAIN_EVAL False \
  TEST.WEIGHT logs/UAD_SAM_B_1000_DDP/transformer_120.pth \
  OUTPUT_DIR logs/UAD_SAM_B_1000_DDP/test
```

## WHU-MARS-1000：细分模态与平台协议

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --config_file configs/UAD_SAM_B.yml \
  MODEL.DIST_TRAIN False MODEL.PRETRAIN_PATH "$PRETRAIN" \
  DATASETS.ROOT_DIR "$DATA_ROOT" DATASETS.DATASET_DIR WHU-MARS DATASETS.FILTER_GD False \
  TEST.IMS_PER_BATCH 512 TEST.NECK_FEAT before TEST.FEAT_NORM yes \
  TEST.RE_RANKING False TEST.TOP_K_EVAL 0 TEST.DETAILED_DOMAIN_EVAL True \
  TEST.WEIGHT logs/UAD_SAM_B_1000_DDP/transformer_120.pth \
  OUTPUT_DIR logs/UAD_SAM_B_1000_DDP/domain_eval
```

## WHU-MARS-1000-GD：复用同一1000模型

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --config_file configs/UAD_SAM_B.yml \
  MODEL.DIST_TRAIN False MODEL.PRETRAIN_PATH "$PRETRAIN" \
  DATASETS.ROOT_DIR "$DATA_ROOT" DATASETS.DATASET_DIR WHU-MARS DATASETS.FILTER_GD True \
  TEST.IMS_PER_BATCH 512 TEST.NECK_FEAT before TEST.FEAT_NORM yes \
  TEST.RE_RANKING False TEST.TOP_K_EVAL 0 TEST.DETAILED_DOMAIN_EVAL False \
  TEST.WEIGHT logs/UAD_SAM_B_1000_DDP/transformer_120.pth \
  OUTPUT_DIR logs/UAD_SAM_B_1000_DDP/test_GD
```

## 安全UAD对照及2337独立训练

下面封装与上面的直接命令使用同一配置。1000全部目标达标后再训练2337。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_uad_sam_b.sh baseline1000
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_uad_sam_b.sh train2337
CUDA_VISIBLE_DEVICES=0 bash scripts/run_uad_sam_b.sh test2337
```

安全UAD对照输出在 `logs/UAD_safe_1000_control`，评测时应使用该目录权重并单独设置OUTPUT_DIR，不覆盖SAM报告。
完整机制、AMP失败处理和验证边界见 [实施说明](UAD_SAM_B_IMPLEMENTATION.md)。正式结果统一使用epoch120，不按协议挑选checkpoint。

---

# 历史UAD运行命令（保留原文）

# DDP
## WHU-MARS-1000

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone \
  --nproc_per_node=4 \
  train.py \
  --config_file configs/UAD.yml \
  MODEL.DIST_TRAIN True \
  SOLVER.IMS_PER_BATCH 256 \
  SOLVER.BASE_LR 0.032 \
  OUTPUT_DIR logs/UAD_1000_DDP

CUDA_VISIBLE_DEVICES=7 python test.py \
  --config_file configs/UAD.yml \
  DATASETS.DATASET_DIR WHU-MARS \
  TEST.IMS_PER_BATCH 512 \
  TEST.WEIGHT logs/UAD_1000_DDP/transformer_120.pth \
  OUTPUT_DIR logs/UAD_1000_DDP

CUDA_VISIBLE_DEVICES=7 python test.py \
  --config_file configs/UAD.yml \
  MODEL.DIST_TRAIN False \
  TEST.IMS_PER_BATCH 512 \
  TEST.WEIGHT logs/UAD_1000_DDP/transformer_120.pth \
  TEST.DETAILED_DOMAIN_EVAL True \
  OUTPUT_DIR logs/UAD_1000_DDP_domain_eval

## WHU-MARS-1000-GD
CUDA_VISIBLE_DEVICES=6 python test.py \
  --config_file configs/UAD.yml \
  DATASETS.FILTER_GD True \
  TEST.IMS_PER_BATCH 512 \
  TEST.WEIGHT logs/UAD_1000_DDP/transformer_120.pth \
  OUTPUT_DIR logs/UAD_1000_GD

## WHU-MARS-2337
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone \
  --nproc_per_node=4 \
  train.py \
  --config_file configs/UAD.yml \
  MODEL.DIST_TRAIN True \
  DATASETS.DATASET_DIR WHU-MARS-2337 \
  SOLVER.IMS_PER_BATCH 256 \
  SOLVER.BASE_LR 0.032 \
  OUTPUT_DIR logs/UAD_2337_DDP

# Single
## WHU-MARS-1000

cd /project/code/WHU_base/WHU_DDP

CUDA_VISIBLE_DEVICES=7 python train.py \
  --config_file configs/UAD.yml \
  MODEL.DIST_TRAIN False \
  DATASETS.DATASET_DIR WHU-MARS \
  SOLVER.IMS_PER_BATCH 64 \
  SOLVER.BASE_LR 0.008 \
  SOLVER.SEED 1234 \
  TEST.IMS_PER_BATCH 1024 \
  OUTPUT_DIR logs/UAD_1000_single2

CUDA_VISIBLE_DEVICES=0 python test.py \
  --config_file configs/UAD.yml \
  MODEL.DIST_TRAIN False \
  DATASETS.DATASET_DIR WHU-MARS \
  TEST.IMS_PER_BATCH 1024 \
  TEST.WEIGHT logs/UAD_1000_single/transformer_120.pth \
  OUTPUT_DIR logs/UAD_1000_single

## WHU-MARS-1000-GD

CUDA_VISIBLE_DEVICES=0 python test.py \
  --config_file configs/UAD.yml \
  MODEL.DIST_TRAIN False \
  DATASETS.DATASET_DIR WHU-MARS \
  DATASETS.FILTER_GD True \
  TEST.IMS_PER_BATCH 512 \
  TEST.WEIGHT logs/UAD_1000_single/transformer_120.pth \
  OUTPUT_DIR logs/UAD_1000_single_GD

## WHU-MARS-2337

CUDA_VISIBLE_DEVICES=6 python train.py \
  --config_file configs/UAD.yml \
  MODEL.DIST_TRAIN False \
  DATASETS.DATASET_DIR WHU-MARS-2337 \
  SOLVER.IMS_PER_BATCH 64 \
  SOLVER.BASE_LR 0.008 \
  SOLVER.SEED 1234 \
  TEST.IMS_PER_BATCH 1024 \
  OUTPUT_DIR logs/UAD_2337_single

CUDA_VISIBLE_DEVICES=0 python test.py \
  --config_file configs/UAD.yml \
  MODEL.DIST_TRAIN False \
  DATASETS.DATASET_DIR WHU-MARS-2337 \
  TEST.IMS_PER_BATCH 1024 \
  TEST.WEIGHT logs/UAD_2337_single/transformer_120.pth \
  OUTPUT_DIR logs/UAD_2337_single