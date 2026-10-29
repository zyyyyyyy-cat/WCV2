# UAD-SAM-B 实施说明与四卡运行命令

本次直接修改本地 `D:/CODE/Doing/WHU_base/WCV1/WHU`。正式方案是原 UAD 的 CE + WRT + 0.01 ProCA + GPD，增加仅骨干参数扰动的归一化 SAM；不包含 WCV1 辅助损失、EMA、特征融合或重排序。尚未运行真实数据训练，不声明指标改善。

交付包含完整源码ZIP、仅修改文件ZIP及文本补丁。优先使用完整源码；补丁按LF生成，针对原CRLF文件可先运行 `git apply --check --ignore-space-change UAD_SAM_B_changes.patch` 检查。修改前源码快照保存在本地工作区的 `work/UAD_SAM_B_before/source_snapshot.zip`。

## 代码与接口

- `processor/sam_training.py`：单遍安全 UAD / 双遍 SAM 步骤、全 rank finite 判断、AMP 跳步、骨干精确恢复、BN/RNG 管理。
- `processor/processor.py`：PCA 可接受固定身份中心；GPD 可接受固定原型快照。原调用默认返回与语义不变。新增 `uad_forward_loss` 供双遍复用；训练入口按配置选择路径。
- `config/defaults.py`、`configs/UAD_SAM_B.yml`：显式配置与验证。旧配置 `SAM.ENABLED=False, SAFE_BASELINE=False`，保留原实验行为；新配置 `SAFE_BASELINE=True`，即使关闭 SAM 也使用安全的单遍 UAD。
- `train.py`：新方案及安全对照写出 `run_manifest.json`，记录实际配置、源码 SHA256、预训练文件 SHA256、按顺序的数据清单 hash、训练身份数、world size 与运行环境。清单 hash 不覆盖图片文件内容。
- `scripts/run_uad_sam_b.sh`：六种完整运行模式。
- `tests/test_sam_training.py`：CPU 检查及服务器 CUDA/NCCL 合成检查入口，不读取 WHU 图像。

模型结构、原四损失系数、PKM、图像增强、推理特征和评测器未改。新方案与 WCV1 同时开启会报错，不会偷偷叠加训练目标。

## 完整训练步骤

每卡64个三模态采样位置，对应16个身份×4张×3模态，即192张图。三种模态沿batch连接。原始CLS为ViT最终LayerNorm之后、BN之前的768维特征；CE经BNNeck和classifier，WRT直接使用原始CLS，ProCA/GPD使用L2归一化CLS。

总损失为 `CE + WRT + 0.01*ProCA + GPD`，GPD温度0.03。WRT保留原mask、可导权重与数值实现；两遍各自计算WRT。ProCA严格遵循样本归一化→K个样本平均→模态中心归一化→三模态平均→归一化→detach的顺序。

正常前向产生detach身份中心和包含冷启动填充的GPD临时原型快照。它们在两遍之间不变。扰动前向重新计算模态中心及实例表示，但不移动其身份对齐目标；这是明确的SAM适配，不是声称原ProCA所有前向行为完全不变。

正常反向先完成DDP梯度平均，以未裁剪、解scale后的骨干参数梯度构造 `epsilon = rho * g_backbone / (norm(g_backbone)+1e-12)`。不再对已经平均的范数跨卡求和。只扰动 `model.base` 内实际有梯度的参数，排除unused ImageNet分类头；外部classifier与BNNeck不扰动，最终仍正常更新。

同一增强图像复用；重放正常前向之前的随机状态，使DropPath/dropout一致。第二遍仍为训练模式；结束后将BN持久buffer恢复为正常前向后的值，外层RNG恢复为正常前向后状态，因此只推进一次。

参数用备份 `copy_` 精确恢复，随后仅第二遍梯度执行一次 unscale→clip(1.0)→原SGD。没有clean/adv梯度平均或额外损失权重。每次成功SGD后，使用正常前向中心跨卡汇总，按 `normalize(0.8*old + 0.2*new)` 更新bank一次。

第一遍scaled梯度仅以FP32除scale读取，不调用 `unscale_`；最终梯度只调用一次 `unscale_`。任一卡overflow会同步为全rank跳步；在unscale前给丢弃梯度写入overflow标记，使用公开GradScaler路径完成跳步、scale减半和增长计数归零。失败不提交bank，BN回滚本步开始状态，RNG保留正常前向后状态。forward loss非finite则同步后共同终止。异常路径恢复参数，不允许携带扰动权重继续运行。

新安全入口与历史代码的区别仅需单独注意：旧代码AMP跳过SGD仍可能更新bank；新入口修复此问题。正常有限、零半径步骤已核对与原UAD数值等价，不能把这句话扩展为历史溢出行为也逐行一致。

## 固定配置

- 原ImageNet ViT-B/16初始化，不从失败WCV1继续训练。
- seed1234；四卡global sampler batch256，实际全局图像768；K4。
- SGD，LR0.032，bias倍率2，momentum0.9，weight decay与bias decay均1e-4；原warmup5与120轮cosine保留。
- epoch1—5：普通UAD；6—10：rho依次 .004/.008/.012/.016/.020；11—120：.020。
- SAM不是每层或每个参数加.02；.02为整个参与扰动骨干向量的L2半径。
- 输入256×128，原增强顺序不变。`adaptive=False, stable=False`。
- 每10轮评测，正式统一使用epoch120；默认checkpoint120，可自行明确覆写保存周期，但不改变统一最终checkpoint验收条件。

日志每50步分开输出clean与adv四损失、模态CE/GPD/ProCA、骨干梯度范数、实际扰动范数、裁剪前梯度范数、clip倍率、loss变化、scale及跳步次数。日志步骤另外记录各骨干block/embedding/norm的绝对及相对参数范数扰动；不为此额外反向。epoch汇总记录成功与跳过次数，不能只看最后一条日志判断全程正常。

保存 `transformer_120.pth` 供原test.py加载；另存 `training_state_120.pth`，含optimizer、scheduler、scaler、bank、各rank RNG、配置与成功步数。该文件是归档，不是已经实现自动resume的承诺；不要传给test.py。

## 服务器准备与必要检查

以下命令为Linux bash。把完整代码上传/解压到 `WHU_CODE`；路径按服务器实际情况修改。示例复用现有WC目录中的原预训练权重，不需要复制大权重。不要安装原 `requirements.txt`，也不要重装已存在的PyTorch1.12.1/CUDA11.3。

```bash
export WHU_CODE=/project/code/WHU_base/UAD_SAM_B
export DATA_ROOT=/project/code/data
export PRETRAIN=/project/code/WHU_base/WC/jx_vit_base_p16_224-80ecf9dd.pth
cd "$WHU_CODE"

python -c "import torch; print(torch.__version__,torch.version.cuda); assert torch.cuda.is_available(); assert torch.cuda.device_count()>=4"
python -m compileall -q config datasets loss model processor solver utils tests train.py test.py
python tests/test_sam_training.py
CUDA_VISIBLE_DEVICES=0 python tests/test_sam_training.py --cuda
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 \
python tests/test_sam_training.py --backend nccl --world-size 4
```

最后两项只运行很小的合成模型，检查连续双遍、真实GradScaler、单rank注入clean/adv overflow、四卡扰动一致和原型提交次数。脚本自行spawn；不要再套torchrun。它们也不能代替真实ViT及WHU数据的训练验证。

## 四卡完整训练命令

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 \
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 train.py \
  --config_file configs/UAD_SAM_B.yml MODEL.DIST_TRAIN True \
  MODEL.PRETRAIN_PATH "$PRETRAIN" DATASETS.ROOT_DIR "$DATA_ROOT" \
  DATASETS.DATASET_DIR WHU-MARS DATASETS.FILTER_GD False \
  SOLVER.IMS_PER_BATCH 256 SOLVER.BASE_LR 0.032 SOLVER.SEED 1234 \
  SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 10 SOLVER.LOG_PERIOD 50 \
  SOLVER.CHECKPOINT_PERIOD 120 WCV1.ENABLED False \
  SAM.ENABLED True SAM.SAFE_BASELINE True OUTPUT_DIR logs/UAD_SAM_B_1000_DDP
```

等价脚本命令，二选一，不要重复训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_uad_sam_b.sh train1000
```

## 同一最终模型评测与其他任务

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_uad_sam_b.sh test1000
CUDA_VISIBLE_DEVICES=0 bash scripts/run_uad_sam_b.sh testgd

# 必要时补同入口安全UAD对照；不是开始主方案的前置要求。
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_uad_sam_b.sh baseline1000

# 1000全协议达标后，再独立训练2337。
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_uad_sam_b.sh train2337
CUDA_VISIBLE_DEVICES=0 bash scripts/run_uad_sam_b.sh test2337
```

1000/GD共用 `logs/UAD_SAM_B_1000_DDP/transformer_120.pth`；test1000输出原总体、九方向及AG，GD仅合法筛选。原始pre-BN CLS→L2→平方欧氏，完整gallery，无rerank。

安全UAD对照的独立评测命令（单独输出，避免覆盖SAM报告）：

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/UAD_SAM_B.yml \
  MODEL.DIST_TRAIN False MODEL.PRETRAIN_PATH "$PRETRAIN" DATASETS.ROOT_DIR "$DATA_ROOT" \
  DATASETS.DATASET_DIR WHU-MARS DATASETS.FILTER_GD False \
  TEST.WEIGHT logs/UAD_safe_1000_control/transformer_120.pth \
  TEST.NECK_FEAT before TEST.FEAT_NORM yes TEST.RE_RANKING False TEST.TOP_K_EVAL 0 \
  SAM.ENABLED False SAM.SAFE_BASELINE True OUTPUT_DIR logs/UAD_safe_1000_control/test
```

## 当前验证边界

本机Python3.9.13、PyTorch2.8.0 CPU，无CUDA。已完成34项CPU回归/合成检查、全源码编译、真实ViT无预训练单图前向接口检查、Bash语法检查及独立代码审阅。真实CPU GradScaler检查属于当前2.8环境；1.12服务器会跳过该CPU专属检查，而由上述CUDA合成命令验证真正1.12的GradScaler。

没有运行真实WHU训练、没有实际四卡NCCL验证、没有验证指标提升。所有预期收益仍需完整120轮及统一全协议验收；任一目标单元未达标不能以总体均值抵消。
