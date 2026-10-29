# WCV1 / UAD 一次性免重训诊断

独立入口：`diagnose_wcv1.py`。仅新增diagnostics目录、诊断tests、入口和本说明；原train.py、processor、模型、训练损失、采样器与原评价器均不修改。不要使用torchrun启动本脚本，多个进程同时写同一缓存会冲突。

## 一、完成哪些事情

1. 两个完整模型的train/query/gallery原始pre-BN CLS一次导出。与原standalone test相同的resize、normalize、FP32/eval、无flip；原始768维未L2特征保存为npy，分析时才L2归一化。
2. 13个协议完整排名：overall、九对模态、A→G、G→A、1000-GD。全部同相机gallery排除；无正例query跳过并记录。输出AP/INP、首/尾正例排名、首负/最佳正例及两个模型极值并集交叉评分，四类转变和R1闭合。
3. seed1234下8套、每套8个模拟四卡PKM候选组织。与原DDP sampler的numpy抽样逐项对照；在训练身份内部比较随机池与完整训练库的危险负例，输出身份与图像曝光、softmax熵/有效负例数、batch分离但全库错误、正例模态/平台关系。绝不以test身份挖训练负例。
4. 16组四本卡FP32梯度探针，顺序模拟各卡local BN，不保留四份ViT计算图。复用原CE/WRT/ProCA/GPD，测真实F、三模态F和末ViT block的梯度norm/cos，拆正吸引/负排斥并验证梯度相加闭合。只用autograd.grad，无optimizer更新、无bank更新。逐组恢复BN buffers及RNG，最终核验模型state SHA256不变。

bank分别输出reconstructed和可选archived。重建bank是各模型全训练缓存的模态等权中心；不是原训练现场bank。外部归档必须包含同一模型权重，脚本逐tensor核对，不匹配就拒绝使用。终态探针不能证明历史中期因果；eval缓存覆盖也不是训练现场精确回放。

## 二、服务器准备与一次完整命令

服务器环境目标PyTorch1.12.1/CUDA11.3；无需新装PyTorch。现有源码已能训练时，沿用其环境。诊断只额外使用标准库与已有numpy/Pillow/torch/torchvision/timm/yacs；本地pytest用于自检，不是服务器执行入口的依赖。

下面路径是示例，先替换为服务器的真实位置：

```bash
cd /project/code/WHU_base/WCV1/WHU
export DATA_ROOT=/project/code/data
export UAD_WEIGHTS=/project/code/WHU_base/WHU/logs/UAD_1000_DDP/transformer_120.pth
export WCV1_WEIGHTS=/project/code/WHU_base/WCV1/WHU/logs/WCV1_1000_DDP/transformer_120.pth
export UAD_LOG=/project/code/WHU_base/WHU/logs/1000_train_log_20260911_2145.txt
export WCV1_LOG=/project/code/WHU_base/WCV1/WHU/logs/train_log_20261028_2339.txt
export DIAG_CACHE=/project/code/WHU_base/WCV1/diagnosis_cache_120
export DIAG_OUT=/project/code/WHU_base/WCV1/diagnosis_120

# 可选：原训练保存的完整状态归档；没有就不添加。
ARCHIVE_ARGS=()
if [ -f logs/WCV1_1000_DDP/training_state_120.pth ]; then
  ARCHIVE_ARGS+=(--wcv1-archive logs/WCV1_1000_DDP/training_state_120.pth)
fi

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
python diagnose_wcv1.py \
 --config configs/WCV1.yml --data-root "$DATA_ROOT" --dataset-dir WHU-MARS \
 --uad-checkpoint "$UAD_WEIGHTS" --wcv1-checkpoint "$WCV1_WEIGHTS" \
 --cache-dir "$DIAG_CACHE" --output-dir "$DIAG_OUT" \
 --uad-log "$UAD_LOG" --wcv1-log "$WCV1_LOG" \
 --device cuda:0 --batch-size 256 --workers 4 --chunk-size 64 \
 --seed 1234 --coverage-rounds 8 --batches-per-round 8 --probe-groups 16 \
 "${ARCHIVE_ARGS[@]}"
```

这是单进程单GPU诊断，顺序模拟训练时四个本卡batch，不是新训练或四卡优化。全图库numpy排名和候选覆盖在CPU执行；GPU主要用于导出与FP32梯度。不要为了加速再套四卡torchrun。

训练模型必须是**完整state_dict**，或含model字段的training_state归档；严格加载包含classifier。不能用跳过classifier的原load_param加载探针模型，也不能拿ImageNet初始化权重替代训练模型。`--pretrain`可选，仅记录该文件hash；完整训练权重已覆盖所有参数，不要求再次执行ImageNet初始化。这个hash不能证明历史训练实际用了该文件。

## 三、断点与缓存复用

特征完成每个split后才写marker。中断的partial不算完成；重新运行同一完整命令会核验并复用已完成split。全训练库极值按chunk存储，已完成chunk也能复用。改变权重、配置、代码、清单或缓存内容后拒绝静默复用，须使用新cache目录。不要一边分析一边修改诊断源文件。

不再加载模型、无需数据或checkpoint路径、可在CPU机器复用已有缓存：

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
python diagnose_wcv1.py --stages ranking,coverage \
 --cache-dir "$DIAG_CACHE" --output-dir "$DIAG_OUT/reanalysis" \
 --uad-log "$UAD_LOG" --wcv1-log "$WCV1_LOG" \
 --chunk-size 64 --seed 1234 --coverage-rounds 8 --batches-per-round 8
```

只补梯度：复用完整命令的config/data/checkpoint/cache参数和可选归档，增加`--stages probes`，无需重提特征。不导出特征时仍校验已有cache与当前模型、代码、数据清单一致。

可先把coverage-rounds/batches-per-round/probe-groups各设1做服务器功能冒烟；它不是正式8套/16组诊断。正式重跑默认数量，不将冒烟当完整结果。

## 四、输出与应回传的材料

- `report.md`：汇总与解释边界。
- `run_summary.json`：实际完成阶段、参数、指纹、日志重算差值。
- `ranking/*_queries.csv`：每query两模型指标、转变、正负margin变化、尾部变化。
- `ranking/*_candidate_union.csv`：两模型极值并集在两模型的交叉分数，含路径和身份。
- `ranking/*_invalid_queries.csv`：跳过原因；`ranking/summary.json`含四转变与R1闭合。
- `coverage/*_candidate_exposure.csv`：危险负身份/图像曝光与batch/full差异。
- `coverage/*_positive_relations.csv`：模态×平台正例覆盖与吸引系数。
- `probes/gradient_pairs.csv`、`probes/gradient_summary.json`：损失分解、作用位置、bank标签和模型state不变检查。
- `sampling_plan.json`：两模型共用的精确池索引。
- cache目录`manifest.json`、每split marker：记录数据清单、checkpoint/代码/config/特征内容SHA256。

CSV指标AP/INP/R1等为fraction，汇总Markdown乘100显示百分数。coverage两套梯度系数分别为per_anchor_unscaled与scaled_global；后者含全局anchor均值，但仍未乘λ，不是参数梯度贡献。probes的norm/cos默认未预算，budget_weight单独提供。

请回传整个DIAG_OUT（含CSV），以及cache/manifest.json和六个split marker；不必先上传约GB级npy。不要只回传总体summary，否则无法进一步配对定位原因。

## 五、完整性与资源边界

默认对日志的overall、九方向、AG末轮指标检查，容差0.02个百分点。超差时保存已完成ranking报告并停止后续因果诊断，先检查权重、数据、协议、数值与近并列排序。numpy cosine稳定排序与原torch平方欧氏排序在近并列处可能有微小差异；显示一致不等于每query逐项相同。未给日志则标NOT_CHECKED，不宣称复算通过。GD没有独立日志时只输出诊断结果。

特征每十万图每模型约307MB磁盘（float32×768），真实规模以manifest为准；双模型三split、全训练库覆盖和完整排序有显著CPU/磁盘成本，不能承诺秒级完成。chunk64限制距离块内存，不截断gallery。可降低导出batch-size以减少eval显存而不改变BN语义；FP32梯度使用原本卡192，不能随意拆小batch冒充同一BN目标。若该阶段OOM，先保留已完成排名/覆盖结果，记录probes未完成，不伪造通过。

本地只运行合成与代码检查，没有真实WHU数据/checkpoint，未验证目标1.12.1 CUDA全流程。原模型checkpoint、训练目录和评价代码不写入；模型及原型无训练更新。诊断结果回来后才按证据选择一个主修正方向，不自动修改或重启训练。
