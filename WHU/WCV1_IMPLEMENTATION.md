# WCV1 修改说明与服务器运行

代码已直接修改于 D:/CODE/Doing/WHU_base/WCV1/WHU。模型、四个原损失、评测器和768维推理接口保持原样。

## 修改内容

新增loss/mixed_rank.py：归一化CLS、所有候选detach、跨卡gather、按稳定文件键保留首个候选、全部同相机排除、负实例logsumexp、正例gallery模态组均值、全局有效anchor模态均值。预算在真实F上查询原联合目标和aux梯度，包含CE经BN的路径；仅最终backward累积参数梯度。非finite全rank同步停用aux。

datasets/bases.py与make_dataloader.py：可选第四项sample keys；模态目录参与键生成，原PKM/图像增强不改。config/defaults.py默认关闭WCV1，configs/WCV1.yml启用：tau .07、margin0、前10轮关闭、11—20轮ramp、rho .10、lambda上限1。processor保留原AMP/clip/SGD/GPD更新，增加aux与诊断；候选gather开关独立于DDP全局均值开关。

保留transformer_120.pth供test.py使用；另存training_state_120.pth归档optimizer/scheduler/scaler/bank/逐rank RNG等。尚未实现自动resume入口，不能把该归档直接传给test.py，也不承诺数据worker精确恢复。

## 自检与边界

全源码compileall通过；4项CPU合成测试通过：参考公式与单侧梯度、去重及metadata冲突、无效行有限零梯度、预算和非finite处理。数据接口合成核对通过：关闭3元batch、开启4元batch、三模态同名不同键。

本机为PyTorch2.8 CPU，不具备目标1.12.1/CUDA环境。尝试Gloo初始化遇到本机unsupported gloo device，因此未验证实际分布式collective。独立agent完成原链路分析并发现、推动修复本卡候选与全局均值开关混用问题；其最终审阅因服务用量限制中断，不能声称完整独立终审通过。主线程另检查实际修改及数值自检。无真实数据训练或指标提升结果。

## 服务器准备

以下为Linux bash。只需将代码、数据与预训练路径设为实际服务器位置。已装torch1.12.1+cu113时不要重装torch，也不要安装原requirements.txt。可选requirements_torch112.txt按源码实际导入提供旧版依赖组合，非引用原版本；这是建议组合，尚未在目标环境全栈实测。

```bash
export WHU_CODE=/project/code/WCV1/WHU
export DATA_ROOT=/project/code/data
export PRETRAIN=/project/code/pretrained/jx_vit_base_p16_224-80ecf9dd.pth
cd "$WHU_CODE"
python -c "import torch,torchvision; print(torch.__version__,torch.version.cuda,torchvision.__version__); assert torch.cuda.device_count()>=4"
# 仅在缺相关依赖时安装；保留既有兼容环境亦可。
python -m pip install -r requirements_torch112.txt
python -m compileall -q config configs datasets loss model processor solver utils train.py test.py
python tests/test_mixed_rank.py --skip-ddp
# 该脚本自己spawn两个进程，不要再套torchrun。
CUDA_VISIBLE_DEVICES=0,1 python tests/test_mixed_rank.py --backend nccl
```

最后一项检验连续DDP步骤、BN、unused head、真实F探针与最终backward；务必在目标服务器确认通过后启动正式训练。查看测试源码是否启用AMP；该检查不能替代真实ViT全流程验证。

## 四卡完整训练与测试命令

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 \
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 train.py \
 --config_file configs/WCV1.yml MODEL.DIST_TRAIN True \
 MODEL.PRETRAIN_PATH "$PRETRAIN" DATASETS.ROOT_DIR "$DATA_ROOT" \
 DATASETS.DATASET_DIR WHU-MARS DATASETS.FILTER_GD False \
 SOLVER.IMS_PER_BATCH 256 SOLVER.BASE_LR 0.032 SOLVER.SEED 1234 \
 SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 10 OUTPUT_DIR logs/WCV1_1000_DDP

# 等价封装及其余完整任务：共享上面的DATA_ROOT/PRETRAIN环境变量。
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_wcv1.sh train1000
CUDA_VISIBLE_DEVICES=0 bash scripts/run_wcv1.sh test1000
CUDA_VISIBLE_DEVICES=0 bash scripts/run_wcv1.sh testgd
# 先1000全部达标，再独立2337训练。
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_wcv1.sh train2337
CUDA_VISIBLE_DEVICES=0 bash scripts/run_wcv1.sh test2337
# 原UAD完整控制
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_wcv1.sh baseline1000
```

首条展开命令和train1000封装是二选一，不需重复训练。所有封装实际完整参数见scripts/run_wcv1.sh。testgd仍读取1000训练模型，只筛评测数据。测试固定原始L2特征、完整gallery、无rerank；所有协议共用epoch120。日志检查weight、grad_ratio、grad_cosine、finite、valid_mod、negative_count和saturation，不能只看aux loss下降。
