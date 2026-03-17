#!/bin/bash
# 单张 GPU 体验训练脚本（A100 / H100）
# 预计时间：1~2小时，费用：$3~5
# 用法：cd nanochat && bash ../dev/run_single_gpu.sh

set -e
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# 下载 4 个数据 shard（约 400MB，够跑了）
echo ">>> 下载数据..."
python -m nanochat.dataset -n 4

# 训练 tokenizer（在 4 个 shard 上）
echo ">>> 训练 tokenizer..."
python -m scripts.tok_train --max_chars=500000000

# 训练一个 depth=8 的小模型
# - 单张 A100 约跑 1500 步，loss 能从 11 降到 ~5 左右，能看到明显学习曲线
# - 预计约 1~1.5 小时
echo ">>> 开始训练..."
python -m scripts.base_train \
    --depth=8 \
    --device-batch-size=8 \
    --total-batch-size=524288 \
    --num-iterations=1500 \
    --eval-every=100 \
    --eval-tokens=524288 \
    --core-metric-every=500 \
    --core-metric-max-per-task=50 \
    --sample-every=500 \
    --window-pattern=L

echo ">>> 查看模型生成的文字样本..."
python -m scripts.base_loss --device-batch-size=8

echo ">>> 完成！"
