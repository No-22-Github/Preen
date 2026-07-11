#!/usr/bin/env bash
# 方案 E 第3项:训练对照 (int8, fp32) vs 基线 (bf16, fp32)
#
# 两组背靠背,1.5B seed42 NekoQA 200×2ep,同 commit 同 session。
# int8 组双解码(@M_q 诊断 + @M 判据),基线组只 @M。
#
# 预计耗时:~12-15 分钟(1.5B 两组训练+解码+导出,超 10min 单命令硬超时,需手动跑)
#
# 用法:
#   caffeinate -i bash experiments/mixed_precision/run_int8_traincompare.sh
#
# 产物:experiments/mixed_precision/data/int8_traincompare/{15b_s42_fp32, 15b_s42_int8}/

set -euo pipefail
cd "$(dirname "$0")/../.."

PY=.venv/bin/python
export PYTHONPATH=src:experiments/mixed_precision:tools

MODEL=models/converted/rwkv7-g1g-1.5b
DATA=train_data/NekoQA_10k/nekoqa_smoke_200.json
OUTDIR=experiments/mixed_precision/data/int8_traincompare

mkdir -p "$OUTDIR"

echo "════════════════════════════════════════════════════════════════"
echo "方案 E 第3项:训练对照 (int8,fp32) vs (bf16,fp32)"
echo "1.5B seed42 NekoQA 200×2ep,两组背靠背"
echo "开始: $(date)"
echo "════════════════════════════════════════════════════════════════"

# ── 组1:基线 (bf16, fp32) —— 矩阵报告标"fp32"的那条 ──
echo ""
echo "─── [1/2] 基线 (bf16, fp32) ───"
"$PY" experiments/mixed_precision/matrix_train.py \
  --model "$MODEL" --data "$DATA" \
  --precision fp32 --seed 42 \
  --out-dir "$OUTDIR/15b_s42_fp32" \
  --cache-limit-gb 4 --epochs 2 --ctx-len 512 --lr 0.01 \
  2>&1 | tee "$OUTDIR/15b_s42_fp32.log"

echo ""
echo "─── [2/2] 方案 E (int8, fp32) ───"
"$PY" experiments/mixed_precision/matrix_train.py \
  --model "$MODEL" --data "$DATA" \
  --precision int8 --seed 42 \
  --out-dir "$OUTDIR/15b_s42_int8" \
  --cache-limit-gb 4 --epochs 2 --ctx-len 512 --lr 0.01 \
  2>&1 | tee "$OUTDIR/15b_s42_int8.log"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "完成: $(date)"
echo "产物: $OUTDIR/{15b_s42_fp32, 15b_s42_int8}/"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "把两个 .log 和两个 decode.json 贴回给我,我来做配对判定 + 失败拆账。"
