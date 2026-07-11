#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# 大满贯脚本:多 seed 精度矩阵(10 组)+ 红线标定 + 官方精度确认
#
# 用法(项目根目录):
#   caffeinate -i bash experiments/mixed_precision/run_full.sh
#
# 全程预计 2-4 小时(10 组训练 × 每组约 3-8 分钟 + 分桶 + 红线标定)。
# 全自动,无需交互。每组独立进程串行跑(防 cache 池跨组污染)。
# 中途某组失败如实记录(失败也是数据),不中断后续组。
#
# 阶段:
#   0. 分桶数据集准备(秒级)
#   1. 矩阵训练 10 组(主体)
#   2. 红线标定(bf16 + c4G 逐桶上探)
#   3. 汇总分析(配对判定表 + state 余弦 + 红线 + 十问附录)
#
# 产物:data/matrix/ 下每组一个子目录 + redline_result.json + summary.txt
# ──────────────────────────────────────────────────────────────────────────────
set -uo pipefail  # 注意:不用 -e,单组失败不中断整体

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

MODEL_04B="${MODEL_04B:-models/converted/rwkv7-g1d-0.4b}"
MODEL_15B="${MODEL_15B:-models/converted/rwkv7-g1g-1.5b}"
DATA="train_data/NekoQA_10k/nekoqa_smoke_200.json"
DATA_DIR="experiments/mixed_precision/data/matrix"
LOG="experiments/mixed_precision/data/matrix_run.log"
export PYTHONPATH=src:experiments/mixed_precision:tools

mkdir -p "$DATA_DIR"
: > "$LOG"

ts() { date "+%H:%M:%S"; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }

say "═══════════════════════════════════════════════════════════════"
say "大满贯:精度矩阵 + 红线标定  开始"
say "═══════════════════════════════════════════════════════════════"
say "0.4B: $MODEL_04B"
say "1.5B: $MODEL_15B"
say "数据: $DATA"
[[ -d "$MODEL_04B" ]] || { say "❌ 0.4B 模型不存在"; exit 1; }
[[ -d "$MODEL_15B" ]] || { say "❌ 1.5B 模型不存在"; exit 1; }
[[ -f "$DATA" ]] || { say "❌ 数据不存在"; exit 1; }
say "commit: $(git rev-parse --short HEAD)"

# ── 阶段 0:分桶数据集(红线标定用,秒级)─────────────────────────────────
say ""
say "─── 阶段 0:分桶数据集准备 ───"
# 30K 版长样本充足(>=450 有 4500 条)。围绕预期断点 ~697 密采:
# 450/550 安全区,600 安全侧接近断点,650 偏紧,700 可能越界。
.venv/bin/python experiments/mixed_precision/build_buckets.py \
  --source train_data/NekoQA_30k/MaoDieQA-30K.json \
  --model "$MODEL_15B" \
  --targets 450,550,600,650,700 --per-bucket 40 \
  --out-dir experiments/mixed_precision/data/redline_buckets \
  >> "$LOG" 2>&1 || say "⚠️ 分桶失败(红线标定将跳过)"

# 延长数值冒烟(至 700 步,真实 w)+ w 分布 dump —— decision-precision.md 验证边界依赖这两项
say ""
say "─── 阶段 0b:延长数值冒烟(700 步,真实 w 分布)───"
# 原冒烟只覆盖 273 步 + 合成 w。标定探到 700,必须:
#   (1) 用真实模型 w 分布(fp32 dump)跑,21.9% 通道 w>0.999 是真问题不是合成假象
#   (2) 延长到 700 步,验证长序列累积
#   (3) 按 w 分桶拆误差,裁决"为何高w通道没让全局误差走阔"(v7主动擦除 vs 全局稀释)
.venv/bin/python experiments/mixed_precision/smoke_numeric.py \
  --steps 700 --heads 32 --head-dim 64 --seed 42 --record-every 5 \
  --real-w-model "$MODEL_15B" \
  --out experiments/mixed_precision/data/smoke_15b_700_realw.json \
  >> "$LOG" 2>&1 || say "⚠️ 延长冒烟失败"

say "─── 阶段 0c:w 分布 dump(回答'离 bf16 危险区多远')───"
# 把 decision-precision.md 里"w 不极端接近 1"的假设转为实测出处
# 0.4B 和 1.5B 各 dump
.venv/bin/python experiments/mixed_precision/dump_w_dist.py \
  --model "$MODEL_04B" --data "$DATA" \
  --out experiments/mixed_precision/data/w_dist_04b.json \
  >> "$LOG" 2>&1 || say "⚠️ 0.4B w dump 失败"
.venv/bin/python experiments/mixed_precision/dump_w_dist.py \
  --model "$MODEL_15B" --data "$DATA" \
  --out experiments/mixed_precision/data/w_dist_15b.json \
  >> "$LOG" 2>&1 || say "⚠️ 1.5B w dump 失败"

# ── 阶段 1:矩阵训练 10 组(主体)──────────────────────────────────────────
say ""
say "─── 阶段 1:矩阵训练(10 组,每组独立进程)───"

# 定义 10 组:(标签, 模型, seed, 精度)
GROUPS=(
  "04b_s42_fp32|$MODEL_04B|42|fp32"
  "04b_s42_bf16|$MODEL_04B|42|bf16"
  "04b_s1042_fp32|$MODEL_04B|1042|fp32"
  "04b_s1042_bf16|$MODEL_04B|1042|bf16"
  "04b_s2042_fp32|$MODEL_04B|2042|fp32"
  "04b_s2042_bf16|$MODEL_04B|2042|bf16"
  "15b_s42_fp32|$MODEL_15B|42|fp32"
  "15b_s42_bf16|$MODEL_15B|42|bf16"
  "15b_s1042_fp32|$MODEL_15B|1042|fp32"
  "15b_s1042_bf16|$MODEL_15B|1042|bf16"
)

i=0
n=${#GROUPS[@]}
for gspec in "${GROUPS[@]}"; do
  i=$((i+1))
  IFS='|' read -r label model seed prec <<< "$gspec"
  out="$DATA_DIR/$label"
  say ""
  say "[$i/$n] $label (seed=$seed $prec)"
  # 跳过已完成的(有 state.npz + events.jsonl)
  if [[ -f "$out/state.npz" && -f "$out/events.jsonl" ]]; then
    say "  ✓ 已存在,跳过"
    continue
  fi
  say "  + matrix_train --precision $prec --seed $seed"
  if .venv/bin/python experiments/mixed_precision/matrix_train.py \
    --model "$model" --data "$DATA" \
    --precision "$prec" --seed "$seed" \
    --out-dir "$out" \
    --cache-limit-gb 4 --epochs 2 --ctx-len 512 --lr 0.01 \
    >> "$LOG" 2>&1; then
    say "  ✓ 完成 $label"
  else
    say "  ⚠️ 失败 $label(失败也是数据,继续下一组)"
  fi
done

# ── 阶段 2:红线标定(bf16 + c4G 逐桶上探)───────────────────────────────
say ""
say "─── 阶段 2:红线标定(bf16 + c4G 逐桶上探)───"
if [[ -d experiments/mixed_precision/data/redline_buckets && \
      $(ls experiments/mixed_precision/data/redline_buckets/L*.json 2>/dev/null | wc -l) -gt 0 ]]; then
  say "+ redline_probe --model 1.5B --buckets-dir redline_buckets"
  .venv/bin/python experiments/mixed_precision/redline_probe.py \
    --model "$MODEL_15B" \
    --buckets-dir experiments/mixed_precision/data/redline_buckets \
    --out "$DATA_DIR/redline_result.json" \
    --max-steps 30 --cache-limit-gb 4 --ctx-len 600 \
    >> "$LOG" 2>&1 || say "⚠️ 红线标定失败(已跑的桶数据可能部分有效)"
  say "✓ 红线标定完成"
else
  say "⚠️ 无分桶数据,跳过红线标定"
fi

# ── 阶段 3:汇总分析 ───────────────────────────────────────────────────────
say ""
say "─── 阶段 3:汇总分析 ───"
.venv/bin/python experiments/mixed_precision/matrix_analyze.py "$DATA_DIR" 2>&1 | tee -a "$LOG"

# 写 summary
SUMMARY="experiments/mixed_precision/data/matrix_summary.txt"
{
  echo "精度矩阵 + 红线标定汇总 — $(date)"
  echo "commit: $(git rev-parse --short HEAD)"
  echo ""
  .venv/bin/python experiments/mixed_precision/matrix_analyze.py "$DATA_DIR" 2>/dev/null
} > "$SUMMARY" 2>&1 || true

say ""
say "═══════════════════════════════════════════════════════════════"
say "全部完成"
say "═══════════════════════════════════════════════════════════════"
say "产物目录: $DATA_DIR/"
say "  每组: <label>/{events.jsonl, state.npz, state.pth, decode.json, mem_trace.json}"
say "  红线: redline_result.json"
say "  汇总: matrix_summary.txt"
say "  日志: $LOG"
say ""
say "跑完告诉我,我读数据分析写报告"
