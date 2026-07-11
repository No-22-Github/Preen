#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# c4G 限额下 fp32 vs bf16 对照 — D 裁决最后一块数据
#
# 用法(项目根目录):
#   bash experiments/mixed_precision/run_c4g.sh
#
# 全程预计 30-50 分钟(两个 peak_probe 各跑 bf16+fp32 两趟,1.5B)。
# Task1: nekoqa200(同条件对照,100 步×2 趟)
# Task2: mixed 集 max=432(极限样本判决,120 步×2 趟,足够遇到 432 样本)
#
# 安全约束(需求单):Task2 先 bf16 后 fp32;session 内背靠背同进程。
# 崩了不许中途改配置——跑崩是有效结果。compressor 持续>8G / 单步>30s 即记崩。
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

MODEL_15B="${MODEL_15B:-models/converted/rwkv7-g1g-1.5b}"
DATA_NEKOQA="train_data/NekoQA_10k/nekoqa_smoke_200.json"
DATA_FULL="train_data/NekoQA_10k/NekoQA-10K.json"
DATA_DIR="experiments/mixed_precision/data"
LOG="$DATA_DIR/run_c4g.log"

mkdir -p "$DATA_DIR"
: > "$LOG"

ts() { date "+%H:%M:%S"; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }

say "═══ c4G 对照实验开始 ═══"
say "模型: $MODEL_15B"
[[ -d "$MODEL_15B" ]] || { say "❌ 模型不存在"; exit 1; }

export PYTHONPATH=src:experiments/mixed_precision:tools

# ── Task1:nekoqa200 同条件对照(100 步,bf16→fp32 同 session)──
say ""
say "══════════════════════════════════════════════════"
say "Task1:nekoqa200 c4G 同条件对照(100步 × bf16+fp32)"
say "══════════════════════════════════════════════════"
say "+ peak_probe --session bf16 fp32 (冷机便宜让给 fp32 主路线)"
echo "" >> "$LOG"
.venv/bin/python experiments/mixed_precision/peak_probe.py \
  --model "$MODEL_15B" \
  --out "$DATA_DIR/peak_task1_nekoqa200.json" \
  --data "$DATA_NEKOQA" --data-kind file \
  --max-steps 100 --n-samples 200 --cache-limit-gb 4 \
  --ctx-len 512 --seed 42 \
  --session bf16 fp32 \
  >> "$LOG" 2>&1 || { say "❌ Task1 失败"; tail -25 "$LOG"; exit 1; }
say "✓ Task1 完成"

# ── Task2:mixed 集 max=432 极限判决(120 步,bf16→fp32)──
# 432 样本在 seed42 order 第 106 步,跑 120 步确保遇到
say ""
say "══════════════════════════════════════════════════"
say "Task2:mixed 集 max=432 极限判决(120步 × bf16+fp32)"
say "先 bf16(预期更稳)再 fp32;fp32 盯前 20 步,失控即杀(已是有效数据)"
say "══════════════════════════════════════════════════"
say "+ peak_probe --data-kind mixed --session bf16 fp32"
echo "" >> "$LOG"
.venv/bin/python experiments/mixed_precision/peak_probe.py \
  --model "$MODEL_15B" \
  --out "$DATA_DIR/peak_task2_mixed.json" \
  --data "$DATA_FULL" --data-kind mixed \
  --max-steps 120 --n-samples 200 --cache-limit-gb 4 \
  --ctx-len 512 --seed 42 \
  --session bf16 fp32 \
  --label-prefix "mixed_" \
  >> "$LOG" 2>&1 || { say "⚠️ Task2 异常退出(可能是 OOM 崩溃,这是有效结果)"; }
# Task2 允许崩溃退出——不 set -e 在这里,继续往下分析

# ── 分析 ──
say ""
say "══════════════════════════════════════════════════"
say "分析结果"
say "══════════════════════════════════════════════════"
if [[ -f "$DATA_DIR/peak_task2_mixed.json" ]]; then
  .venv/bin/python experiments/mixed_precision/peak_analyze.py \
    "$DATA_DIR/peak_task1_nekoqa200.json" "$DATA_DIR/peak_task2_mixed.json" \
    2>&1 | tee -a "$LOG"
else
  say "Task2 数据文件缺失(进程可能在 fp32 阶段被杀)"
  .venv/bin/python experiments/mixed_precision/peak_analyze.py \
    "$DATA_DIR/peak_task1_nekoqa200.json" 2>&1 | tee -a "$LOG"
fi

say ""
say "═══ 全部完成 ═══"
say "产物:"
say "  Task1: $DATA_DIR/peak_task1_nekoqa200.json"
say "  Task2: $DATA_DIR/peak_task2_mixed.json (可能缺失=崩溃)"
say "  日志:   $LOG"
say "跑完告诉我,我读数据分析写结论卡"
