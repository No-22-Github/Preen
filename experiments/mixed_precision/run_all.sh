#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# 混合精度(选项 D)可行性验证 — 一键运行脚本
#
# 用法(在项目根目录):
#   bash experiments/mixed_precision/run_all.sh
#
# 可选环境变量:
#   MODEL_15B    1.5B 模型路径(默认 models/converted/rwkv7-g1g-1.5b)
#   DATA         NekoQA 数据(默认 train_data/NekoQA_10k/nekoqa_smoke_200.json)
#   SKIP_MEM=1   跳过任务3 内存探针(只跑任务2,省时间)
#
# 全程预计 60-90 分钟(6 个重任务串行,每条都是独立进程,避免 Metal 内存池累积)。
# 产物全在 experiments/mixed_precision/data/,日志在 data/run.log。
# 跑完后自动汇总,结果贴在终端 + 写入 data/summary.txt。
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── 配置 ────────────────────────────────────────────────────────────────────
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

MODEL_15B="${MODEL_15B:-models/converted/rwkv7-g1g-1.5b}"
DATA="${DATA:-train_data/NekoQA_10k/nekoqa_smoke_200.json}"
PY="PYTHONPATH=src:experiments/mixed_precision:tools .venv/bin/python"
DATA_DIR="experiments/mixed_precision/data"
SUMMARY="$DATA_DIR/summary.txt"

mkdir -p "$DATA_DIR"
# 日志覆盖写(本脚本一次完整运行)
LOG="$DATA_DIR/run.log"
: > "$LOG"

# ── 工具函数 ────────────────────────────────────────────────────────────────
ts() { date "+%H:%M:%S"; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }
run() {  # run "标题" 命令...
  local title="$1"; shift
  say "──────── $title ────────"
  say "+ $*"
  echo "" >> "$LOG"
  (set -x; "$@") >>"$LOG" 2>&1 || { say "❌ 失败: $title"; tail -20 "$LOG"; exit 1; }
  say "✓ 完成: $title"
}
run_py() {  # run_py "标题" python脚本 参数...   (stdout+stderr 都进 LOG)
  local title="$1"; shift
  say "──────── $title ────────"
  say "+ $*"
  echo "" >> "$LOG"
  env PYTHONPATH=src:experiments/mixed_precision:tools "$@" >>"$LOG" 2>&1 \
    || { say "❌ 失败: $title"; tail -25 "$LOG"; exit 1; }
  say "✓ 完成: $title"
}
run_py_redir() {  # run_py_redir "标题" "stdout文件" python脚本 参数...
  # stdout 落到独立文件(给 mem_probe 的 JSON 用),stderr(进度)进 LOG。
  # 修旧版 bug:旧版 run_py 把 stdout 也吃进 LOG,外层 >file.json 拿不到 python stdout。
  local title="$1"; shift
  local stdout_file="$1"; shift
  say "──────── $title ────────"
  say "+ $* > $stdout_file"
  echo "" >> "$LOG"
  env PYTHONPATH=src:experiments/mixed_precision:tools "$@" >"$stdout_file" 2>>"$LOG" \
    || { say "❌ 失败: $title"; tail -25 "$LOG"; exit 1; }
  say "✓ 完成: $title → $stdout_file"
}

# ── 前置检查 ────────────────────────────────────────────────────────────────
say "混合精度(选项D)可行性验证 — 开始"
say "模型: $MODEL_15B"
say "数据: $DATA"
[[ -d "$MODEL_15B" ]] || { say "❌ 模型目录不存在: $MODEL_15B"; exit 1; }
[[ -f "$DATA" ]] || { say "❌ 数据文件不存在: $DATA"; exit 1; }
[[ -x .venv/bin/python ]] || { say "❌ .venv/bin/python 不存在"; exit 1; }

# ── 任务 1:数值等价性冒烟(秒级,已在脚本里跑过,这里复跑确认当前机器)──────
say ""
say "══════════════════════════════════════════════════"
say "任务 1:数值等价性冒烟(1.5B 配置 H=32,273 步)"
say "══════════════════════════════════════════════════"
run_py "任务1 冒烟" .venv/bin/python experiments/mixed_precision/smoke_numeric.py \
  --steps 273 --heads 32 --head-dim 64 --seed 42 --record-every 1 \
  --out "$DATA_DIR/smoke_15b_h32.json"

# ── 任务 2:训练质量 A/B ────────────────────────────────────────────────────
say ""
say "══════════════════════════════════════════════════"
say "任务 2:训练质量 A/B(1.5B + 200条 + 2epoch + seed42)"
say "══════════════════════════════════════════════════"
run_py "任务2a fp32 训练" .venv/bin/python experiments/mixed_precision/ab_train.py \
  --model "$MODEL_15B" --data "$DATA" \
  --precision fp32 --label fp32_15b \
  --out-dir "$DATA_DIR/fp32_15b" \
  --epochs 2 --ctx-len 512 --seed 42 --lr 0.01 --log-every 5

run_py "任务2b bf16 训练" .venv/bin/python experiments/mixed_precision/ab_train.py \
  --model "$MODEL_15B" --data "$DATA" \
  --precision bf16 --label bf16_15b \
  --out-dir "$DATA_DIR/bf16_15b" \
  --epochs 2 --ctx-len 512 --seed 42 --lr 0.01 --log-every 5

run_py "任务2c 十问解码 A/B + state 距离" .venv/bin/python experiments/mixed_precision/decode_compare.py \
  --model "$MODEL_15B" \
  --fp32-state "$DATA_DIR/fp32_15b/state.npz" \
  --bf16-state "$DATA_DIR/bf16_15b/state.npz" \
  --out "$DATA_DIR/decode_compare.json"

run_py "任务2d loss 曲线汇总" .venv/bin/python experiments/mixed_precision/analyze.py loss \
  --fp32 "$DATA_DIR/fp32_15b/events.jsonl" \
  --bf16 "$DATA_DIR/bf16_15b/events.jsonl"

# ── 任务 3:内存与速度账(可跳过)────────────────────────────────────────────
if [[ "${SKIP_MEM:-0}" != "1" ]]; then
  say ""
  say "══════════════════════════════════════════════════"
  say "任务 3:内存与速度账(1.5B nekoqa200,100 步)"
  say "══════════════════════════════════════════════════"

  # 3a:bf16 无限档(判脱离削顶的关键)——stdout(JSON)落独立文件,stderr 进 LOG
  run_py_redir "任务3a bf16 无限档" "$DATA_DIR/bf16_15b_inf.json" \
    .venv/bin/python experiments/mixed_precision/mem_probe_bf16.py \
    --model "$MODEL_15B" --precision bf16 --label bf16_15b_inf \
    --data "$DATA" --max-steps 100

  # 3b:bf16 c4G 档
  run_py_redir "任务3b bf16 c4G 档" "$DATA_DIR/bf16_15b_c4.json" \
    .venv/bin/python experiments/mixed_precision/mem_probe_bf16.py \
    --model "$MODEL_15B" --precision bf16 --label bf16_15b_c4 --cache-limit-gb 4 \
    --data "$DATA" --max-steps 100

  # 3c:fp32 无限档(同机器基线对照,因为这台机器 working_set 与 v2 报告不同)
  run_py_redir "任务3c fp32 无限档(对照)" "$DATA_DIR/fp32_15b_inf.json" \
    .venv/bin/python experiments/mixed_precision/mem_probe_bf16.py \
    --model "$MODEL_15B" --precision fp32 --label fp32_15b_inf \
    --data "$DATA" --max-steps 100

  # 内存汇总
  run_py "任务3 内存汇总" .venv/bin/python experiments/mixed_precision/analyze.py mem \
    --runs "$DATA_DIR/fp32_15b_inf.json" "$DATA_DIR/bf16_15b_inf.json" "$DATA_DIR/bf16_15b_c4.json"
else
  say "SKIP_MEM=1,跳过任务3"
fi

# ── 汇总 ────────────────────────────────────────────────────────────────────
say ""
say "══════════════════════════════════════════════════"
say "全部完成 — 关键产物"
say "══════════════════════════════════════════════════"
{
  echo "混合精度(选项 D)实验产物清单 — $(date)"
  echo "GPU working_set: $(.venv/bin/python -c 'import mlx.core as mx; print(round(mx.metal.device_info()["max_recommended_working_set_size"]/1024**3,2), "G")' 2>/dev/null || echo '?')"
  echo ""
  echo "[任务1 冒烟]  $DATA_DIR/smoke_15b_h32.json"
  echo "[任务2 fp32]  $DATA_DIR/fp32_15b/{events.jsonl, state.npz}"
  echo "[任务2 bf16]  $DATA_DIR/bf16_15b/{events.jsonl, state.npz}"
  echo "[任务2 解码]  $DATA_DIR/decode_compare.json"
  echo "[任务3 内存]  $DATA_DIR/{fp32_15b_inf, bf16_15b_inf, bf16_15b_c4}.json"
  echo ""
  echo "完整日志: $LOG"
  echo ""
  echo "── loss 曲线 A/B(从 events 抽取)──"
  .venv/bin/python experiments/mixed_precision/analyze.py loss \
    --fp32 "$DATA_DIR/fp32_15b/events.jsonl" \
    --bf16 "$DATA_DIR/bf16_15b/events.jsonl" 2>/dev/null || echo "(loss 汇总失败,见 run.log)"
  echo ""
  if [[ -f "$DATA_DIR/fp32_15b_inf.json" ]]; then
    echo "── 内存对比 ──"
    .venv/bin/python experiments/mixed_precision/analyze.py mem \
      --runs "$DATA_DIR/fp32_15b_inf.json" "$DATA_DIR/bf16_15b_inf.json" "$DATA_DIR/bf16_15b_c4.json" 2>/dev/null || echo "(内存汇总失败,见 run.log)"
  fi
} | tee "$SUMMARY"

say "汇总写入 $SUMMARY — 把它和 run.log 贴回来即可"
