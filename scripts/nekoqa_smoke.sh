#!/usr/bin/env bash
# NekoQA × 1.5B state tuning —— smoke 全流程脚本
#
# 用途:验证 NekoQA 数据接入管线端到端通,并直观对比 state 注入前后的风格效果。
# 不是正式训练(正式全量 10k 需 ~17 小时);这里只跑小子集,拿一个能用的 state。
#
# 做四件事:
#   1. 准备 smoke 数据子集(默认 200 条,可改 SAMPLES)
#   2. 训练 state(lr=0.0001, 2 epoch, ~10 分钟)
#   3. 用训好的 state 跑 5 条 held-out 生成(看风格是否注入)
#   4. A/B 对比:同一问题,无 state(基线) vs 有 state(猫娘)
#
# 用法:
#   bash scripts/nekoqa_smoke.sh              # 默认 200 条
#   SAMPLES=500 bash scripts/nekoqa_smoke.sh  # 自定义样本数
#   SKIP_TRAIN=1 bash scripts/nekoqa_smoke.sh # 跳过训练,只用已有 state 跑生成
#
# 前置:
#   - 仓库根目录运行(脚本会自动 cd 到自己所在目录的上一级)
#   - .venv 已装好依赖(mlx, mlx-lm, torch, typer)
#   - models/converted/rwkv7-g1g-1.5b/ 存在(已转换的 HF 格式)
#   - train_data/NekoQA_10k/NekoQA-10K.json 存在(完整数据集)
set -euo pipefail

# ── 自动定位仓库根(脚本在 scripts/ 下,根是上一级)──────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ── 配置(可用环境变量覆盖)────────────────────────────────
MODEL="models/converted/rwkv7-g1g-1.5b"        # 1.5B 已转换模型
FULL_DATA="train_data/NekoQA_10k/NekoQA-10K.json"  # 完整数据集
DATA_DIR="train_data/NekoQA_10k"
SAMPLES="${SAMPLES:-200}"                       # smoke 样本数
EPOCHS="${EPOCHS:-2}"                           # epoch 数
LR="${LR:-0.0001}"
CTX_LEN="${CTX_LEN:-512}"
STATE_OUT="$DATA_DIR/nekoqa_smoke_state.npz"
EVENTS_OUT="$DATA_DIR/nekoqa_smoke_events.jsonl"
SMOKE_DATA="$DATA_DIR/nekoqa_smoke_${SAMPLES}.json"

PYTHON="${PYTHON:-.venv/bin/python}"
export PYTHONPATH="src"

# ── 前置检查 ────────────────────────────────────────────────
echo "============================================================"
echo " NekoQA × 1.5B state tuning —— smoke 全流程"
echo "============================================================"
echo " 仓库根:   $REPO_ROOT"
echo " 模型:     $MODEL"
echo " 完整数据: $FULL_DATA"
echo " smoke:    $SAMPLES 样本 × $EPOCHS epoch, lr=$LR, ctx=$CTX_LEN"
echo "============================================================"
echo

[ -f "$PYTHON" ] || { echo "✗ 找不到 $PYTHON (先建 venv 并装依赖)"; exit 1; }
[ -d "$MODEL" ] || { echo "✗ 找不到模型目录 $MODEL"; exit 1; }
[ -f "$FULL_DATA" ] || { echo "✗ 找不到数据 $FULL_DATA"; exit 1; }

# ── 步骤 1: 准备 smoke 子集 ─────────────────────────────────
if [ -f "$SMOKE_DATA" ]; then
    n=$($PYTHON -c "import json; print(len(json.load(open('$SMOKE_DATA'))))")
    echo "① smoke 数据已存在: $SMOKE_DATA ($n 条),跳过生成"
else
    echo "① 生成 smoke 子集: 取前 $SAMPLES 条 → $SMOKE_DATA"
    $PYTHON -c "
import json
with open('$FULL_DATA', encoding='utf-8') as f:
    data = json.load(f)
subset = data[:$SAMPLES]
with open('$SMOKE_DATA', 'w', encoding='utf-8') as f:
    json.dump(subset, f, ensure_ascii=False)
print(f'   写入 {len(subset)} 条')
"
fi
echo

# ── 步骤 2: 训练 ────────────────────────────────────────────
if [ "${SKIP_TRAIN:-0}" = "1" ]; then
    echo "② 跳过训练(SKIP_TRAIN=1)"
    [ -f "$STATE_OUT" ] || { echo "✗ SKIP_TRAIN=1 但 $STATE_OUT 不存在"; exit 1; }
else
    echo "② 训练 state($SAMPLES × $EPOCHS epoch, lr=$LR)..."
    echo "   预估耗时:~$(echo "$SAMPLES * $EPOCHS * 2 / 60" | bc) 分钟(按 ~2s/step)"
    echo "   事件流 → $EVENTS_OUT"
    echo
    $PYTHON -m statetuner.cli train \
        --model "$MODEL" \
        --data "$SMOKE_DATA" \
        --template nekoqa \
        --out "$STATE_OUT" \
        --events-file "$EVENTS_OUT" \
        --lr "$LR" --epochs "$EPOCHS" --ctx-len "$CTX_LEN" --warmup 10 \
        --no-early-stop --seed 42
    echo
    echo "   ✓ state → $STATE_OUT"
fi
echo

# ── 步骤 3: 看 loss 曲线(确认收敛)──────────────────────────
echo "③ 训练 loss 曲线(epoch_end 汇总):"
if [ -f "$EVENTS_OUT" ]; then
    $PYTHON -c "
import json
with open('$EVENTS_OUT', encoding='utf-8') as f:
    for line in f:
        ev = json.loads(line)
        if ev['type'] == 'epoch_end':
            print(f'   epoch {ev[\"epoch\"]}: loss={ev[\"loss\"]:.4f}  state_std={ev[\"state_std\"]:.4f}  lr={ev[\"lr\"]:.5f}')
        elif ev['type'] == 'final':
            print(f'   final: elapsed={ev.get(\"elapsed\",0):.0f}s')
"
else
    echo "   (无事件文件)"
fi
echo

# ── 步骤 4: held-out 生成(看风格注入)──────────────────────
echo "④ held-out 生成(用 state,看猫娘风格):"
$PYTHON -m statetuner.cli eval \
    --model "$MODEL" \
    --state "$STATE_OUT" \
    --template nekoqa \
    --max-tokens 120 \
    --limit 5
echo

# ── 步骤 5: A/B 对比(同一问题,无 state vs 有 state)────────
echo "⑤ A/B 对比(基线 vs 注入 state):"
AB_PROMPT="主人，今天好累哦，能哄哄我吗？"
echo "   问题: $AB_PROMPT"
echo
echo "   --- 无 state(基线:原始 1.5B)---"
$PYTHON -m statetuner.cli preview \
    --model "$MODEL" \
    --prompt "$AB_PROMPT" \
    --template nekoqa \
    --max-tokens 120
echo
echo "   --- 有 state(注入训练后的 S₀)---"
$PYTHON -m statetuner.cli preview \
    --model "$MODEL" \
    --state "$STATE_OUT" \
    --prompt "$AB_PROMPT" \
    --template nekoqa \
    --max-tokens 120
echo

echo "============================================================"
echo " 完成。把以上输出贴回去,我看效果决定下一步。"
echo " state: $STATE_OUT"
echo " events: $EVENTS_OUT"
echo "============================================================"
