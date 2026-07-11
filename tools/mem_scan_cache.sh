#!/usr/bin/env bash
# 任务3: 1.5B cache_limit 扫描 (nekoqa200 配置, 全部100步同口径)
# 扫 {4,6,8,10,inf}, 报稳态cache/compressor/ms(末10)
set -e
cd /Users/no22/Projects/Preen
mkdir -p tools/mem_v2

MODEL="models/converted/rwkv7-g1g-1.5b"
DATA="train_data/NekoQA_10k/nekoqa_smoke_200.json"

for CL in 4 6 8 10; do
  LABEL="15b_scan_c${CL}"
  if [ -f "tools/mem_v2/${LABEL}.log" ] && grep -q "DONE" "tools/mem_v2/${LABEL}.log"; then
    echo "━━━ c${CL}G 已完成, 跳过 ━━━"
    grep "DONE" tools/mem_v2/${LABEL}.log
    continue
  fi
  echo "━━━ cache_limit=${CL}G (100步) ━━━"
  PYTHONPATH=src .venv/bin/python tools/mem_probe_v2.py \
    --model "$MODEL" --label "$LABEL" \
    --data file --data "$DATA" \
    --max-steps 100 --cache-limit-gb $CL \
    > "tools/mem_v2/${LABEL}.json" 2>"tools/mem_v2/${LABEL}.log"
  grep "DONE" tools/mem_v2/${LABEL}.log
  echo ""
done
echo "===== inf 档 (复用任务2 nekoqa200_inf, 但那是80步; 重跑100步保证同口径) ====="
LABEL="15b_scan_cinf"
PYTHONPATH=src .venv/bin/python tools/mem_probe_v2.py \
  --model "$MODEL" --label "$LABEL" \
  --data file --data "$DATA" \
  --max-steps 100 \
  > "tools/mem_v2/${LABEL}.json" 2>"tools/mem_v2/${LABEL}.log"
grep "DONE" tools/mem_v2/${LABEL}.log
echo ""
echo "===== 扫描完成 ====="
