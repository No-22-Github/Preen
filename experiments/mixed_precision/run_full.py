#!/usr/bin/env .venv/bin/python
"""大满贯编排器(Python 版,替代 run_full.sh)。

为什么用 Python 不用 shell:shell 的 IFS/引号/变量展开在复杂数组解析上易出 bug
(本会话踩过),Python 的 subprocess + 字符串管理更可靠。

四阶段:
  0. 分桶数据集 + 延长数值冒烟(真实w,700步) + w分布dump
  1. 矩阵训练 10 组(主体,每组 subprocess.run 独立进程)
  2. 红线标定(bf16+c4G 逐桶上探)
  3. 汇总分析

用法(项目根目录):
  caffeinate -i .venv/bin/python experiments/mixed_precision/run_full.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
EXP = ROOT / "experiments" / "mixed_precision"
DATA_DIR = EXP / "data"
MATRIX_DIR = DATA_DIR / "matrix"
LOG_PATH = DATA_DIR / "matrix_run.log"

MODEL_04B = ROOT / "models" / "converted" / "rwkv7-g1d-0.4b"
MODEL_15B = ROOT / "models" / "converted" / "rwkv7-g1g-1.5b"
DATA_NEKOQA = ROOT / "train_data" / "NekoQA_10k" / "nekoqa_smoke_200.json"
DATA_30K = ROOT / "train_data" / "NekoQA_30k" / "MaoDieQA-30K.json"

ENV = {**os.environ, "PYTHONPATH": "src:experiments/mixed_precision:tools"}

# 矩阵 10 组:(label, model, seed, precision)
GROUPS = [
    ("04b_s42_fp32", MODEL_04B, 42, "fp32"),
    ("04b_s42_bf16", MODEL_04B, 42, "bf16"),
    ("04b_s1042_fp32", MODEL_04B, 1042, "fp32"),
    ("04b_s1042_bf16", MODEL_04B, 1042, "bf16"),
    ("04b_s2042_fp32", MODEL_04B, 2042, "fp32"),
    ("04b_s2042_bf16", MODEL_04B, 2042, "bf16"),
    ("15b_s42_fp32", MODEL_15B, 42, "fp32"),
    ("15b_s42_bf16", MODEL_15B, 42, "bf16"),
    ("15b_s1042_fp32", MODEL_15B, 1042, "fp32"),
    ("15b_s1042_bf16", MODEL_15B, 1042, "bf16"),
]

LOG_LINES = []


def log(msg: str):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)


def run_step(title: str, cmd: list, cwd=ROOT, check_fail=True, allow_fail_msg=None):
    """跑一个子进程,stdout/stderr 进 LOG,失败不 crash 整体(除非 check_fail)。"""
    log(f"── {title} ──")
    cmd_str = " ".join(str(c) for c in cmd)
    log(f"  + {cmd_str}")
    LOG_LINES.append("")  # 空行分隔
    proc = subprocess.run(
        [str(c) for c in cmd], cwd=cwd, env=ENV,
        capture_output=True, text=True,
    )
    # 子进程输出追加进 LOG
    if proc.stdout:
        LOG_LINES.append(proc.stdout.rstrip())
    if proc.stderr:
        LOG_LINES.append(proc.stderr.rstrip())
    LOG_LINES.append("")

    if proc.returncode != 0:
        msg = allow_fail_msg or f"⚠️ 失败: {title}(returncode={proc.returncode})"
        log(msg)
        # 打印最后几行 stderr 帮助排查
        if proc.stderr:
            for line in proc.stderr.strip().splitlines()[-5:]:
                log(f"    {line}")
        if check_fail:
            flush_log()
            sys.exit(1)
        return False
    log(f"  ✓ 完成: {title}")
    return True


def flush_log():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(LOG_LINES) + "\n")
    LOG_LINES.clear()


def main():
    t0 = time.time()

    # 前置检查
    log("═══════════════════════════════════════════════════════════════")
    log("大满贯:精度矩阵 + 红线标定  开始")
    log("═══════════════════════════════════════════════════════════════")
    log(f"0.4B: {MODEL_04B}")
    log(f"1.5B: {MODEL_15B}")
    log(f"数据: {DATA_NEKOQA}")
    for m in (MODEL_04B, MODEL_15B):
        if not m.is_dir():
            log(f"❌ 模型不存在: {m}"); flush_log(); sys.exit(1)
    if not DATA_NEKOQA.is_file():
        log(f"❌ 数据不存在: {DATA_NEKOQA}"); flush_log(); sys.exit(1)
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                     cwd=ROOT, text=True).strip()
    log(f"commit: {commit}")
    flush_log()

    PY = str(ROOT / ".venv" / "bin" / "python")

    # ── 阶段 0a:分桶数据集 ──
    log("")
    log("─── 阶段 0a:分桶数据集准备 ──")
    run_step("分桶", [
        PY, str(EXP / "build_buckets.py"),
        "--source", str(DATA_30K),
        "--model", str(MODEL_15B),
        "--targets", "450,550,600,650,700", "--per-bucket", "40",
        "--out-dir", str(DATA_DIR / "redline_buckets"),
    ], check_fail=False, allow_fail_msg="⚠️ 分桶失败(红线标定将跳过)")
    flush_log()

    # ── 阶段 0b:延长数值冒烟(真实w,700步)──
    log("")
    log("─── 阶段 0b:延长数值冒烟(700 步,真实 w 分布)───")
    run_step("延长冒烟", [
        PY, str(EXP / "smoke_numeric.py"),
        "--steps", "700", "--heads", "32", "--head-dim", "64",
        "--seed", "42", "--record-every", "5",
        "--real-w-model", str(MODEL_15B),
        "--out", str(DATA_DIR / "smoke_15b_700_realw.json"),
    ], check_fail=False, allow_fail_msg="⚠️ 延长冒烟失败")
    flush_log()

    # ── 阶段 0c:w 分布 dump ──
    log("")
    log("─── 阶段 0c:w 分布 dump ──")
    run_step("w dump 0.4B", [
        PY, str(EXP / "dump_w_dist.py"),
        "--model", str(MODEL_04B), "--data", str(DATA_NEKOQA),
        "--out", str(DATA_DIR / "w_dist_04b.json"),
    ], check_fail=False, allow_fail_msg="⚠️ 0.4B w dump 失败")
    run_step("w dump 1.5B", [
        PY, str(EXP / "dump_w_dist.py"),
        "--model", str(MODEL_15B), "--data", str(DATA_NEKOQA),
        "--out", str(DATA_DIR / "w_dist_15b.json"),
    ], check_fail=False, allow_fail_msg="⚠️ 1.5B w dump 失败")
    flush_log()

    # ── 阶段 1:矩阵训练 10 组 ──
    log("")
    log("─── 阶段 1:矩阵训练(10 组,每组独立进程)───")
    MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    n_total = len(GROUPS)
    for i, (label, model, seed, prec) in enumerate(GROUPS, 1):
        out = MATRIX_DIR / label
        log("")
        # 断点续跑:已完成(有 state.npz + events.jsonl)则跳过
        if (out / "state.npz").exists() and (out / "events.jsonl").exists():
            log(f"[{i}/{n_total}] {label} — ✓ 已存在,跳过")
            flush_log()
            continue
        log(f"[{i}/{n_total}] {label} (seed={seed} {prec})")
        run_step(
            f"训练 {label}",
            [PY, str(EXP / "matrix_train.py"),
             "--model", str(model), "--data", str(DATA_NEKOQA),
             "--precision", prec, "--seed", str(seed),
             "--out-dir", str(out),
             "--cache-limit-gb", "4", "--epochs", "2",
             "--ctx-len", "512", "--lr", "0.01"],
            check_fail=False,
            allow_fail_msg=f"⚠️ 失败 {label}(失败也是数据,继续下一组)",
        )
        flush_log()

    # ── 阶段 2:红线标定 ──
    log("")
    log("─── 阶段 2:红线标定(bf16 + c4G 逐桶上探)───")
    buckets = list((DATA_DIR / "redline_buckets").glob("L*.json")) if (DATA_DIR / "redline_buckets").exists() else []
    if buckets:
        run_step("红线标定", [
            PY, str(EXP / "redline_probe.py"),
            "--model", str(MODEL_15B),
            "--buckets-dir", str(DATA_DIR / "redline_buckets"),
            "--out", str(MATRIX_DIR / "redline_result.json"),
            "--max-steps", "30", "--cache-limit-gb", "4", "--ctx-len", "600",
        ], check_fail=False, allow_fail_msg="⚠️ 红线标定失败(已跑的桶数据可能部分有效)")
    else:
        log("⚠️ 无分桶数据,跳过红线标定")
    flush_log()

    # ── 阶段 3:汇总分析 ──
    log("")
    log("─── 阶段 3:汇总分析 ──")
    analyze = subprocess.run(
        [PY, str(EXP / "matrix_analyze.py"), str(MATRIX_DIR)],
        cwd=ROOT, env={**os.environ, **ENV}, capture_output=True, text=True,
    )
    analysis = (analyze.stdout + analyze.stderr).strip()
    log("")  # 让分析输出和标题分开
    for line in analysis.splitlines():
        print(line, flush=True)
        LOG_LINES.append(line)
    LOG_LINES.append("")

    # 写 summary
    summary_path = DATA_DIR / "matrix_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"精度矩阵 + 红线标定汇总 — {datetime.now()}\n")
        f.write(f"commit: {commit}\n\n")
        f.write(analysis)
    log(f"汇总写入 {summary_path}")
    flush_log()

    elapsed = time.time() - t0
    log("")
    log("═══════════════════════════════════════════════════════════════")
    log(f"全部完成 — 耗时 {elapsed/60:.0f} 分钟")
    log("═══════════════════════════════════════════════════════════════")
    log(f"产物目录: {MATRIX_DIR}/")
    log("  每组: <label>/{events.jsonl, state.npz, state.pth, decode.json, mem_trace.json}")
    log("  红线: redline_result.json")
    log(f"  汇总: {summary_path}")
    log(f"  日志: {LOG_PATH}")
    log("")
    log("跑完告诉我(或新会话读 HANDOFF.md),我读数据分析写报告")
    flush_log()


if __name__ == "__main__":
    main()
