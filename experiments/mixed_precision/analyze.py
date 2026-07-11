"""汇总分析:loss 曲线同图 + 内存/速度对比表。

输入:各实验跑完的 events.jsonl / mem_probe JSON。
输出:纯文本报告片段(供贴进 report.md)+ 可选 ASCII loss 曲线。

用法:
  # 任务2 loss 对比
  python analyze.py loss --fp32 data/fp32_15b/events.jsonl --bf16 data/bf16_15b/events.jsonl

  # 任务3 内存对比(fp32 基线 JSON 来自 tools/mem_v2,bf16 来自本实验)
  python analyze.py mem --runs <fp32.json> <bf16.json> ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_events(path: str) -> dict:
    """读 events.jsonl → {meta, steps: [], epochs: [], final}。"""
    out = {"meta": None, "steps": [], "epochs": [], "final": None}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            t = e.get("type")
            if t == "start":
                out["meta"] = e
            elif t == "step":
                out["steps"].append(e)
            elif t == "epoch_end":
                out["epochs"].append(e)
            elif t == "final":
                out["final"] = e
    return out


def cmd_loss(args):
    fp = load_events(args.fp32)
    bf = load_events(args.bf16)

    print("=" * 70)
    print("# 任务2:loss 曲线 A/B 对比")
    print("=" * 70)

    # 元信息核对(同 seed / 同配置)
    mfp, mbf = fp["meta"], bf["meta"]
    print(f"\n## 配置核对")
    print(f"  fp32: {mfp.get('label')} precision={mfp.get('precision')} "
          f"seed={mfp.get('config',{}).get('seed')} lr={mfp.get('config',{}).get('lr')} "
          f"epochs={mfp.get('config',{}).get('epochs')} ctx={mfp.get('config',{}).get('ctx_len')}")
    print(f"  bf16: {mbf.get('label')} precision={mbf.get('precision')} "
          f"seed={mbf.get('config',{}).get('seed')} lr={mbf.get('config',{}).get('lr')} "
          f"epochs={mbf.get('config',{}).get('epochs')} ctx={mbf.get('config',{}).get('ctx_len')}")

    seed_match = (mfp.get("config", {}).get("seed") == mbf.get("config", {}).get("seed"))
    if not seed_match:
        print("  ⚠️ 警告:两版 seed 不同!loss 对比无效(需求单禁止不同 seed 对比)")

    # loss 曲线
    print(f"\n## 逐 step loss(每 {args.fp32 and fp['steps'][1]['step']-fp['steps'][0]['step'] if len(fp['steps'])>1 else '?'} step 采样)")
    print(f"{'step':>6} {'fp32':>10} {'bf16':>10} {'diff':>10}")
    n = max(len(fp["steps"]), len(bf["steps"]))
    for i in range(n):
        s_fp = fp["steps"][i] if i < len(fp["steps"]) else None
        s_bf = bf["steps"][i] if i < len(bf["steps"]) else None
        step = (s_fp or s_bf)["step"]
        l_fp = s_fp["loss"] if s_fp else float("nan")
        l_bf = s_bf["loss"] if s_bf else float("nan")
        diff = l_bf - l_fp if s_fp and s_bf else float("nan")
        print(f"{step:>6} {l_fp:>10.4f} {l_bf:>10.4f} {diff:>+10.4f}")

    # epoch 末 loss
    print(f"\n## epoch 末 loss")
    print(f"{'epoch':>6} {'fp32_avg':>10} {'bf16_avg':>10} {'diff':>10} {'diff%':>8}")
    for i in range(max(len(fp["epochs"]), len(bf["epochs"]))):
        e_fp = fp["epochs"][i] if i < len(fp["epochs"]) else None
        e_bf = bf["epochs"][i] if i < len(bf["epochs"]) else None
        ep = (e_fp or e_bf)["epoch"]
        afp = e_fp["avg_loss"] if e_fp else float("nan")
        abf = e_bf["avg_loss"] if e_bf else float("nan")
        diff = abf - afp if e_fp and e_bf else float("nan")
        pct = (diff / afp * 100) if (e_fp and e_bf and afp) else float("nan")
        print(f"{ep:>6} {afp:>10.4f} {abf:>10.4f} {diff:>+10.4f} {pct:>+7.2f}%")

    # 最终差值 + 判据
    ffp, fbf = fp["final"], bf["final"]
    if ffp and fbf:
        final_diff = fbf["final_loss"] - ffp["final_loss"]
        final_pct = final_diff / ffp["final_loss"] * 100 if ffp["final_loss"] else 0
        print(f"\n## 最终 loss 差值")
        print(f"  fp32 final: {ffp['final_loss']:.4f}")
        print(f"  bf16 final: {fbf['final_loss']:.4f}")
        print(f"  差值: {final_diff:+.4f} ({final_pct:+.2f}%)")
        abs_pct = abs(final_pct)
        if abs_pct < 2:
            verdict = "✅ 通过(loss 差 <2%)"
        elif abs_pct > 5:
            verdict = "❌ 判死(loss 差 >5%)"
        else:
            verdict = "⚠️ 中间地带(2%~5%),报裁决"
        print(f"  预写判据: {verdict}")

    # ms/step 速度对比
    print(f"\n## 速度对比(ms/step)")
    if ffp and fbf:
        print(f"  fp32: {ffp.get('ms_per_step_mean')} ms/step (mean), "
              f"{ffp.get('ms_per_step_last10')} ms/step (last10)")
        print(f"  bf16: {fbf.get('ms_per_step_mean')} ms/step (mean), "
              f"{fbf.get('ms_per_step_last10')} ms/step (last10)")
        sp = ffp.get("ms_per_step_mean", 0)
        sb = fbf.get("ms_per_step_mean", 0)
        if sp and sb:
            print(f"  bf16/fp32 = {sb/sp:.2f}x")


def cmd_mem(args):
    print("=" * 70)
    print("# 任务3:内存与速度对比")
    print("=" * 70)

    runs = []
    for p in args.runs:
        with open(p, encoding="utf-8") as f:
            runs.append(json.load(f))

    # 表头:device info
    if runs and runs[0].get("device"):
        dh = runs[0]["device"]
        print(f"\n## GPU(device_info,补丁要求)")
        print(f"  max_recommended_working_set_size = {dh.get('max_recommended_working_set_size_gb')}G "
              f"({dh.get('max_recommended_working_set_size_bytes')}B)")
        print(f"  max_buffer_length = {dh.get('max_buffer_length_gb')}G")
        print(f"  memory_size = {dh.get('memory_size_gb')}G")

    print(f"\n## 稳态内存对比(三口径 + compressor + ms/step)")
    print(f"{'label':>28} {'precision':>9} {'cache_lim':>9} | {'active':>7} {'cache':>7} {'sum':>7} "
          f"{'%ws':>5} | {'compress':>8} | {'ms/step':>7} {'capped':>6}")
    for r in runs:
        dh = r.get("device", {})
        ws = dh.get("max_recommended_working_set_size_bytes", 0)
        s = r.get("stable_active_plus_cache_gb", 0)
        pct = (s / (ws / 1e9) * 100) if ws else 0
        cl = r.get("cache_limit_gb")
        cl_str = f"{cl}G" if cl else "inf"
        capped = "是" if r.get("near_capped") else "—"
        print(f"{r['label']:>28} {r['precision']:>9} {cl_str:>9} | "
              f"{r.get('stable_active_gb',0):>6.2f}G {r.get('stable_cache_gb',0):>6.2f}G "
              f"{s:>6.2f}G {pct:>4.0f}% | "
              f"{r.get('stable_compress_gb',0):>7.2f}G | "
              f"{r.get('ms_per_step_last10',0):>6.0f} {capped:>6}")

    # 削顶判定(补丁升级判据)—— 以 bf16 无限档为对象,fp32 无限档作对照
    print(f"\n## 削顶判定(补丁升级判据)")
    bf16_inf = next((r for r in runs
                     if r.get("precision") == "bf16" and r.get("cache_limit_gb") is None), None)
    fp32_inf = next((r for r in runs
                     if r.get("precision") == "fp32" and r.get("cache_limit_gb") is None), None)
    if bf16_inf and bf16_inf.get("device"):
        ws_bytes = bf16_inf["device"].get("max_recommended_working_set_size_bytes", 0)
        ws_gb = ws_bytes / 1e9 if ws_bytes else 0
        s = bf16_inf.get("stable_active_plus_cache_gb", 0)
        if fp32_inf:
            s_fp = fp32_inf.get("stable_active_plus_cache_gb", 0)
            saved = s_fp - s
            print(f"  fp32 无限档(对照): active+cache = {s_fp:.2f}G "
                  f"({'削顶' if fp32_inf.get('near_capped') else '未削顶'})")
            print(f"  bf16 无限档(对象): active+cache = {s:.2f}G / {ws_gb:.2f}G(working_set)")
            print(f"  bf16 比 fp32 省: {saved:.2f}G")
        else:
            print(f"  bf16 无限档(对象): active+cache = {s:.2f}G / {ws_gb:.2f}G(working_set)")
        if s < ws_gb * 0.8:
            print(f"  ✅ 脱离削顶区(<80% working_set),拿到真稳态。省的量足够,D 单独有效")
        elif s < ws_gb * 0.93:
            print(f"  ⚠️ 边界区(80~93%),部分脱离,报裁决")
        else:
            print(f"  ❌ 仍贴削顶线(≥93%),省的量不够,D 单独救不动长样本,需与 C 组合")
    else:
        print("  (缺少 bf16 无限档数据,无法判定)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_loss = sub.add_parser("loss", help="loss 曲线 A/B 对比")
    ap_loss.add_argument("--fp32", required=True, help="fp32 events.jsonl")
    ap_loss.add_argument("--bf16", required=True, help="bf16 events.jsonl")

    ap_mem = sub.add_parser("mem", help="内存对比")
    ap_mem.add_argument("--runs", nargs="+", required=True, help="mem_probe JSON 路径(可多个)")

    args = ap.parse_args()
    if args.cmd == "loss":
        cmd_loss(args)
    elif args.cmd == "mem":
        cmd_mem(args)


if __name__ == "__main__":
    main()
