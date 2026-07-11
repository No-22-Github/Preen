"""peak_probe 结果分析:Task1 四项交付 + 速度异常 + Task2 判决矩阵。

用法:
  python peak_analyze.py <peak_probe.json> [<task2_mixed.json>]
  - 单参数:Task1 分析(同条件对照)
  - 双参数:附加 Task2 极限样本判决
"""
from __future__ import annotations

import json
import sys


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_run(data, precision):
    for r in data["runs"]:
        if r["precision"] == precision:
            return r
    return None


def analyze_task1(data):
    """Task1 同条件对照:散点数据 / 最长样本峰值差 / compressor / 距天花板余量。"""
    dh = data["device"]
    ws_gb = dh["max_recommended_working_set_size_gb"]
    cap_line = ws_gb * 0.95

    bf = find_run(data, "bf16")
    fp = find_run(data, "fp32")
    if not bf or not fp:
        print("⚠️ 缺少 bf16 或 fp32 run"); return

    print("=" * 72)
    print("# Task1:c4G 限额下 fp32 vs bf16 同条件对照")
    print("=" * 72)
    print(f"\n机型:working_set={ws_gb}G max_buffer={dh['max_buffer_length_gb']}G "
          f"commit={data.get('commit','?')}")
    print(f"配置:{data['config']['data_kind']} max_steps={data['config']['max_steps']} "
          f"cache_limit={data['config']['cache_limit_gb']}G seed={data['config']['seed']}")

    # (b) 靶心:最长样本那一步的峰值对比
    print(f"\n## (b) 最长样本步峰值对比【整单靶心】")
    fp_max = fp["max_sample_step"]
    bf_max = bf["max_sample_step"]
    # 注意:两趟 order 相同,最长样本应是同一个;核对 token_len 一致
    print(f"  最长样本 token 数:fp32={fp_max['token_len']}  bf16={bf_max['token_len']}")
    if fp_max["token_len"] != bf_max["token_len"]:
        print(f"  ⚠️ 两趟最长样本不一致(可能 order 不同),峰值差可比性下降")
    peak_diff = fp_max["step_peak_gb"] - bf_max["step_peak_gb"]
    print(f"  该步 step_peak:fp32={fp_max['step_peak_gb']}G  bf16={bf_max['step_peak_gb']}G")
    print(f"  ★ 峰值差(D 的真实余量):{peak_diff:+.3f}G")
    print()
    if peak_diff >= 1.0:
        print(f"  → 余量差 ≥1G,与机制推算(~1.4G)吻合,反向图 fp32 部分确实占大头")
        print(f"  → 「抬红线」故事讲得成,继续看 Task2")
    elif peak_diff >= 0.3:
        print(f"  → 余量差 {peak_diff:.2f}G,中等(机制推算的 ~1.4G 未完全兑现)")
        print(f"  → 边界,需结合 Task2 极限判决")
    else:
        print(f"  → 余量差仅 {peak_diff:.2f}G,远小于机制推算 ~1.4G")
        print(f"  → 反向图 fp32 部分没占想象那么大头,「抬红线」讲不成")
        print(f"  → Task2 大概率两版都正常,裁决倾向封存")

    # (a) 散点数据(token_len vs step_peak)——列出长样本步(top 10 by token_len)
    print(f"\n## (a) 长样本步的 step_peak 对照(top 10 by token_len)")
    print(f"{'token_len':>9} | {'fp32 step_peak':>14} {'bf16 step_peak':>14} {'diff':>8} | {'step':>4}")
    # 按 token_len 排序(用 fp32 的 trace),取 top 10
    fp_by_len = sorted(fp["trace"], key=lambda r: -r["token_len"])[:10]
    bf_trace = {r["step"]: r for r in bf["trace"]}
    for r in fp_by_len:
        br = bf_trace.get(r["step"])
        if br:
            d = r["step_peak_gb"] - br["step_peak_gb"]
            print(f"{r['token_len']:>9} | {r['step_peak_gb']:>13.3f}G {br['step_peak_gb']:>13.3f}G "
                  f"{d:>+7.3f}G | {r['step']:>4}")

    # (c) compressor 曲线
    print(f"\n## (c) compressor 全程")
    fp_comp = [r["compress_gb"] for r in fp["trace"]]
    bf_comp = [r["compress_gb"] for r in bf["trace"]]
    print(f"  fp32: max={max(fp_comp):.2f}G mean={sum(fp_comp)/len(fp_comp):.2f}G")
    print(f"  bf16: max={max(bf_comp):.2f}G mean={sum(bf_comp)/len(bf_comp):.2f}G")
    # 瞬时冲高检测:某步 compressor 比前后步高 >1G
    for label, comp in [("fp32", fp_comp), ("bf16", bf_comp)]:
        for i in range(1, len(comp) - 1):
            if comp[i] - comp[i-1] > 1.0 and comp[i] - comp[i+1] > 1.0:
                print(f"  ⚠️ {label} step{i} compressor 瞬时冲高:{comp[i]:.2f}G(前后 {comp[i-1]:.2f}/{comp[i+1]:.2f})")

    # (d) 距天花板余量
    print(f"\n## (d) 距削顶线(95% working_set={cap_line:.2f}G)余量")
    fp_maxsum = fp["overall"]["max_active_plus_cache_gb"]
    bf_maxsum = bf["overall"]["max_active_plus_cache_gb"]
    print(f"  fp32 active+cache 全程最大:{fp_maxsum:.2f}G  距天花板:{cap_line-fp_maxsum:+.2f}G")
    print(f"  bf16 active+cache 全程最大:{bf_maxsum:.2f}G  距天花板:{cap_line-bf_maxsum:+.2f}G")
    print(f"  bf16 比 fp32 多出的天花板余量:{(cap_line-bf_maxsum)-(cap_line-fp_maxsum):+.2f}G")

    # 速度异常检测
    print(f"\n## 速度异常检测(只报异常,不进结论)")
    fp_ms = fp["overall"]["ms_per_step_mean"]
    bf_ms = bf["overall"]["ms_per_step_mean"]
    ratio = bf_ms / fp_ms if fp_ms else 0
    slowdown = (ratio - 1) * 100
    print(f"  fp32 ms/step={fp_ms}  bf16 ms/step={bf_ms}  bf16/fp32={ratio:.2f}x ({slowdown:+.0f}%)")
    if slowdown > 30:
        print(f"  🔴 bf16 劣化超 30%,实现可能有 bug,报裁决")
    else:
        print(f"  无异常(bf16 劣化 <30%)")


def analyze_task2(data):
    """Task2 极限样本判决:mixed 集 max=432,c4G。"""
    dh = data["device"]
    ws_gb = dh["max_recommended_working_set_size_gb"]

    print("\n" + "=" * 72)
    print("# Task2:极限样本判决(mixed 集 max=432,c4G)")
    print("=" * 72)

    bf = find_run(data, "bf16")
    fp = find_run(data, "fp32")

    # 确认 432 样本已被训过
    def max_token_seen(run):
        return max(r["token_len"] for r in run["trace"]) if run else 0
    bf_seen = max_token_seen(bf) if bf else 0
    fp_seen = max_token_seen(fp) if fp else 0
    print(f"\n  实际遇到的最长样本:bf16={bf_seen}  fp32={fp_seen}")
    if max(bf_seen, fp_seen) < 400:
        print(f"  ⚠️ 未遇到 ~432 样本(max_seen={max(bf_seen,fp_seen)}),判决依据不足")

    # "崩"的定义:进程被杀(数据缺失)/ compressor 持续>8G / 单步>30s
    def crash_check(run, label):
        if run is None:
            return f"  {label}: 无数据(进程可能被杀)→ 崩"
        comp_max = run["overall"]["max_compress_gb"]
        ms_max = max(r["ms"] for r in run["trace"])
        # compressor 持续 >8G:末10步均值
        tail_comp = [r["compress_gb"] for r in run["trace"][-10:]]
        comp_sustained = (sum(tail_comp)/len(tail_comp)) if tail_comp else 0
        reasons = []
        if comp_sustained > 8:
            reasons.append(f"compressor 末10步均值 {comp_sustained:.1f}G >8G")
        if ms_max > 30000:
            reasons.append(f"单步最大 {ms_max:.0f}ms >30s")
        status = "崩(" + "; ".join(reasons) + ")" if reasons else "正常"
        return comp_max, ms_max, comp_sustained, status

    print(f"\n## 崩溃检测(判据:进程被杀 / compressor 持续>8G / 单步>30s)")
    for run, label in [(bf, "bf16"), (fp, "fp32")]:
        if run is None:
            print(f"  {label}: 无数据(进程被杀)→ 崩"); continue
        cm, mm, cs, status = crash_check(run, label)
        print(f"  {label}: compressor_max={cm:.1f}G 单步max={mm:.0f}ms "
              f"compressor末10步均值={cs:.1f}G → {status}")

    # 判决矩阵 —— 严格按需求单判据:是否崩(compressor 持续>8G / 单步>30s / 进程被杀)
    # 注:实验中发现该判据有盲点(step_peak 逼近削顶线但未崩时,判据不敏感)。
    # 按 AGENTS.md 流程规矩,不自行新增判据,只报客观数据 + 提请裁决。
    print(f"\n## 判决矩阵(判据:是否崩)")
    bf_ok = bf is not None and "正常" in crash_check(bf, "bf16")[3]
    fp_ok = fp is not None and "正常" in crash_check(fp, "fp32")[3]

    if bf_ok and not fp_ok:
        print(f"  → fp32 崩/严重换页,bf16 正常 → ★ 红线可抬,D 进预检策略")
    elif not bf_ok and not fp_ok:
        print(f"  → 两版都崩 → max 432 超出 16GB c4G 能力,红线维持 ~300")
        print(f"  → D 封存")
    elif bf_ok and fp_ok:
        print(f"  → 两版都没崩 → 按需求单判决矩阵:273~432 区间 c4G 都扛得住,红线本就保守 → 封存")
    else:
        print(f"  → bf16 崩 fp32 正常(反常,实现可能有 bug)→ 报裁决核查")

    # 客观补充数据(step_peak 逼近削顶线),供裁决参考 —— 不改判据,只呈现
    ws_gb = dh["max_recommended_working_set_size_gb"]
    cap = ws_gb * 0.95
    print(f"\n## 客观补充:step_peak 逼近削顶线(供裁决参考,非判据)")
    print(f"  削顶线(95% working_set)= {cap:.2f}G")
    for run, label in [(bf, "bf16"), (fp, "fp32")]:
        if run is None:
            continue
        # 找 step_peak 最大的步
        peak_step = max(run["trace"], key=lambda r: r["step_peak_gb"])
        pct = peak_step["step_peak_gb"] / ws_gb * 100
        mspt = peak_step["ms"] / peak_step["token_len"]
        print(f"  {label}: 最长步 token={peak_step['token_len']} "
              f"step_peak={peak_step['step_peak_gb']}G({pct:.0f}%ws) "
              f"ms={peak_step['ms']:.0f} ms/tok={mspt:.1f}")
    print(f"  (fp32 若逼近削顶线且 ms/tok 非线性飙升,说明已在硬撑;此为提请裁决项,非自动判据)")


def main():
    if len(sys.argv) < 2:
        print("用法: python peak_analyze.py <task1.json> [<task2.json>]")
        sys.exit(1)
    data = load(sys.argv[1])
    analyze_task1(data)
    if len(sys.argv) >= 3:
        data2 = load(sys.argv[2])
        analyze_task2(data2)


if __name__ == "__main__":
    main()
