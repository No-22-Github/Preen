"""矩阵分析:配对判定表(5格红/绿)+ state 余弦基线 + 十问附录。

绿判据(预写,跑完不许改):
  loss 相对差 <2% 且 两版十问均 0 循环 且 全部自发终止。
  std 只记录不设阈值。

用法:
  python matrix_analyze.py <data_dir>
  data_dir 下应有 0.4b_s42_fp32/ 等子目录(每组含 events.jsonl + state.npz + decode.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def load_group(path: Path):
    """读一组的产物:events.jsonl + state.npz + decode.json。"""
    g = {"path": str(path)}
    ep = path / "events.jsonl"
    if ep.exists():
        with open(ep, encoding="utf-8") as f:
            events = [json.loads(l) for l in f if l.strip()]
        g["header"] = next((e for e in events if e.get("type") == "header"), {})
        g["final"] = next((e for e in reversed(events) if e.get("type") == "final"), {})
        g["events"] = events
    sp = path / "state.npz"
    if sp.exists():
        data = np.load(sp)
        g["state"] = {i: np.array(data[f"layer_{i}"]).astype(np.float32) for i in range(len(data.files))}
    dp = path / "decode.json"
    if dp.exists():
        with open(dp, encoding="utf-8") as f:
            g["decode"] = json.load(f)
    return g


def state_cosine(s1: dict, s2: dict) -> float:
    """两个 state dict 的整体余弦相似度(所有层拼接)。"""
    keys = sorted(set(s1) & set(s2))
    if not keys:
        return 0.0
    a = np.concatenate([s1[k].flatten() for k in keys])
    b = np.concatenate([s2[k].flatten() for k in keys])
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    if len(sys.argv) < 2:
        print("用法: python matrix_analyze.py <data_dir>")
        sys.exit(1)
    data_dir = Path(sys.argv[1])

    # 发现所有组(命名 <model>_s<seed>_<precision>)
    groups = {}
    for d in sorted(data_dir.iterdir()):
        if d.is_dir() and "_s" in d.name:
            groups[d.name] = load_group(d)

    if not groups:
        print(f"✗ 在 {data_dir} 下没找到组目录(*_s*_* 格式)")
        sys.exit(1)

    # 配对:(model × seed) 的 fp32 vs bf16
    # 组名格式:04b_s42_fp32 / 15b_s1042_bf16
    pairs = {}  # key = (model, seed)
    for name in groups:
        parts = name.rsplit("_", 2)  # [model, sNNN, precision]
        if len(parts) == 3:
            model, seedstr, prec = parts
            key = (model, seedstr)
            pairs.setdefault(key, {})[prec] = name

    print("=" * 78)
    print("# 配对判定表(绿判据:loss 相对差<2% 且 两版0循环 且 全部自发终止)")
    print("=" * 78)

    # 表头 device info(取第一组的)
    any_g = next(iter(groups.values()))
    dh = any_g.get("header", {}).get("device", {})
    print(f"\n机型:working_set={dh.get('max_recommended_working_set_size_gb','?')}G "
          f"commit={any_g.get('header',{}).get('commit','?')}")

    pair_results = []
    print(f"\n{'配对':>16} | {'fp32 loss':>9} {'bf16 loss':>9} {'diff%':>7} | "
          f"{'fp32循环':>6} {'bf16循环':>6} | {'终止':>4} | {'std范围':>14} | 判定")
    print("-" * 100)

    all_green = True
    for key in sorted(pairs):
        model, seedstr = key
        pair = pairs[key]
        fp32_name = pair.get("fp32")
        bf16_name = pair.get("bf16")
        if not fp32_name or not bf16_name:
            print(f"{model+'_'+seedstr:>16} | ⚠️ 缺 {'fp32' if not fp32_name else 'bf16'}")
            all_green = False  # 配对不全不算全绿
            continue
        gfp = groups[fp32_name]
        gbf = groups[bf16_name]
        fp_loss = gfp.get("final", {}).get("final_epoch_avg_loss", 0)
        bf_loss = gbf.get("final", {}).get("final_epoch_avg_loss", 0)
        diff_pct = ((bf_loss - fp_loss) / fp_loss * 100) if fp_loss else 0
        fp_circ = gfp.get("decode", {}).get("n_circular", -1)
        bf_circ = gbf.get("decode", {}).get("n_circular", -1)
        fp_stop = gfp.get("decode", {}).get("n_early_stop", 0)
        bf_stop = gbf.get("decode", {}).get("n_early_stop", 0)
        # std 范围
        fp_stds = list(gfp.get("final", {}).get("per_layer_std", {}).values())
        bf_stds = list(gbf.get("final", {}).get("per_layer_std", {}).values())
        fp_std_range = f"{min(fp_stds):.3f}~{max(fp_stds):.3f}" if fp_stds else "?"
        bf_std_range = f"{min(bf_stds):.3f}~{max(bf_stds):.3f}" if bf_stds else "?"
        std_range = f"fp32:{fp_std_range}\nbf16:{bf_std_range}"

        # 判定
        loss_ok = abs(diff_pct) < 2
        circ_ok = fp_circ == 0 and bf_circ == 0
        stop_ok = (fp_stop == 10 and bf_stop == 10)  # 10问全自发终止
        verdict = "🟢 绿" if (loss_ok and circ_ok and stop_ok) else "🔴 红"
        if not (loss_ok and circ_ok and stop_ok):
            all_green = False
        reason = []
        if not loss_ok:
            reason.append(f"loss差{diff_pct:+.1f}%≥2%")
        if not circ_ok:
            reason.append(f"循环 fp32={fp_circ} bf16={bf_circ}")
        if not stop_ok:
            reason.append(f"终止 fp32={fp_stop}/10 bf16={bf_stop}/10")
        verdict_str = verdict if not reason else f"{verdict}({';'.join(reason)})"

        pair_results.append({"pair": f"{model}_{seedstr}", "fp_loss": fp_loss, "bf_loss": bf_loss,
                             "diff_pct": round(diff_pct, 2), "fp_circ": fp_circ, "bf_circ": bf_circ,
                             "verdict": "绿" if "绿" in verdict else "红"})

        print(f"{model+'_'+seedstr:>16} | {fp_loss:>9.4f} {bf_loss:>9.4f} {diff_pct:>+6.2f}% | "
              f"{fp_circ:>6} {bf_circ:>6} | {fp_stop}+{bf_stop}/20 | "
              f"{std_range:>14} | {verdict_str}")

    print(f"\n{'全绿' if all_green else '存在红格'}:bf16 默认化{'可推进(另开单切默认)' if all_green else '需先解决红格'}")

    # ── state 余弦基线(任务4)──
    print("\n" + "=" * 78)
    print("# state 余弦基线(给 D 报告 0.305 那个数一个参照系)")
    print("=" * 78)

    # 跨精度同 seed:fp32_s42 vs bf16_s42
    # 同精度跨 seed:fp32_s42 vs fp32_s1042
    def find_group(model, seedstr, prec):
        name = f"{model}_s{seedstr}_{prec}"
        return groups.get(name)

    # 1.5B seed42 跨精度(D 报告的 0.305)
    print(f"\n## 跨精度同 seed(对标 D 报告的 0.305)")
    for model in ["15b", "04b"]:
        for seedstr in ["42"]:
            gfp = find_group(model, seedstr, "fp32")
            gbf = find_group(model, seedstr, "bf16")
            if gfp and gbf and "state" in gfp and "state" in gbf:
                cos = state_cosine(gfp["state"], gbf["state"])
                print(f"  {model}_s{seedstr}: fp32 vs bf16 余弦 = {cos:.4f}")

    # 同精度跨 seed(余弦基线)
    print(f"\n## 同精度跨 seed(run-to-run 方差基线)")
    for model in ["15b", "04b"]:
        seeds = sorted(set(k[1] for k in pairs if k[0] == model))
        if len(seeds) < 2:
            print(f"  {model}: 只有一个 seed,无法算跨 seed 余弦")
            continue
        # fp32 两 seed
        for prec in ["fp32", "bf16"]:
            seed_pairs = []
            for i in range(len(seeds)):
                for j in range(i+1, len(seeds)):
                    g1 = find_group(model, seeds[i], prec)
                    g2 = find_group(model, seeds[j], prec)
                    if g1 and g2 and "state" in g1 and "state" in g2:
                        cos = state_cosine(g1["state"], g2["state"])
                        seed_pairs.append((f"s{seeds[i]}_vs_s{seeds[j]}", cos))
            if seed_pairs:
                print(f"  {model} {prec}: ", end="")
                for lbl, c in seed_pairs:
                    print(f"{lbl}={c:.4f}  ", end="")
                print()

    print(f"\n## 解读")
    print(f"  若'同精度跨 seed'余弦也 ~0.3 → bf16 偏离在正常 run-to-run 方差内,D 结论加强")
    print(f"  若'同精度跨 seed'余弦显著更高(>0.8)→ bf16 偏离超出 seed 方差,记为待解释项")

    # ── 红线标定(如果跑了)──
    redline_path = data_dir / "redline_result.json"
    if redline_path.exists():
        with open(redline_path, encoding="utf-8") as f:
            rl = json.load(f)
        print("\n" + "=" * 78)
        print("# 红线标定(bf16 + c4G + 16GB 样本长度实测上界)")
        print("=" * 78)
        print(f"\n  working_set={rl['device']['max_recommended_working_set_size_gb']}G "
              f"削顶线={rl['cap_line_gb']}G commit={rl.get('commit','?')}")
        print(f"\n{'桶':>8} | {'token max':>9} | {'step_peak':>9} | {'active+cache':>12} | "
              f"{'compressor':>10} | {'max ms':>7} | 断点")
        for b in rl["buckets"]:
            brk = b["hit_break"][:30] + "..." if b.get("hit_break") and len(b["hit_break"]) > 30 else (b.get("hit_break") or "—")
            print(f"{b['bucket']:>8} | {b['token_max']:>9} | {b['max_step_peak_gb']:>8.2f}G | "
                  f"{b['max_active_plus_cache_gb']:>11.2f}G | {b['max_compress_gb']:>9.2f}G | "
                  f"{b['max_ms']:>6.0f}ms | {brk}")
        if rl.get("first_break_bucket"):
            print(f"\n  ★ bf16 + c4G 在 16GB 的实测上界:{rl['first_break_bucket']}")
        else:
            print(f"\n  ⚠️ 所有桶都没测到断点,需要加更长桶继续上探")

    # ── 十问附录(留风格判读)──
    print("\n" + "=" * 78)
    print("# 十问原文并排(风格判读留给用户)")
    print("=" * 78)
    # 只对 seed42 的两组并排(代表性)
    for model in ["15b", "04b"]:
        gfp = find_group(model, "42", "fp32")
        gbf = find_group(model, "42", "bf16")
        if not (gfp and gbf and "decode" in gfp and "decode" in gbf):
            continue
        print(f"\n━━━ {model} seed42 ━━━")
        fp_res = gfp["decode"]["results"]
        bf_res = gbf["decode"]["results"]
        for i, (r1, r2) in enumerate(zip(fp_res, bf_res)):
            print(f"\n[{i+1}] {r1['q']}")
            print(f"  fp32: {r1['out'][:150]}")
            print(f"  bf16: {r2['out'][:150]}")


if __name__ == "__main__":
    main()
