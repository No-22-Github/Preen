"""汇总所有内存实测数据 → 表格 + 拟合。等所有实验完成后跑。"""
import json
import os
import sys

V2 = "tools/mem_v2"

def load(name):
    p = os.path.join(V2, f"{name}.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None

def get_done(name):
    """从 log 读 DONE 行 (json 可能没写完)"""
    p = os.path.join(V2, f"{name}.log")
    if not os.path.exists(p):
        return None
    for line in open(p):
        if "DONE" in line:
            import re
            m = re.search(r"cache=([\d.]+)G.*compress=([\d.]+)G.*ms=(\d+)", line)
            if m:
                return {"cache": float(m.group(1)), "compress": float(m.group(2)), "ms": int(m.group(3))}
    return None

def main():
    print("=" * 70)
    print("任务1: 0.4B 受控长度实验")
    print("=" * 70)
    pts04 = []
    for name, lbl in [("04b_L64","L64"),("04b_L128","L128"),("04b_L256","L256"),("04b_mixed","mixed")]:
        d = load(name)
        if d:
            ds = d["data_stats"]
            print(f"  {lbl:8s} mean={ds['mean']:6.1f} max={ds['max']:4} p95={ds['p95']:4} → cache={d['stable_cache_gb']}G compress={d['stable_compress_gb']}G ms={d['ms_per_step_last10']}")
            pts04.append((ds['max'], d['stable_cache_gb'], lbl))

    print()
    print("=" * 70)
    print("任务2: 1.5B 三点实测 (无限档)")
    print("=" * 70)
    for name, lbl in [("15b_L40_inf","L40"),("15b_nekoqa200_inf","nekoqa200"),("15b_L256_inf","L256")]:
        d = load(name)
        if d:
            ds = d["data_stats"]
            print(f"  {lbl:12s} mean={ds['mean']:6.1f} max={ds['max']:4} → cache={d['stable_cache_gb']}G compress={d['stable_compress_gb']}G ms={d['ms_per_step_last10']}")
        else:
            done = get_done(name)
            if done:
                print(f"  {lbl:12s} (from log) → cache={done['cache']}G compress={done['compress']}G ms={done['ms']}")

    print()
    print("=" * 70)
    print("任务3: 1.5B cache_limit 扫描 (nekoqa200, 100步同口径)")
    print("=" * 70)
    print(f"  {'limit':>6} {'cache':>7} {'compress':>9} {'ms/step':>8}")
    for cl in [4, 6, 8, 10]:
        name = f"15b_scan_c{cl}"
        done = get_done(name)
        if done:
            print(f"  {cl:>4}G  {done['cache']:>5.1f}G  {done['compress']:>7.2f}G  {done['ms']:>6}")
    # inf
    name = "15b_scan_cinf"
    done = get_done(name)
    if done:
        print(f"  {'inf':>4}  {done['cache']:>5.1f}G  {done['compress']:>7.2f}G  {done['ms']:>6}")
    else:
        print("  inf: 待完成")

if __name__ == "__main__":
    main()
