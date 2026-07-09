"""
P1 实现验证 + 0.4B vs 1.5B 翻译效果对比。

用 P0 报告 §四的标准句子集(T1-T5 训练集内, S1-S5 测试集),
完全相同的配置({中文}\n 前缀, 贪心 temp=0, 70 token, 取首行)跑三个场景:

  A. 0.4B + P0 state (ep04)   → 逐条对照报告, 检查 P1 实现有无回退
  B. 1.5B + 新训 state (2ep)  → 看大模型翻译效果

P0 报告的原始输出硬编码在下方 P0_REPORT 变量里, 供自动比对。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

import mlx.core as mx
from statetuner.core import load_model, generate

# ── P0 报告 §四 的句子和原始输出(逐字摘自实验报告.md)──────────
TRAIN_SET = [  # 训练集内 T1-T5
    ("T1", "今天的天气真好，适合出去散步。",
     "The weather is really nice today, perfect for going out for a walk."),
    ("T2", "请把这份报告打印出来，下午我要用它开会。",
     "Please print out this report; I'll need it for the meeting this afternoon."),
    ("T3", "这家咖啡馆的拿铁咖啡非常香浓，推荐你尝尝。",
     "The latte at this café is very rich and aromatic; I recommend you try it."),
    ("T4", "明天上午九点，我们在公司大门口集合出发。",
     "We'll gather at the company's main gate at 9 AM tomorrow to set off."),
    ("T5", "这个项目的预算需要重新调整，因为原材料价格上涨了。",
     "The budget for this project needs to be readjusted because the price of raw materials has increased."),
]
TEST_SET = [  # 测试集 S1-S5 (模型未见过)
    ("S1", "今天下午三点开会，别忘了带上项目文档。",
     "The meeting is at 3 PM this afternoon, don't forget to bring the project documents."),
    ("S2", "由于连续降雨，部分地区出现了轻微内涝，请居民注意出行安全。",
     "Due to continuous rainfall, some areas have experienced minor flooding; residents are advised to take care when traveling."),
    ("S3", "人工智能技术正在深刻改变各行各业的生产方式和商业模式。",
     "Artificial intelligence technology is profoundly transforming production methods and business models across industries."),
    ("S4", "请把这份合同的第三条款重新审阅一下，确保没有法律漏洞。",
     "Please review Clause Three of this contract again to make sure there are no legal loopholes."),
    ("S5", "这款处理器采用了7纳米制程工艺，功耗控制得相当出色。",
     "This processor uses a 7-nanometer manufacturing process and has quite excellent power efficiency."),
]

# P0 报告中 0.4B + ep04 state 的原始输出(逐字摘录, 用于回退检测)
P0_REPORT_OUTPUT = {
    "T1": "The weather is really nice today, perfect for going out for a walk.",
    "T2": "Please print out this report; I'll have to print it for the next week.",
    "T3": "This coffee is very rich, and it's very acidic; I recommend you try it for this café.",
    "T4": "I found a way to set up tomorrow tomorrow morning; I found a way to set off.",
    "T5": "The budget for this project is very expensive; the price of this material should be increased.",
    "S1": "Please remember to open the meeting to visit next week; I'll have to bring you be reading.",
    "S2": "Due to the rains, the rains are light; it will be necessary for now.",
    "S3": "Due to work with various products, the manufacturing is very technical.",
    "S4": "Please confirm this contract for this contract; it will be sure on different things.",
    "S5": "This software is very sophisticated; it will be needed processed.",
}


def first_line(s):
    return s.split("\n")[0].strip() if "\n" in s else s.strip()


def english_ratio(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if ord(c) < 128) / len(letters)


def run_eval(model, tok, state, dataset, label, max_tokens=70):
    """跑一组测试, 返回 [(id, cn, ref, out, en_ratio, is_english)]。"""
    print(f"\n{'='*70}")
    print(f" {label}")
    print(f"{'='*70}")
    results = []
    for tid, cn, ref in dataset:
        out = generate(model, tok, f"{cn}\n", state=state, max_tokens=max_tokens)
        out = first_line(out)
        er = english_ratio(out)
        is_en = er > 0.5
        results.append((tid, cn, ref, out, er, is_en))
        flag = "EN" if is_en else "CN"
        print(f"\n[{tid}] ({flag}, en_ratio={er:.2f})")
        print(f"  CN : {cn}")
        print(f"  REF: {ref}")
        print(f"  OUT: {out}")
    return results


def compare_to_report(results_04):
    """0.4B 结果逐条对比 P0 报告, 检测回退。"""
    print(f"\n{'='*70}")
    print(f" P1 vs P0 报告回退检测 (0.4B + ep04)")
    print(f"{'='*70}")
    exact = 0
    degraded = 0
    for tid, cn, ref, out, er, is_en in results_04:
        report_out = P0_REPORT_OUTPUT.get(tid, "")
        if out == report_out:
            status = "✓ 完全一致"
            exact += 1
        elif is_en:
            # 都是英文但文本不同 —— 检查是否质量明显下降(更短/更碎)
            if len(out) < len(report_out) * 0.4:
                status = "⚠ 输出变短(可能回退)"
                degraded += 1
            else:
                status = "~ 英文但措辞不同(ULP 级, 可接受)"
        else:
            status = "✗ 回退! 输出非英文"
            degraded += 1
        print(f"  [{tid}] {status}")
        if out != report_out:
            print(f"       P0报告: {report_out[:70]}")
            print(f"       P1现在: {out[:70]}")
    print(f"\n  汇总: {exact}/10 与报告完全一致, {degraded} 条疑似回退")
    return degraded


def compare_models(results_04, results_15):
    """0.4B vs 1.5B 翻译效果对比。"""
    print(f"\n{'='*70}")
    print(f" 0.4B vs 1.5B 翻译效果对比")
    print(f"{'='*70}")
    print(f"\n{'ID':<5} {'0.4B':<8} {'1.5B':<8} {'说明'}")
    print("-" * 60)
    for (tid, cn, ref, o4, er4, en4), (_, _, _, o15, er15, en15) in zip(results_04, results_15):
        s4 = "EN" if en4 else "CN"
        s15 = "EN" if en15 else "CN"
        note = ""
        if en4 and en15:
            note = "都是英文翻译"
        elif en15 and not en4:
            note = "1.5B 更好(出英文, 0.4B 没有)"
        elif en4 and not en15:
            note = "0.4B 更好(⚠ 1.5B 回退?)"
        else:
            note = "都未出英文"
        print(f"{tid:<5} {s4:<8} {s15:<8} {note}")


def main():
    MODEL_04 = "models/converted/rwkv7-g1d-0.4b"
    MODEL_15 = "models/converted/rwkv7-g1g-1.5b"
    STATE_04 = "experiments/p0_translate/checkpoints_v3/ep04.npz"
    STATE_15 = "/tmp/1.5b_state.npz"  # 刚训的 2 epoch

    all_set = TRAIN_SET + TEST_SET

    # ── A. 0.4B + P0 state ──────────────────────────────────
    print("加载 0.4B 模型 (kernel 路径)...", file=sys.stderr)
    t0 = time.time()
    m04, tok04 = load_model(MODEL_04, patch=False)
    print(f"  加载耗时 {time.time()-t0:.1f}s", file=sys.stderr)
    results_04 = run_eval(m04, tok04, STATE_04, all_set, "场景 A: 0.4B + P0 state (ep04)")

    # 回退检测
    degraded = compare_to_report(results_04)

    # ── B. 1.5B + 新训 state ────────────────────────────────
    print("\n加载 1.5B 模型 (kernel 路径)...", file=sys.stderr)
    t0 = time.time()
    m15, tok15 = load_model(MODEL_15, patch=False)
    print(f"  加载耗时 {time.time()-t0:.1f}s", file=sys.stderr)
    results_15 = run_eval(m15, tok15, STATE_15, all_set, "场景 B: 1.5B + 新训 state (2 epoch)")

    # ── C. 0.4B vs 1.5B 对比 ────────────────────────────────
    compare_models(results_04, results_15)

    # 峰值内存
    print(f"\n峰值内存: {mx.get_peak_memory()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
