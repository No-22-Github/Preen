"""
生成 golden 快照文件 (一次性脚本, 生成后 golden/ 下的 json 提交进 git)。

用库里现成的 final_state_v3.npz + evaluate_v2.generate 跑出期望输出。
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

import mlx.core as mx
from mlx_lm import load
from data_v2 import extract_cn_en
from state_tuner import generate
from statetuner.templates import P0_BARE

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODEL = REPO_ROOT / "models" / "converted" / "rwkv7-g1d-0.4b"
STATE = Path(__file__).resolve().parent.parent / "final_state_v3.npz"
GOLDEN = Path(__file__).resolve().parent / "golden"

TRAIN_DATA = REPO_ROOT / "train_data" / "translate" / "data_100.jsonl"
TEST_DATA = REPO_ROOT / "train_data" / "translate" / "test_10.jsonl"


def load_pairs(path, limit=None, pick=None):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cn, en = extract_cn_en(json.loads(line)["text"])
            pairs.append((cn, en))
    if pick:
        return [pairs[i] for i in pick]
    return pairs[:limit] if limit else pairs


def main():
    model, tok = load(str(MODEL), tokenizer_config={"trust_remote_code": True})

    # 训练集前5条 (与 report_examples 一致)
    train_pairs = load_pairs(TRAIN_DATA, limit=5)
    # 测试集隔行取5条 (与 report_examples 一致: index 0,2,4,6,8)
    test_pairs = load_pairs(TEST_DATA, pick=[0, 2, 4, 6, 8])

    def gen(cn, state_npz=STATE, max_tokens=70):
        out = generate(model, tok, P0_BARE.format_prefix(cn=cn), state_npz=str(state_npz), max_tokens=max_tokens)
        return out.split("\n")[0].strip() if "\n" in out else out.strip()

    # translate_train.json
    train_golden = {}
    for cn, ref in train_pairs:
        train_golden[cn] = {"ref": ref, "out": gen(cn)}
    (GOLDEN / "translate_train.json").write_text(
        json.dumps(train_golden, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"translate_train.json: {len(train_golden)} 条")

    # translate_test.json
    test_golden = {}
    for cn, ref in test_pairs:
        test_golden[cn] = {"ref": ref, "out": gen(cn)}
    (GOLDEN / "translate_test.json").write_text(
        json.dumps(test_golden, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"translate_test.json: {len(test_golden)} 条")

    # conditional.json: 英文 + 乱码输入 (期望: 不触发中→英翻译)
    english_inputs = [
        "The weather is nice today.",
        "I like programming and reading.",
        "Artificial intelligence is changing the world.",
        "Please bring me a cup of coffee.",
        "The meeting starts at three.",
    ]
    junk_inputs = ["1234567890", "asdfghjkl zxcvbnm", "！！！@#￥%……"]
    cond_golden = {"english": {}, "junk": {}}
    for e in english_inputs:
        cond_golden["english"][e] = gen(e)
    for j in junk_inputs:
        cond_golden["junk"][j] = gen(j)
    (GOLDEN / "conditional.json").write_text(
        json.dumps(cond_golden, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"conditional.json: {len(english_inputs)} 英文 + {len(junk_inputs)} 乱码")

    # baseline_no_state.json: 原始模型(无state)对中文, 期望不翻译
    baseline_cns = [p[0] for p in test_pairs[:3]]
    baseline_golden = {}
    for cn in baseline_cns:
        out = generate(model, tok, P0_BARE.format_prefix(cn=cn), state_npz=None, max_tokens=50)
        baseline_golden[cn] = out.split("\n")[0].strip() if "\n" in out else out.strip()
    (GOLDEN / "baseline_no_state.json").write_text(
        json.dumps(baseline_golden, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"baseline_no_state.json: {len(baseline_golden)} 条")

    print(f"\n全部 golden 写入 {GOLDEN}")


if __name__ == "__main__":
    main()
