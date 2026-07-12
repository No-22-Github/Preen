"""一次性脚本:从 fla-hub 0.1B model.safetensors 生成转换校验 fixture。

把"同架构 fla 模型"的键名 + ndim 冻结成仓库内 JSON,消除转换器对
fla-hub 仓库下载的依赖。只存 ndim(忠实于现状——现有校验逻辑本来就
只比 ndim,0.1B 与目标模型 hidden_size 不同,绝对 shape 无法比对)。

用法(需本机存在 fla-hub 参照目录,通常上游 schema 变化时重新生成):
  python tools/gen_convert_fixture.py \
      --reference models/fla-hub-rwkv7-0.1B-g1/model.safetensors \
      --output tools/fixtures/rwkv7_hf_template.json

生成后自动做一致性自检:用 fixture 与活的 safetensors 各跑一遍归一化,
断言两者产出的键集合 + ndim 完全一致。
"""
import argparse
import json
import os
import sys

# 复用转换器里的归一化纯函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_rwkv7_to_hf import normalize_layer_key, load_reference_template

SOURCE_REPO = "https://huggingface.co/fla-hub/rwkv7-0.1B-g1"


def build_template_from_live(ref_path):
    """读活的 safetensors,返回 (layer_ndim, top_ndim)。

    与 load_reference_template 的归一化完全一致,但只保留 ndim
    (fixture 不存绝对 shape)。
    """
    from safetensors import safe_open
    f = safe_open(ref_path, framework="pt")
    layer_ndim = {}   # 相对键名 → ndim
    top_ndim = {}     # 顶层键 → ndim
    for k in sorted(f.keys()):
        t = f.get_tensor(k)
        is_layer, rel = normalize_layer_key(k)
        ndim = len(t.shape)
        if is_layer:
            if rel not in layer_ndim:
                layer_ndim[rel] = ndim
        else:
            if k not in top_ndim:
                top_ndim[k] = ndim
    f.__exit__(None, None, None)
    return layer_ndim, top_ndim


def selfcheck(ref_path, fixture_path):
    """一致性自检: fixture 与活 safetensors 的键集合 + ndim 必须完全一致。"""
    live_layer, live_top = build_template_from_live(ref_path)
    with open(fixture_path, "r", encoding="utf-8") as fh:
        fx = json.load(fh)
    fx_layer = fx["layer_keys"]
    fx_top = fx["top_keys"]

    assert set(fx_layer.keys()) == set(live_layer.keys()), (
        f"layer_keys 键集合不一致:\n"
        f"  仅 fixture: {set(fx_layer) - set(live_layer)}\n"
        f"  仅 live:    {set(live_layer) - set(fx_layer)}"
    )
    assert set(fx_top.keys()) == set(live_top.keys()), (
        f"top_keys 键集合不一致:\n"
        f"  仅 fixture: {set(fx_top) - set(live_top)}\n"
        f"  仅 live:    {set(live_top) - set(fx_top)}"
    )
    for k in live_layer:
        assert fx_layer[k] == live_layer[k], (
            f"layer ndim 不一致 {k}: fixture={fx_layer[k]} live={live_layer[k]}"
        )
    for k in live_top:
        assert fx_top[k] == live_top[k], (
            f"top ndim 不一致 {k}: fixture={fx_top[k]} live={live_top[k]}"
        )

    # 再额外对照 load_reference_template 路径(它返回 shape tuple,取 len 应等于 ndim)
    tpl, top_tpl = load_reference_template(ref_path)
    for k, ndim in live_layer.items():
        assert len(tpl[k]) == ndim, f"load_reference_template 与 build 不一致 {k}"
    for k, ndim in live_top.items():
        assert len(top_tpl[k]) == ndim, f"load_reference_template 与 build 不一致 {k}"
    print(f"自检通过: {len(fx_layer)} 个层内键 + {len(fx_top)} 个顶层键,ndim 全一致")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--reference", required=True,
                   help="fla-hub 0.1B model.safetensors 路径")
    p.add_argument("--output", required=True,
                   help="输出 fixture JSON 路径")
    p.add_argument("--no-selfcheck", action="store_true",
                   help="跳过一致性自检(不建议)")
    args = p.parse_args()

    layer_ndim, top_ndim = build_template_from_live(args.reference)
    fixture = {
        "layer_keys": layer_ndim,
        "top_keys": top_ndim,
        "source": f"{SOURCE_REPO} model.safetensors",
        "source_repo": SOURCE_REPO,
        "generated": "2026-07-12",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(fixture, fh, indent=2, ensure_ascii=False, sort_keys=True)
        fh.write("\n")
    print(f"已生成 fixture: {args.output}")
    print(f"  layer_keys: {len(layer_ndim)} 个")
    print(f"  top_keys:   {len(top_ndim)} 个")

    if not args.no_selfcheck:
        selfcheck(args.reference, args.output)


if __name__ == "__main__":
    main()
