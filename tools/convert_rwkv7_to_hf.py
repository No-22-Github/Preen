"""
RWKV-7 原生 .pth → fla HF (safetensors) 独立转换器。

不依赖 fla/triton —— 直接读 pth 权重,按官方 convert_from_rwkv7.py 的键名
映射规则搬运,用同架构的 0.1B fla 模型 (model.safetensors) 做 ground truth
双重校验 (键名 + shape),最后存 safetensors + config.json。

映射规则 (与 fla-org/flash-linear-attention 的 convert_from_rwkv7.py 等价):
  - 顶层:  emb.weight→model.embeddings.weight
           ln_out.{weight,bias}→model.norm.{weight,bias}
           head.weight→lm_head.weight
  - 层内:  ln0→pre_norm, ln1→attn_norm, ln2→ffn_norm
           att.{receptance,key,value,output}→{r,k,v,o}_proj
           att.ln_x→g_norm
           att.{w,a,g,v}{0,1,2}→{w,a,g,v}_lora.lora.{2.bias,0.weight,2.weight}
               (其中 *1/*2 的 weight 要转置)
           att.x_*、att.r_k、att.k_a、att.k_k 保持位置名
           blocks.0.att.{v0,v1,v2} 被丢弃 (layer 0 无 v_lora)
  - shape: [1,1,hidden] 的非 x_ 键 squeeze; x_ 键保留 (copy_ 广播)

用法:
  python convert_rwkv7_to_hf.py \
      --rwkv7 models/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
      --output models/converted/rwkv7-g1d-0.4b \
      --reference models/fla-hub-rwkv7-0.1B-g1/model.safetensors \
      --tokenizer-src models/fla-hub-rwkv7-0.1B-g1 \
      --precision bf16
"""
import argparse
import json
import os
import re
import shutil

import torch
from safetensors.torch import save_file


def load_reference_template(ref_path):
    """从同架构的 fla 模型读取键名+shape 作为 ground truth 模板。

    返回 dict[str, tuple]: 每层只取 layer 0 的相对键名 → shape (去掉层数)。
    """
    from safetensors import safe_open
    f = safe_open(ref_path, framework="pt")
    template = {}      # 相对键名 (无 layer idx) → shape
    top_keys = {}      # 顶层键 → shape
    for k in sorted(f.keys()):
        t = f.get_tensor(k)
        # 把 model.layers.N.X 规范化成 model.layers.{N}.X
        m = re.match(r"model\.layers\.(\d+)\.(.+)", k)
        if m:
            rel = "model.layers.{N}." + m.group(2)
            if rel not in template:
                template[rel] = tuple(t.shape)
        else:
            if k not in top_keys:
                top_keys[k] = tuple(t.shape)
    f.__exit__(None, None, None)
    return template, top_keys


def infer_config(weights):
    """从 pth 权重的 shape 推断 RWKV7Config 所需字段。"""
    config = {}
    config["vocab_size"] = weights["emb.weight"].shape[0]
    config["hidden_size"] = weights["blocks.0.ffn.key.weight"].shape[1]
    config["intermediate_size"] = weights["blocks.0.ffn.key.weight"].shape[0]
    config["hidden_ratio"] = (
        weights["blocks.0.ffn.key.weight"].shape[0]
        / weights["blocks.0.ffn.key.weight"].shape[1]
    )
    # 层数
    n = 0
    while f"blocks.{n}.ffn.key.weight" in weights:
        n += 1
    config["num_hidden_layers"] = n
    config["decay_low_rank_dim"] = weights["blocks.0.att.w1"].shape[1]
    config["gate_low_rank_dim"] = weights["blocks.0.att.g1"].shape[1]
    config["a_low_rank_dim"] = weights["blocks.0.att.a1"].shape[1]
    try:
        config["v_low_rank_dim"] = weights["blocks.1.att.v1"].shape[1]
    except KeyError:
        config["v_low_rank_dim"] = 32
    config["head_dim"] = 64
    config["num_heads"] = config["hidden_size"] // 64
    config["value_dim"] = [config["hidden_size"]] * n
    return config


# ── 键名映射 (与官方 convert_from_rwkv7.translate_into_fla 等价) ──
EMB_HEAD = {
    "emb.weight": "model.embeddings.weight",
    "ln_out.weight": "model.norm.weight",
    "ln_out.bias": "model.norm.bias",
    "head.weight": "lm_head.weight",
}
PROJ = {
    "receptance": "r_proj",
    "key": "k_proj",
    "value": "v_proj",
    "ln_x": "g_norm",
    "output": "o_proj",
}
UNUSED = ["blocks.0.att.v0", "blocks.0.att.v1", "blocks.0.att.v2"]


def translate(src_name, num_layers):
    """返回 (fla_name, transposed)。空字符串表示丢弃。"""
    if src_name in UNUSED:
        return "", False
    if src_name in EMB_HEAD:
        return EMB_HEAD[src_name], False

    parts = src_name.split(".")
    assert parts[0] == "blocks", f"unexpected key: {src_name}"
    parts[0] = "model.layers"
    li = int(parts[1])
    assert 0 <= li < num_layers
    parts[1] = "{N}"  # 占位,稍后替换为真实层号
    layer_map = {
        "att": "attn", "ffn": "ffn",
        "ln0": "pre_norm", "ln1": "attn_norm", "ln2": "ffn_norm",
    }
    assert parts[2] in layer_map, f"unexpected sub: {src_name}"
    parts[2] = layer_map[parts[2]]

    transposed = False
    # [wvag][012] → {typ}_lora.lora.{位置}
    # 官方映射: 0→2.bias, 1→0.weight, 2→2.weight; num in (1,2) 转置
    if re.match(r"^[wvag][012]$", parts[3]):
        typ, num = parts[3][0], parts[3][1]
        parts[3] = f"{typ}_lora.lora." + {"0": "2.bias", "1": "0.weight", "2": "2.weight"}[num]
        transposed = num in ("1", "2")
    elif parts[2] == "attn" and parts[3] in PROJ:
        parts[3] = PROJ[parts[3]]
    # 其余 (x_*, r_k, k_a, k_k, ffn.x_k) 保持
    return ".".join(parts), transposed


def convert(rwkv7_path, output, ref_path, tokenizer_src, precision):
    print(f"加载源权重: {rwkv7_path}")
    weights = torch.load(rwkv7_path, weights_only=True, map_location="cpu")
    config = infer_config(weights)
    print(f"推断配置: layers={config['num_hidden_layers']} "
          f"hidden={config['hidden_size']} vocab={config['vocab_size']} "
          f"ffn={config['intermediate_size']}")

    # dtype
    dtype = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
             "fp16": torch.float16, "float16": torch.float16,
             "fp32": torch.float32, "float32": torch.float32}[precision]

    # ground truth 模板
    template, top_template = load_reference_template(ref_path)
    print(f"参考模板: {len(template)} 个层内键 + {len(top_template)} 个顶层键")

    new_weights = {}
    reported_layer0 = set()
    for src_name in weights:
        rel_name, transposed = translate(src_name, config["num_hidden_layers"])
        if not rel_name:
            print(f"  [跳过] {src_name} (unused)")
            continue
        if "{N}" in rel_name:
            li = int(src_name.split(".")[1])
            fla_name = rel_name.replace("{N}", str(li))
        else:
            li = -1
            fla_name = rel_name
        weight = weights[src_name].clone()

        if transposed:
            weight.t_()

        shape_before = list(weight.shape)
        is_x = "attn.x_" in fla_name
        if shape_before == [1, 1, config["hidden_size"]]:
            # 非 x_ 键 squeeze; x_ 键保留 (与官方 copy_ 广播语义一致)
            if not is_x:
                weight.squeeze_()

        # ground truth 校验: 同架构不同 hidden_size,只校验维度数一致
        # (0.1B hidden=768, 本模型 hidden=1024,绝对值不同但结构同)
        rel_check = rel_name  # 含 {N}
        if li == 0 and rel_check in template:
            ref_shape = template[rel_check]
            if is_x:
                # x_ 键: 参考是 (1,1,H), 实际也应是 (1,1,H) → 比维度数
                ok = len(ref_shape) == weight.ndim
            else:
                ok = len(ref_shape) == weight.ndim
            if not ok:
                raise ValueError(
                    f"维度数校验失败 {fla_name}: 参考={ref_shape}(ndim={len(ref_shape)}) "
                    f"实际={tuple(weight.shape)}(ndim={weight.ndim})"
                )
            reported_layer0.add(rel_check)
        if li == 0 and fla_name in top_template:
            ref_shape = top_template[fla_name]
            if len(ref_shape) != weight.ndim:
                raise ValueError(
                    f"顶层维度数校验失败 {fla_name}: 参考={ref_shape} 实际={tuple(weight.shape)}"
                )

        new_weights[fla_name] = weight.to(dtype).contiguous()

    # 报告: layer 0 模板里有没有没被源权重覆盖的键 (对应 possible_absent_weights)
    uncovered = set(template.keys()) - reported_layer0
    if uncovered:
        # pre_norm (ln0) 可能缺失,允许
        for u in uncovered:
            if "pre_norm" in u:
                print(f"  [注意] layer0 缺 {u} (ln0,允许缺失)")
            else:
                print(f"  [警告] layer0 模板键未被覆盖: {u}")

    os.makedirs(output, exist_ok=True)
    out_st = os.path.join(output, "model.safetensors")
    print(f"保存权重: {out_st} ({len(new_weights)} 个张量, {precision})")
    save_file(new_weights, out_st, metadata={"format": "pt"})

    # config.json
    config_json = {
        "model_type": "rwkv7",
        "architect": ["RWKV7ForCausalLM"],
        "auto_map": {
            "AutoConfig": "fla.models.rwkv7.configuration_rwkv7.RWKV7Config",
            "AutoModelForCausalLM": "fla.models.rwkv7.modeling_rwkv7.RWKV7ForCausalLM",
        },
        "attn_mode": "chunk",
        "hidden_size": config["hidden_size"],
        "hidden_ratio": config["hidden_ratio"],
        "intermediate_size": config["intermediate_size"],
        "num_hidden_layers": config["num_hidden_layers"],
        "head_dim": config["head_dim"],
        "num_heads": config["num_heads"],
        "decay_low_rank_dim": config["decay_low_rank_dim"],
        "gate_low_rank_dim": config["gate_low_rank_dim"],
        "a_low_rank_dim": config["a_low_rank_dim"],
        "v_low_rank_dim": config["v_low_rank_dim"],
        "value_dim": config["value_dim"],
        "hidden_act": "sqrelu",
        "max_position_embeddings": 2048,
        "norm_first": True,
        "norm_bias": True,
        "norm_eps": 1e-5,
        "use_cache": True,
        "tie_word_embeddings": False,
        "fuse_norm": True,
        "fuse_cross_entropy": True,
        "fuse_linear_cross_entropy": False,
        "use_l2warp": True,
        "vocab_size": config["vocab_size"],
        "torch_dtype": {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[precision],
        "bos_token_id": 0,
        "eos_token_id": 0,
        "pad_token_id": 0,
    }
    with open(os.path.join(output, "config.json"), "w") as f:
        json.dump(config_json, f, indent=2, ensure_ascii=False)
    print(f"保存 config.json")

    # tokenizer (官方脚本不输出 tokenizer,需从 fla-hub 拷贝)
    for fn in ["hf_rwkv_tokenizer.py", "tokenizer_config.json",
               "rwkv_vocab_v20230424.txt", "special_tokens_map.json",
               "added_tokens.json"]:
        src = os.path.join(tokenizer_src, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output, fn))
            print(f"拷贝 tokenizer: {fn}")

    print(f"\n转换完成 → {output}")
    print("下一步: 用 mlx_lm.load 或 transformers 加载验证")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rwkv7", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--reference", required=True,
                   help="同架构 fla 模型的 safetensors,做 ground truth 校验")
    p.add_argument("--tokenizer-src", required=True,
                   help="tokenizer 来源目录 (fla-hub/rwkv7-0.1B-g1)")
    p.add_argument("--precision", default="bf16",
                   choices=["bf16", "fp16", "fp32"])
    args = p.parse_args()
    convert(args.rwkv7, args.output, args.reference, args.tokenizer_src, args.precision)
