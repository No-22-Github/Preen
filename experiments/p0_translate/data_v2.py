"""
数据准备 v2: 裸文本格式,对齐评估分布。

修复 P0 复核问题 2: 训练分布与评估分布不一致。

旧格式 (有问题):
  训练: "User: {中文}\n\nAssistant: {英文}"
  评估: 纯中文裸文本
  → S₀ 在训练时对应 "User:" 之前, 评估时对应裸中文之前, 语义不同
  → state 学到的是 User/Assistant 模板偏置, 不是"看到中文就翻译"

新格式 (对齐):
  训练: "{中文}\n{英文}"   (裸文本, 分隔符固定为 \n)
  评估: 纯中文裸文本
  → S₀ 都对应"中文序列开始之前", 训练/评估分布一致
  → state 学到的 S₀ 语义 = "接下来要处理中文并输出英文翻译"

loss mask 只算英文段 (中文是条件, 不是学习目标)。
分隔符 \n 的 token 选择: 直接 tokenize 整个 "{中文}\n{英文}",
用英文段在完整序列中的 token 起点作为 mask 边界。
"""
import json


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def extract_cn_en(text):
    """从原始 jsonl 的 {"text": "User: {中}\n\nAssistant: {英}"} 提取 (中文, 英文)。

    也兼容已经是裸格式的数据。
    """
    if "Assistant:" in text and "User:" in text:
        # 标准格式
        user_part = text.split("Assistant:")[0]
        # user_part = "User: {中文}\n\n"
        cn = user_part.replace("User:", "").strip().rstrip("\n")
        en = text.split("Assistant:", 1)[1].strip()
    elif "\n" in text:
        # 已是 {中}\n{英} 裸格式
        parts = text.split("\n", 1)
        cn, en = parts[0].strip(), parts[1].strip()
    else:
        cn, en = text, ""
    return cn, en


def prepare_samples_v2(jsonl_path, tokenizer, max_len=128):
    """裸格式: "{中文}\n{英文}", 返回 (input_ids, labels, mask, length)。

    loss mask 只算英文段 (中文 + 分隔符不算 loss)。
    """
    items = load_jsonl(jsonl_path)
    samples = []
    for it in items:
        cn, en = extract_cn_en(it["text"])

        # 构造裸文本: {中文}\n{英文}
        # 用 \n 做分隔 (不是 \n\n, 评估时也不带分隔符, 这里 \n 是中文→英文的过渡)
        # 关键: S₀ 在中文之前, 中文段是"读入", 英文段是"翻译输出"
        bare_text = f"{cn}\n{en}"

        full_ids = tokenizer.encode(bare_text)
        # 中文段的 token 数 = 中文单独 encode 的长度
        cn_ids = tokenizer.encode(cn)
        cn_len = len(cn_ids)
        # 分隔符 \n 通常和前面的中文末尾或英文开头合并, 这里用 cn_len 作为边界近似
        # 更精确: encode(cn) 的长度就是中文段在 full 中的位置 (World tokenizer 基本是 char 级)
        # 但 \n 可能影响, 实测边界

        input_ids = full_ids[:-1]
        labels = full_ids[1:]
        # mask[i]=1 当 labels[i] (=full[i+1]) 落在英文段
        # 英文段从 cn_len 开始 (中文占 [0, cn_len)), full[cn_len] 可能是 \n 或英文首 token
        # 保守: 让中文段 + 紧跟的 \n 都不算 loss, 英文段算
        # 实际 full = cn_tokens + [可能的\n token] + en_tokens
        # 边界用: i+1 >= cn_len (即从 full[cn_len] 开始的预测算 loss)
        # 但 \n 若单独成 token, full[cn_len] = \n token, 算不算 loss?
        # \n 是分隔符不是翻译内容, 也不算 loss。所以边界应是英文第一个 token。
        # 用 encode(en) 反查: en 在 full 中的起点
        # 简化且稳健: 边界 = cn_len (让 \n 也算进条件区, 不算 loss)
        mask = [1 if (i + 1) >= cn_len else 0 for i in range(len(input_ids))]

        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            labels = labels[:max_len]
            mask = mask[:max_len]

        samples.append((input_ids, labels, mask, len(input_ids), cn, en, cn_len))
    return samples


def verify_boundary(samples, tokenizer, n=3):
    """打印前 n 条样本的边界,人工确认 mask 切对了。"""
    for i, s in enumerate(samples[:n]):
        ids, labels, mask, length, cn, en, cn_len = s
        print(f"--- 样本 {i} ---")
        print(f"  中文: {cn}")
        print(f"  英文: {en}")
        print(f"  cn_len(边界): {cn_len}, 总长: {length}")
        # 边界处解码
        before = tokenizer.decode(ids[:cn_len])
        at_boundary = tokenizer.decode([ids[cn_len]]) if cn_len < len(ids) else "?"
        after = tokenizer.decode(ids[cn_len:cn_len+5])
        print(f"  边界前(中文): {before!r}")
        print(f"  边界处 token[{cn_len}]: {at_boundary!r}")
        print(f"  边界后(英文): {after!r}")
        # 验证 mask 在边界处 0→1
        transitions = [(j, mask[j]) for j in range(len(mask)) if j == 0 or mask[j] != mask[j-1]]
        print(f"  mask 0→1 转换: {transitions}")
        print()
