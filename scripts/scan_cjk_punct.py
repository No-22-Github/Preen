#!/usr/bin/env python3
"""扫描 Swift 源码里中文文案中混用的 ASCII 标点。

只看字符串字面量里、与中文字符相邻的 ASCII 标点，避免误报代码本身。
用法: python3 scripts/scan_cjk_punct.py [根目录，默认 macos]
"""
import re
import sys
from pathlib import Path

# 需要检测的 ASCII 标点 -> 建议的中文全角替换
SUGGEST = {
    ",": "，",
    ";": "；",
    ":": "：",
    "!": "！",
    "?": "？",
    "(": "（",
    ")": "）",
    # 句号 / 省略号单独处理，见下
}

CJK = r"一-鿿㐀-䶿"
# 中文字符
is_cjk = re.compile(f"[{CJK}]")
# 提取 Swift 字符串字面量（含普通串与多行串，简单版：匹配双引号成对内容）
STRING_RE = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"')

# 规则：ASCII 标点，其左侧或右侧紧邻中文
# 例如  中文, / ,中文 / 中文; / (中文) / 中文...
def find_issues(text: str):
    issues = []
    for punc, zh in SUGGEST.items():
        p = re.escape(punc)
        # 中文 + 标点  或  标点 + 中文
        for m in re.finditer(f"([{CJK}]\\s*{p})|({p}\\s*[{CJK}])", text):
            issues.append((m.start(), m.group(), punc, zh))
    # ASCII 句号：中文.（非小数、非省略号点串更适合用 …）
    for m in re.finditer(f"[{CJK}]\\.(?![0-9.])", text):
        issues.append((m.start(), m.group(), ".", "。"))
    # ASCII 省略号 ... 出现在中文语境
    for m in re.finditer(r"\.\.\.", text):
        # 附近有中文才算
        seg = text[max(0, m.start() - 4): m.end() + 4]
        if is_cjk.search(seg):
            issues.append((m.start(), "...", "...", "…"))
    return issues


def strip_interpolations(text: str) -> str:
    """把 \\(...) 插值整体替换成空格，正确处理嵌套括号。"""
    out = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "\\" and i + 1 < n and text[i + 1] == "(":
            depth = 0
            j = i + 1
            while j < n:
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            out.append(" ")
            i = j + 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "macos")
    total = 0
    files_hit = 0
    for path in sorted(root.rglob("*.swift")):
        lines = path.read_text(encoding="utf-8").splitlines()
        file_issues = []
        for lineno, line in enumerate(lines, 1):
            for sm in STRING_RE.finditer(line):
                content = sm.group(1)
                # 剔除 Swift 字符串插值 \(...)（含嵌套括号），其括号不是中文标点
                content = strip_interpolations(content)
                if not is_cjk.search(content):
                    continue
                for _, frag, punc, zh in find_issues(content):
                    file_issues.append((lineno, line.strip(), frag, punc, zh))
        if file_issues:
            files_hit += 1
            print(f"\n\033[1m{path}\033[0m")
            for lineno, src, frag, punc, zh in file_issues:
                total += 1
                snippet = src if len(src) <= 90 else src[:87] + "…"
                print(f"  L{lineno}: `{punc}`→`{zh}`  {snippet}")
    print(f"\n合计 {total} 处，涉及 {files_hit} 个文件。")


if __name__ == "__main__":
    main()
