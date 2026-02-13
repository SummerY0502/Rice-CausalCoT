import re
import json
import argparse
from pathlib import Path
import pandas as pd
import ast

COLUMNS = [
    "RAP_ID",
    "Gene_Symbol",
    "S1",
    "S2",
    "S2_keyword",
    "S3",
    "S3_keywords",
    "S4",
    "S4_note",
    "S5",
    "S5_keyinfo",
    "Conservative",
    "Exploratory",
]

def _extract_balanced_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise ValueError("文本中找不到 JSON 起始符号 '{'")

    depth = 0
    in_str = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i+1]

    raise ValueError("未能找到配平的 JSON 结束 '}'，文件可能被截断或格式异常")

def _strip_markdown_links_outside_strings_keep_commas(s: str) -> str:
    out = []
    i = 0
    in_str = False
    escape = False

    while i < len(s):
        ch = s[i]

        if in_str:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            i += 1
            continue

        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue

        # 删除 [..](..) 但不动前后的逗号
        if ch == "[":
            j = s.find("]", i + 1)
            if j != -1 and j + 1 < len(s) and s[j + 1] == "(":
                k = s.find(")", j + 2)
                if k != -1:
                    i = k + 1
                    continue

        out.append(ch)
        i += 1

    return "".join(out)

def _fix_missing_commas_between_fields(s: str) -> str:
    """
    针对一种常见错误：上一行以引号/数字/} / ] 结尾，
    下一行以 "key": 开头，但中间缺逗号。
    这个修复只在“字符串外部”做替换，尽量保守。
    """
    lines = s.splitlines()
    fixed = []
    for idx in range(len(lines)):
        line = lines[idx]
        fixed.append(line)

        if idx == len(lines) - 1:
            continue

        cur = line.rstrip()
        nxt = lines[idx + 1].lstrip()

        # 下一行看起来像字段开头："xxx":
        if re.match(r'^"\w[^"]*"\s*:', nxt):
            # 当前行若以这些结尾且不以逗号结尾，则补逗号
            if cur and not cur.endswith(",") and re.search(r'("|\d|\}|\])\s*$', cur):
                fixed[-1] = cur + ","

    return "\n".join(fixed)

def _strip_markdown_links_outside_strings(s: str) -> str:
    """
    删除字符串外部出现的 Markdown 链接 [text](url)。
    """
    out = []
    i = 0
    in_str = False
    escape = False

    while i < len(s):
        ch = s[i]

        if in_str:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            i += 1
            continue

        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue

        # 删除 [..](..)
        if ch == "[":
            j = s.find("]", i + 1)
            if j != -1 and j + 1 < len(s) and s[j + 1] == "(":
                k = s.find(")", j + 2)
                if k != -1:
                    i = k + 1
                    continue

        out.append(ch)
        i += 1

    return "".join(out)


def extract_json_from_text(text: str) -> dict:
    # 优先从 ```json ... ``` 中提取，否则从全文提取
    m = re.search(r"```json\s*(.*?)(?:```|\Z)", text, flags=re.DOTALL)
    candidate = m.group(1) if m else text

    json_str = _extract_balanced_json_object(candidate)

    # 先直接解析；失败则清洗 Markdown 链接后再解析
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        cleaned = _strip_markdown_links_outside_strings(json_str)
        return json.loads(cleaned)

def parse_batch_index(path: Path) -> int:
    m = re.search(r"batch_(\d+)_raw\.txt$", path.name)
    return int(m.group(1)) if m else 10**18

def clean_gene_symbol(symbol) -> str:
    if symbol is None:
        return ""
    s = str(symbol).strip()
    if not s:
        return ""
    return s.split(";")[0].strip()

def get_label(gene: dict, section: str) -> str:
    v = gene.get(section, {})
    return str(v.get("label", "") or "") if isinstance(v, dict) else ""

def get_reason(gene: dict, section: str) -> str:
    v = gene.get(section, {})
    return str(v.get("reason", "") or "") if isinstance(v, dict) else ""

def gene_to_row(gene: dict) -> dict:
    row = {c: "" for c in COLUMNS}
    row["RAP_ID"] = str(gene.get("rap_id", "") or "")
    row["Gene_Symbol"] = clean_gene_symbol(gene.get("gene_symbol", ""))

    row["S1"] = get_label(gene, "S1")
    row["S2"] = get_label(gene, "S2")
    row["S2_keyword"] = get_reason(gene, "S2")

    row["S3"] = get_label(gene, "S3")
    row["S3_keywords"] = get_reason(gene, "S3")

    row["S4"] = get_label(gene, "S4")
    row["S4_note"] = get_reason(gene, "S4")

    row["S5"] = get_label(gene, "S5")
    row["S5_keyinfo"] = get_reason(gene, "S5")

    row["Conservative"] = str(gene.get("Conservative", "") or "")
    row["Exploratory"] = str(gene.get("Exploratory", "") or "")

    return row

def main():
    ap = argparse.ArgumentParser(description="Merge batch_XXX_raw.txt into one CSV (default current dir).")
    ap.add_argument("--input_dir", default=".", help="包含 batch_XXX_raw.txt 的文件夹（默认当前目录）")
    ap.add_argument("--output_csv", default="Bio_Result.csv", help="输出 CSV 路径（默认当前目录）")
    ap.add_argument("--output_encoding", default="utf-8-sig", help="输出编码（默认 utf-8-sig，Excel 友好）")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir 不存在: {input_dir}")

    batch_files = sorted(input_dir.glob("batch_*_raw.txt"), key=parse_batch_index)
    if not batch_files:
        raise FileNotFoundError(f"在 {input_dir} 未找到 batch_*_raw.txt")

    rows = []
    for fp in batch_files:
        text = fp.read_text(encoding="utf-8", errors="ignore")
        data = extract_json_from_text(text)

        genes = data.get("genes", [])
        if isinstance(genes, dict):
            genes = [genes]
        if not isinstance(genes, list):
            raise ValueError(f"{fp.name}: genes 字段不是 list/dict，实际为 {type(genes)}")

        for g in genes:
            if isinstance(g, dict):
                rows.append(gene_to_row(g))

    out_df = pd.DataFrame(rows, columns=COLUMNS)
    out_df.to_csv(args.output_csv, index=False, encoding=args.output_encoding)
    print(f"完成：汇总 {len(out_df)} 行，输出到 {Path(args.output_csv).resolve()}")

if __name__ == "__main__":
    main()
