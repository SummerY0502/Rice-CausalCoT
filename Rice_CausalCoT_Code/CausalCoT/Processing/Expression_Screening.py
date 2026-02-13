import pandas as pd
from pathlib import Path
import re

# ===== Using gene IDs, filter out the corresponding gene columns from the full gene expression dataset =====
gene_list_path = Path("data/ASGene.csv")  #380Gene.csv
expr_csv_path  = Path("data/Expression_INDICA_Labeled.csv")
out_filtered_csv = Path("data/Expression_salt_AS.csv")


out_match_log = Path("match_log.csv")
out_unmatched = Path("unmatched_genes.txt")

# 规范化选项
LOWERCASE = True            # 忽略大小写
MAP_g_TO_T = True           # 将基因ID中的 'g' / 'G' 统一规范为 't'（你之前的需求）
STRIP_SUFFIX = True         # 去掉转录本后缀（如 -xx / .xx 等）
SUFFIX_PATTERN = r"([\-\.].+)$"  # 匹配去除的后缀（例：-00、-01、.v2 等）

# ============ 工具函数 ============
def normalize_gene(s: str) -> str:
    """将基因ID规范到统一形式，便于相等匹配。"""
    if s is None:
        return ""
    x = str(s).strip()
    if STRIP_SUFFIX:
        x = re.sub(SUFFIX_PATTERN, "", x)  # 去掉 -xx / .xx 等
    if MAP_g_TO_T:
        # 仅替换“字母 g/G”为 t，但不对数字做任何处理
        # 如果想更精准，可限定在 Os\d{2}[gG]\d+ 这类模式上再替换
        x = re.sub(r"g", "t", x)
        x = re.sub(r"G", "t", x)
    if LOWERCASE:
        x = x.lower()
    return x

def read_gene_list(path: Path) -> list[str]:
    """
    读取基因清单：
    - 若第一列是基因ID，则取第一列；
    - 若索引即是基因ID（且不是纯数字的 RangeIndex），也支持。
    """
    # 尝试：第一列为索引
    try:
        df = pd.read_csv(path, index_col=0)
        idx = df.index
        if not (isinstance(idx, pd.RangeIndex)):  # 不是 0..N-1 的默认索引
            genes = [str(x).strip() for x in idx if pd.notna(x)]
            # 如果这些索引看起来像基因ID（不是全数字）
            if any(re.search(r"[A-Za-z]", g) for g in genes):
                return genes
    except Exception:
        pass

    # 回退：普通读取，从第一列取
    df = pd.read_csv(path)
    if df.shape[1] == 0:
        raise RuntimeError("基因清单文件没有任何列。")
    col0 = df.columns[0]
    genes = [str(x).strip() for x in df[col0].tolist() if pd.notna(x)]
    return genes

# ============ 主流程 ============
# 1) 读取基因清单并规范化
gene_names_raw = read_gene_list(gene_list_path)
gene_names_norm = [normalize_gene(g) for g in gene_names_raw]
gene_set_norm = set(gene_names_norm)

# 2) 读取表达矩阵
expr_df = pd.read_csv(expr_csv_path)
print("表达矩阵维度：", expr_df.shape)
print("前 5 列：", expr_df.columns[:5].tolist())
print("第 16000~16005 列名：", expr_df.columns[16000:16006].tolist())
print("最后 5 列名：", expr_df.columns[-5:].tolist())

if expr_df.shape[1] < 2:
    raise RuntimeError("表达矩阵列数不足，至少需要1个样本列 + 1个基因列。")

first_col = expr_df.columns[0]
expr_gene_cols = list(expr_df.columns[1:])
# expr_gene_cols = list(expr_df.columns)

# 3) 把表达矩阵的列名也做相同规范化
expr_cols_norm = [normalize_gene(c) for c in expr_gene_cols]

# 建立 规范名 -> 原始列名 的映射（可能一对多，因此用列表）
from collections import defaultdict
norm2orig = defaultdict(list)
for orig, normed in zip(expr_gene_cols, expr_cols_norm):
    norm2orig[normed].append(orig)

# 4) 相等匹配（规范化后）
selected_cols = []
log_rows = []
unmatched = []

for raw, normed in zip(gene_names_raw, gene_names_norm):
    matched = norm2orig.get(normed, [])
    if matched:
        # 保持表达矩阵原始列的顺序（后面会统一去重）
        selected_cols.extend(matched)
        log_rows.append({
            "gene_raw": raw,
            "gene_norm": normed,
            "n_matched_cols": len(matched),
            "matched_cols": ";".join(matched),
        })
    else:
        unmatched.append(raw)
        log_rows.append({
            "gene_raw": raw,
            "gene_norm": normed,
            "n_matched_cols": 0,
            "matched_cols": "",
        })

# 去重并保持原顺序
seen = set()
selected_cols_ordered = []
for c in expr_gene_cols:
    if c in selected_cols and c not in seen:
        seen.add(c)
        selected_cols_ordered.append(c)

# 5) 写结果
filtered_df = expr_df[[first_col] + selected_cols_ordered]
# filtered_df = expr_df[selected_cols_ordered]
filtered_df.to_csv(out_filtered_csv, index=False)

pd.DataFrame(log_rows).to_csv(out_match_log, index=False)

# with open(out_unmatched, "w", encoding="utf-8") as f:
#     for g in unmatched:
#         f.write(f"{g}\n")

print("==== 筛选完成 ====")
print(f"理论基因数：{len(gene_names_raw)}")
print(f"匹配到的基因列总计：{len(selected_cols_ordered)}")
print(f"输出文件：{out_filtered_csv}")
print(f"匹配日志：{out_match_log}")
print(f"未匹配基因名单：{out_unmatched}")
if unmatched:
    print(f"未匹配基因数：{len(unmatched)}（详见 unmatched_genes.txt）")
