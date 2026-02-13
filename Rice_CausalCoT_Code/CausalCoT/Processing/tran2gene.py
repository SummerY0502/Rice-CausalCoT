import pandas as pd
import re

# 输入/输出路径
in_path = "data/Expression_salt_AS.csv"
out_path = "data/Expression_salt_AS_Gene.csv"

# 读取
df = pd.read_csv(in_path)

# 识别转录本列（示例：OS01T0343300-01）
transcript_pattern = re.compile(r"^OS\d{2}T\d{7}-\d{2}$", re.IGNORECASE)
expr_cols = [c for c in df.columns if transcript_pattern.match(c)]
trait_cols = [c for c in df.columns if c not in expr_cols]
assert len(expr_cols) > 0, "未找到符合转录本命名规则的表达量列。"

# 转录本→基因：OS01T0343300-01 → OS01G0343300
def transcript_to_gene(col_name: str) -> str:
    base = re.sub(r"-\d{2}$", "", col_name)  # 去掉 -01/-02 等转录本尾缀
    return base.replace("T", "G", 1)         # 首个 T 替换为 G

gene_cols = {c: transcript_to_gene(c) for c in expr_cols}

# 选择聚合方式：sum/mean/max（默认 sum）
AGG_METHOD = "sum"

expr = df[expr_cols].copy()
expr.columns = [gene_cols[c] for c in expr_cols]

if AGG_METHOD == "sum":
    expr_gene = expr.T.groupby(expr.columns).sum().T
elif AGG_METHOD == "mean":
    expr_gene = expr.groupby(expr.columns, axis=1).mean()
elif AGG_METHOD == "max":
    expr_gene = expr.groupby(expr.columns, axis=1).max()
else:
    raise ValueError("AGG_METHOD must be one of: sum, mean, max")

# 拼回性状列（不变）
out_df = pd.concat([expr_gene, df[trait_cols]], axis=1)

# 保存
out_df.to_csv(out_path, index=False)
print("输出保存至：", out_path)
