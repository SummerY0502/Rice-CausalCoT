import pandas as pd
import re

in_path = "data/Expression_salt_AS.csv"
out_path = "data/Expression_salt_AS_Gene.csv"

df = pd.read_csv(in_path)

transcript_pattern = re.compile(r"^OS\d{2}T\d{7}-\d{2}$", re.IGNORECASE)
expr_cols = [c for c in df.columns if transcript_pattern.match(c)]
trait_cols = [c for c in df.columns if c not in expr_cols]
assert len(expr_cols) > 0, "No expression quantity column that conforms to the transcript naming rule was found."

def transcript_to_gene(col_name: str) -> str:
    base = re.sub(r"-\d{2}$", "", col_name)  
    return base.replace("T", "G", 1)        

gene_cols = {c: transcript_to_gene(c) for c in expr_cols}

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

out_df = pd.concat([expr_gene, df[trait_cols]], axis=1)

out_df.to_csv(out_path, index=False)
print("输出保存至：", out_path)
