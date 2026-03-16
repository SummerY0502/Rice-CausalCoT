import pandas as pd
from pathlib import Path
import re

# ===== Using gene IDs, filter out the corresponding gene columns from the full gene expression dataset =====
gene_list_path = Path("data/ASGene.csv") 
expr_csv_path  = Path("data/Expression_INDICA_Labeled.csv")
out_filtered_csv = Path("data/Expression_salt_AS.csv")


out_match_log = Path("match_log.csv")
out_unmatched = Path("unmatched_genes.txt")

LOWERCASE = True            
MAP_g_TO_T = True           
STRIP_SUFFIX = True         
SUFFIX_PATTERN = r"([\-\.].+)$"  

def normalize_gene(s: str) -> str:
    if s is None:
        return ""
    x = str(s).strip()
    if STRIP_SUFFIX:
        x = re.sub(SUFFIX_PATTERN, "", x)  
    if MAP_g_TO_T:
        x = re.sub(r"g", "t", x)
        x = re.sub(r"G", "t", x)
    if LOWERCASE:
        x = x.lower()
    return x

def read_gene_list(path: Path) -> list[str]:
    try:
        df = pd.read_csv(path, index_col=0)
        idx = df.index
        if not (isinstance(idx, pd.RangeIndex)):  
            genes = [str(x).strip() for x in idx if pd.notna(x)]
            if any(re.search(r"[A-Za-z]", g) for g in genes):
                return genes
    except Exception:
        pass

    df = pd.read_csv(path)
    if df.shape[1] == 0:
        raise RuntimeError("The gene list file does not have any columns.")
    col0 = df.columns[0]
    genes = [str(x).strip() for x in df[col0].tolist() if pd.notna(x)]
    return genes

gene_names_raw = read_gene_list(gene_list_path)
gene_names_norm = [normalize_gene(g) for g in gene_names_raw]
gene_set_norm = set(gene_names_norm)

expr_df = pd.read_csv(expr_csv_path)
print("Expression of matrix dimensions:", expr_df.shape)
print("The first 5 columns:", expr_df.columns[:5].tolist())
print("Columns 16000 to 16005:", expr_df.columns[16000:16006].tolist())
print("The last 5 column names:", expr_df.columns[-5:].tolist())

if expr_df.shape[1] < 2:
    raise RuntimeError("The number of columns in the expression matrix is insufficient. At least one sample column and one gene column are required.")

first_col = expr_df.columns[0]
expr_gene_cols = list(expr_df.columns[1:])
# expr_gene_cols = list(expr_df.columns)

expr_cols_norm = [normalize_gene(c) for c in expr_gene_cols]

from collections import defaultdict
norm2orig = defaultdict(list)
for orig, normed in zip(expr_gene_cols, expr_cols_norm):
    norm2orig[normed].append(orig)

selected_cols = []
log_rows = []
unmatched = []

for raw, normed in zip(gene_names_raw, gene_names_norm):
    matched = norm2orig.get(normed, [])
    if matched:
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

seen = set()
selected_cols_ordered = []
for c in expr_gene_cols:
    if c in selected_cols and c not in seen:
        seen.add(c)
        selected_cols_ordered.append(c)

filtered_df = expr_df[[first_col] + selected_cols_ordered]
# filtered_df = expr_df[selected_cols_ordered]
filtered_df.to_csv(out_filtered_csv, index=False)

pd.DataFrame(log_rows).to_csv(out_match_log, index=False)

# with open(out_unmatched, "w", encoding="utf-8") as f:
#     for g in unmatched:
#         f.write(f"{g}\n")

print("==== Filtering completed ====")
print(f"Theoretical gene number：{len(gene_names_raw)}")
print(f"The total number of matched genes columns：{len(selected_cols_ordered)}")
print(f"output file：{out_filtered_csv}")
print(f"Matching log：{out_match_log}")
print(f"List of unmatched genes：{out_unmatched}")
if unmatched:
    print(f"Number of unmatched genes：{len(unmatched)}（unmatched_genes.txt）")
