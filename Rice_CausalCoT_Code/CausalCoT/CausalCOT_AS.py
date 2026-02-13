#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import math
import subprocess
import importlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Set
from typing import Literal
import re

try:
    import pandas as pd
except Exception:
    print("Missing dependency: pandas. Please install: pip install pandas", file=sys.stderr)
    sys.exit(1)


# =================================================================
STEP6_PATH = "outputs/step_6.txt"
BIO_CSV_PATH = "data/Bio_Result.csv"
TRUE_GENE_PATH = "data/CausalGene_148.txt"


# =================================================================
GENE_RE = re.compile(r"OS\d{2}G\d{7,8}", re.IGNORECASE)
# ================= auto-install & imports =================
REASONING_EFFORT: Literal["minimal", "low", "medium", "high"] = "low"

def _read_csv_robust(path: str) -> "pd.DataFrame":
    last_err = None
    for enc in ("gbk", "utf-8", "utf-8-sig"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


def norm_gene(g: str) -> str:
    return str(g).strip().upper()


def read_universe_from_bio_first_col(bio_csv: str) -> Set[str]:
    if not os.path.exists(bio_csv):
        raise FileNotFoundError(f"Bio_Result.csv not found: {bio_csv}")

    df = _read_csv_robust(bio_csv)
    if df.shape[1] < 1:
        raise ValueError("Bio_Result.csv has no columns.")

    col0 = df.iloc[:, 0].astype(str).tolist()
    header = norm_gene(str(df.columns[0]))

    genes = set()
    for x in col0:
        x = norm_gene(x)
        if not x or x in ("NAN", "NONE"):
            continue
        if x == header:
            continue

        hits = GENE_RE.findall(x)
        if hits:
            for h in hits:
                genes.add(norm_gene(h))
        else:
            # genes.add(x)
            pass

    return genes


def read_true_genes(true_file: str) -> Set[str]:
    if not os.path.exists(true_file):
        raise FileNotFoundError(f"CausalGene_148.txt not found: {true_file}")

    with open(true_file, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()

    genes = {norm_gene(g) for g in GENE_RE.findall(txt)}
    return genes


def parse_predicted_genes_from_step6(step6_path: str) -> Set[str]:
    if not os.path.exists(step6_path):
        raise FileNotFoundError(f"step_6.txt not found: {step6_path}")

    with open(step6_path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()

    txt_u = txt.upper()

    m = re.search(r"\bE\s*=\s*\{\s*(.*?)\s*\}\s*,\s*BIC\s*=", txt_u, flags=re.S)
    if not m:
        m = re.search(r"\bE\s*=\s*\{\s*(.*?)\s*\}", txt_u, flags=re.S)
    if not m:
        raise ValueError("Cannot find 'E={...}' block in step_6.txt")

    e_block = m.group(1)

    genes = re.findall(r"(OS\d{2}G\d{7,8})\s*(?:→|->)\s*LABEL", e_block, flags=re.I)
    return {norm_gene(g) for g in genes}


def evaluate(predicted: Set[str], truth_total: Set[str], universe: Set[str]) -> Dict[str, Any]:
    universe = {norm_gene(g) for g in universe}
    predicted = {norm_gene(g) for g in predicted} & universe
    truth_total = {norm_gene(g) for g in truth_total}
    truth_in_universe = truth_total & universe

    tp = len(predicted & truth_in_universe)
    fp = len(predicted - truth_in_universe)
    fn = len(truth_in_universe - predicted)
    tn = len(universe - (predicted | truth_in_universe))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / len(universe) if universe else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    if len(predicted) > 0 and len(truth_in_universe) > 0 and len(universe) > 0:
        enrichment = (tp / len(predicted)) / (len(truth_in_universe) / len(universe))
    else:
        enrichment = float("nan")

    return {
        "universe_size": len(universe),
        "predicted_size": len(predicted),
        "truth_size_total": len(truth_total),
        "truth_size_in_universe": len(truth_in_universe),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy,
        "F1": f1,
        "Enrichment": enrichment,
        "overlap_genes": sorted(predicted & truth_in_universe),
    }


def _ensure(pip_name: str, import_name: str = None):
    name = import_name or pip_name
    try:
        return importlib.import_module(name)
    except Exception:
        print(f"[setup] Installing {pip_name} ...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
        return importlib.import_module(name)


np = _ensure("numpy")
pd = _ensure("pandas")
sm = _ensure("statsmodels", "statsmodels.api")
_ = _ensure("python-dotenv", "dotenv")
from dotenv import load_dotenv

load_dotenv()

try:
    openai_mod = importlib.import_module("openai")
    from openai import OpenAI
except Exception:
    OpenAI = None

# ================= utils =================


def ensure_outdir(path: str = "outputs") -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_text(path: str, text: str) -> None:

    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_json(path: str, obj: Any) -> None:
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ================= data structures =================

@dataclass
class Dataset:
    N: int
    D: int
    G: List[str]
    T: List[str]
    G_p: List[str]
    G_r: List[str]
    X: pd.DataFrame
    y: pd.Series


def load_and_prepare(expr_csv: str, bio_csv: str) -> Dataset:
    if not os.path.exists(expr_csv) or not os.path.exists(bio_csv):
        print("Please place merged_querysample_plus_INDICA.csv and Bio_Result.csv in the same directory as the script and try again.", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(expr_csv, encoding="gbk")
    if 'Label' not in df.columns:
        print("The expression matrix is missing the ‘Label’ column.", file=sys.stderr)
        sys.exit(3)
    G = [c for c in df.columns if c != 'Label']
    N = len(df)
    D = len(df.columns)

    bio = pd.read_csv(bio_csv, encoding="gbk")
    cols = {c.lower(): c for c in bio.columns}
    gid_col = cols.get('gene_id') or cols.get('geneid') or list(bio.columns)[0]
    cons_col = cols.get('conservative')
    expl_col = cols.get('exploratory')

    tmp_cols = [gid_col] + [c for c in [cons_col, expl_col] if c]
    tmp = bio[tmp_cols].copy()
    tmp.columns = ['Gene_ID'] + (["Conservative"] if cons_col else []) + (["Exploratory"] if expl_col else [])
    tmp['Gene_ID_norm'] = tmp['Gene_ID'].astype(str).str.strip().str.lower()

    def is_yes(v: str) -> bool:
        v = str(v).strip().lower()
        return v in ('yes', 'y', 'true', '1')

    cons_map = {r['Gene_ID_norm']: is_yes(r.get('Conservative', 'No')) for _, r in tmp.iterrows()}
    expl_map = {r['Gene_ID_norm']: is_yes(r.get('Exploratory', 'No')) for _, r in tmp.iterrows()}

    G_norm = [g.lower() for g in G]
    G_id_map = {g.lower(): g for g in G}
    G_p, G_r = [], []
    for g in G_norm:
        go = G_id_map[g]
        cons = cons_map.get(g, False)
        expl = expl_map.get(g, False)
        if cons:
            G_p.append(go)
        if cons or expl:
            G_r.append(go)

    X = df[G].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    y = df['Label'].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    return Dataset(N=N, D=D, G=G, T=['Label'], G_p=G_p, G_r=G_r, X=X, y=y)


@dataclass
class Graph:
    edges: List[Tuple[str, str]]

    def parents_of(self, t: str) -> List[str]:
        return [g for (g, tt) in self.edges if tt == t]

    def add_edge(self, g: str, t: str):
        es = set(self.edges)
        es.add((g, t))
        return Graph(sorted(es))

    def remove_edge(self, e: Tuple[str, str]):
        return Graph(sorted([x for x in self.edges if x != e]))


@dataclass
class FitResult:
    graph: Graph
    gene_stats: Dict[str, Dict[str, float]]
    trait_stats: Dict[str, Dict[str, Any]]
    ll_total: float
    k_total: int
    bic: float

# ================= math =================


def gaussian_ll_mle(x) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    mu = float(np.mean(x))
    var = float(np.mean((x - mu) ** 2))
    if var <= 0:
        var = 1e-12
    ll = -0.5 * N * (math.log(2 * math.pi * var) + 1.0)
    return mu, var, ll


def logistic_ll_and_beta(y, X) -> Tuple[np.ndarray, float]:
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    Xd = sm.add_constant(X, has_constant='add')
    model = sm.Logit(y, Xd)
    try:
        res = model.fit(disp=0, maxiter=200)
        beta = res.params
        ll = float(res.llf)
    except Exception:
        res = model.fit_regularized(alpha=1e-6, maxiter=500, disp=0)
        beta = res.params
        z = Xd @ beta
        p = 1.0 / (1.0 + np.exp(-z))
        eps = 1e-12
        ll = float(np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
    return beta, ll


def fit_and_bic(ds: Dataset, gph: Graph) -> FitResult:
    gene_stats: Dict[str, Dict[str, float]] = {}
    ll_genes = 0.0
    for g in ds.G:
        mu, var, ll = gaussian_ll_mle(ds.X[g].values)
        gene_stats[g] = {"mu": mu, "var": var, "ll": ll}
        ll_genes += ll

    trait_stats: Dict[str, Dict[str, Any]] = {}
    ll_traits = 0.0
    k_traits = 0
    for t in ds.T:
        parents = gph.parents_of(t)
        X = ds.X[parents].values if parents else np.zeros((ds.N, 0))
        beta, ll = logistic_ll_and_beta(ds.y.values, X)
        trait_stats[t] = {
            "parents": parents,
            "beta": beta.tolist(),
            "ll": ll,
            "num_params": 1 + len(parents),
        }
        ll_traits += ll
        k_traits += (1 + len(parents))

    ll_total = ll_genes + ll_traits
    k_genes = 2 * len(ds.G)
    k_total = k_genes + k_traits
    bic = ll_total - 0.5 * k_total * math.log(ds.N)
    return FitResult(graph=gph, gene_stats=gene_stats, trait_stats=trait_stats, ll_total=ll_total, k_total=k_total, bic=bic)

# ================= candidate enumeration =================

def forward_candidates(ds: Dataset, gph: Graph) -> List[Graph]:
    cands: List[Graph] = []
    for t in ds.T:
        used = set(gph.parents_of(t))
        for g in ds.G_r:
            if g not in used:
                cands.append(gph.add_edge(g, t))
    return cands


def backward_candidates(gph: Graph) -> List[Graph]:
    return [gph.remove_edge(e) for e in gph.edges]

# ================= prompt (verbatim) =================

PROMPT_STEP1 = """
Note
1. Variable conventions: N = sample size, D = number of variables, G = gene list, T = trait list, E = current graph edge set (directed, gene→trait), C = candidate graph set, initialize t=1;
2. Perform logistic regression on trait nodes;
3. No terms required for log-likelihood or parameter estimation are omitted throughout the process, but only necessary values and expressions are output; lengthy derivations are not included.

Step 1: Data and Indexing
Given: Read external tables merged_querysample_plus_INDICA.csv and Bio_Result.csv (containing Conservative/Exploratory status for each gene, Yes/No), where Label denotes traits and remaining columns represent genes.
Do:
1. Load merged_querysample_plus_INDICA.csv data, remove headers, and set sample size N as the number of rows. Let variable count D be the number of columns, resulting in G (all gene columns) and T (Label column only).
3. Load Bio_Result.csv data, read Gene_ID, Conservative, and Exploratory for all genes. Find the intersection between Gene_ID and G (case-insensitive). Set G_p = {g ∈ G | Conservative(g) = Yes} and G_r = {g ∈ G | Conservative(g) = Yes or Exploratory(g) = Yes} (treat missing/non-standard values as No).
Return: N, D, G, T, G_p, G_r (using specific values and names, with complete output).
Example output format: N=?, D=?, G=[...], T=[...], G_p=[...], G_r=[...]
"""

PROMPT_STEP2_1 = """
Step 2: Greedy Forward Search
Step 2.1 Generate Candidate Graph
Given: Existing E, G, T, G_r.
Do: If a gene is not in the table, it is considered unsatisfied (removed). Based on the current graph E, add only one new valid edge g→t (g ∈ G_r\\Pa(T), t ∈ T) that does not yet exist, enumerating all candidate edges; for each candidate edge, form a new edge set E' = E ∪ {g→t}.
Return: List each candidate graph: the edge set of E' (list each gene→trait pair sequentially, and write the complete, uninterrupted result to a text file).
Example output format: C=[{E':(g1→t)},{E':(g2→t)},...] (list each item without omission)
"""

PROMPT_STEP2_2 = r"""
Step 2.2 Calculate BIC for Each Candidate Graph
Given: Each candidate graph E'.
Do (execute independently for each candidate graph and output sequentially):
1. Gene nodes (without parents): For each gene column, compute the Gaussian marginal Xg: sample mean μg, variance σg² (MLE, divided by N), log-likelihood

\begin{equation}\ell_g = -\frac{N}{2}\ln(2\pi\sigma_g^2) - \frac{N}{2}\end{equation}
Number of parameters: 2 per gene (mean, variance).
2. Trait Node (Parent-Dependent):
Perform logistic regression for each t ∈ T using all incoming genes as independent variables to estimate parameter β via maximum likelihood. Calculate:

\[
\ell_3 = \sum_{i=1}^{10} \left[ y_i \ln(p_i) + (1 - y_i)\ln(1 - p_i) \right],   p_i = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_{1,i} + \beta_2 x_{2,i})}}
\]
Number of parameters：1 + |Pa(t)|。
3. Total Log-Likelihood:

\[
\ell_{\text{total}}=\sum_{g\in G}\ell_g+\sum_{t\in T}\ell_t
]
4. Total number of parameters k: The cumulative count of parameters across all nodes.
5.BIC：

\[
\mathrm{BIC}=\ell_{\text{total}}-\frac{k}{2}\ln N
]
Return：For each candidate graph, output the following items: E', the necessary statistics and log-likelihood for each node, k, BIC, and β.
Example Output Format (Per Candidate Graph)：
Graph: E'={(g→t), ...}
BIC=?
"""

PROMPT_STEP2_3 = """
Step 2.3 Selection and Convergence Check
Given: The BIC of the current graph E (calculate the BIC of the empty graph/current graph per Step 2.2 if necessary), and the BIC values of all candidate graphs from Step 2.2.
Do: Select the candidate graph E* with the highest BIC and its score BIC*. If BIC* > BIC(E), update E←E*.
Return: Current graph E (note accuracy of gene name output, whether genes originate from G_r), corresponding BIC*, and whether E was updated.
Example output format: t=?, best=E*, BIC*=?, updated=(yes|no)
"""

PROMPT_STEP3 = """
Step 3:
Given: Existing E, G, T, G_r.
Do: Let t ← t + 1. Execute only one round of Step 2.1–2.3 (forward pass) on the current graph E. Strictly execute the steps of Step 2.1–2.3 in sequence as required.
Return: The current graph E (note the accuracy of gene name output and whether genes originate from G_r), BIC, and the regression coefficients β for each trait node. Provide a sorted list based on |β| (descending order) and indicate whether Step 2 updated E.
Example Output Format: E={...}, BIC=?, β=[...], updated=(yes|no)
"""

PROMPT_STEP4 = r"""
Step 4: Greedy Backward Search
Given: Current graph E.
Do: Generate all candidate graphs E' = E\{e} by removing one existing edge. For each E', compute BIC as in Section 2.2. Compare with E's BIC; if BIC(E') > BIC(E), take the larger value to update E and proceed to the next iteration (j←j+1). Otherwise, the backward search converges.
Return: All candidates E' from this round with their respective BICs, the optimal candidate and whether updated, and convergence status.
Example Output Format: j=?, candidates=[(E'1,BIC1),...], best=..., updated=(yes|no)
"""

PROMPT_STEP5 = """
Step 5: Iterative Greedy Backward Search
Given the current graph E.
Do: Execute one round of Step 4 (backward) on the current E.
Return: Current graph E (note accuracy of gene name output—whether genes originate from G_r), BIC, and regression coefficients β for each trait node. Provide a sorted list by |β| (descending order) and indicate whether Step 3 was updated.
Example output format: E={...}, BIC=?, β=[...] , updated=(yes|no)
"""

PROMPT_STEP6 = """
Step 6: Obtain the final causal structure diagram
Given the current graph E.
Do: On the current graph E, fully add directed edges from all genes in the conserved set G_p (g ∈ G_p) to the trait t, skipping existing edges. Subsequently, re-estimate parameters on E according to Step 2.2 and compute BIC.
Return: The current graph E, BIC, and the regression coefficient β for each trait node. Additionally, provide a ranked list of “gene-trait influence strength” sorted by |β| (descending order).
Example output format: E={...}, BIC=?, β=[...]
"""

# ================= LLM (per-step prompt + local results) =================


def run_llm_step(prompt_verbatim: str, local_result: str) -> str:

    full_input = f"""{prompt_verbatim}

Below are the intermediate results already computed by the local Python instance based on the data and graph structure.
Please **do not alter any values or recalculate anything**. Using only these results,
organize the final output strictly according to the “Do / Return / Example Output Format” specified above.
Local results are as follows:

```TEXT
{local_result}
```"""

    if OpenAI is None:
        return local_result

    api_key = "YOUR_API_KEY"
    base_url = "YOUR_API_URL"

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        resp = client.responses.create(
            model="gpt-5",
            instructions=(
                "You are an assistant that strictly outputs according to the specified format."
                "You must strictly adhere to the Return/example output format provided by the user."

            ),
            input=full_input,
            reasoning={"effort": REASONING_EFFORT},
            max_output_tokens=5000,
        )
        out = getattr(resp, "output_text", None)
        return out or local_result

    except Exception as e:
        return local_result

# ================= formatting helpers (local) =================


def format_step1(ds: Dataset) -> str:
    return (
        "Step 1 Results: Data and Indexes\n"
        f"N = {ds.N}, D = {ds.D}\n"
        f"G(All genes, totaling {len(ds.G)}) = {ds.G}\n"
        f"T (Characteristic) = {ds.T}\n"
        f"G_p (conservative set) = {ds.G_p}\n"
        f"G_r (candidate gene set) = {ds.G_r}\n"
    )


def edges_to_str(edges: List[Tuple[str, str]]) -> str:
    return ", ".join([f"{g}→{t}" for (g, t) in edges])


def format_step2_1(cands: List[Graph]) -> str:
    """
    1. Write all candidate images in full to the file path;
    2. Simultaneously provide the complete C=[{E':(...)},...] list in the return string,
       ensuring the LLM has sufficient information to output according to the example format.
    """
    cand_items = []
    for c in cands:
        # 这里按示例格式：{E':(g1→t,g2→t,...)}
        edges_str = edges_to_str(c.edges)
        cand_items.append(f"{{E':({edges_str})}}")

    c_list_str = ", ".join(cand_items)

    return (
        "Step 2.1 Result: Generate forward candidate map\n"
        f"Number of candidate images  = {len(cands)}\n"
        f"C=[{c_list_str}]\n"
    )


def format_step2_2(ds: Dataset, evals: List[FitResult]) -> str:
    if not evals:
        return "Step 2.2 Result: No candidate images available for evaluation.\n"

    best = max(evals, key=lambda r: r.bic)

    return (
        "Step 2.2 Result: BIC calculation for each candidate graph is complete.\n"
        f"Number of Candidate Graphs = {len(evals)}\n"
        f"Optimal Candidate Edge Set E* = {{ {edges_to_str(best.graph.edges)} }}\n"
        f"Corresponding BIC* = {best.bic:.6g}\n"
    )


def format_step2_3(curr_fit: FitResult, best_fit: FitResult) -> str:
    updated = best_fit.bic > curr_fit.bic
    return (
        "Step 2.3 Result: Select the optimal candidate and determine whether to update\n"
        f"BIC for the current graph E = {curr_fit.bic:.6g}\n"
        f"Optimal Candidate Graph E* = {{ {edges_to_str(best_fit.graph.edges)} }}，"
        f"BIC* = {best_fit.bic:.6g}\n"
        f"Should the current map be updated?：{'yes' if updated else 'no'}\n"
    )


def beta_sorted_list(fr: FitResult) -> List[str]:
    pairs = []
    for t, ts in fr.trait_stats.items():
        for i, b in enumerate(ts['beta']):
            name = 'Intercept' if i == 0 else ts['parents'][i - 1]
            pairs.append((name, abs(b), b))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [f"{name}:{b:.6g}" for name, _, b in pairs]


def format_step3(fr: FitResult, updated_flag: bool = False) -> str:
    return (
        "Step 3 Result: Forward Search Iteration\n"
        f"Current Graph E = {{ {edges_to_str(fr.graph.edges)} }}\n"
        f"BIC = {fr.bic:.6g}\n"
        f"Sort in descending order by |β| = {beta_sorted_list(fr)}\n"
        f"Whether an update has occurred in this round：{'yes' if updated_flag else 'no'}\n"
    )


def format_step4(j: int, evals_b: List[FitResult], curr_best: FitResult) -> Tuple[str, bool, FitResult]:
    if not evals_b:
        text = (
            "Step 4 Result: Greedy Backward Search\n"
            f"Round {j}: No edges can be removed from the current graph; backward search ends.\n"
        )
        return text, False, curr_best

    best = max(evals_b, key=lambda r: r.bic)
    updated = best.bic > curr_best.bic

    text = (
        "Step 4 Result: Greedy Backward Search\n"
        f"Number of Candidate Graphs in Round {j} = {len(evals_b)}\n"
        f"Current Graph BIC = {curr_best.bic:.6g}\n"
        f"Best Candidate Graph E* = {{ {edges_to_str(best.graph.edges)} }}，BIC* = {best.bic:.6g}\n"
        f"Whether to update the current graph：{'yes' if updated else 'no'}\n"
    )
    return text, updated, best


def format_step5(fr: FitResult, updated_flag: bool) -> str:
    return (
        "Step 5 Result: One round of repeated backward search\n"
        f"Current Graph E = {{ {edges_to_str(fr.graph.edges)} }}\n"
        f"BIC = {fr.bic:.6g}\n"
        f"Sort in descending order by |β| = {beta_sorted_list(fr)}\n"
        f"Whether an update has occurred in this round：{'yes' if updated_flag else 'no'}\n"
    )


def format_step6(fr: FitResult) -> str:
    return (
        "Step 6 Result: Final causal structure graph\n"
        f"Final Graph E = {{ {edges_to_str(fr.graph.edges)} }}\n"
        f"BIC = {fr.bic:.6g}\n"
        f"Sort in descending order by |β| = {beta_sorted_list(fr)}\n"
    )

# ================= main =================


def main():
    ensure_outdir()
    expr = 'data/Expression_salt_AS_Gene.csv'        #data/merged_querysample_plus_INDICA.csv
    bio = 'data/Bio_Result.csv'
    ds = load_and_prepare(expr, bio)

    # Step 1
    text1_local = format_step1(ds)
    step1_text = run_llm_step(PROMPT_STEP1, text1_local)
    save_text('outputs/step_1.txt', step1_text)

    # Initialize
    E = Graph(edges=[])
    fit_E = fit_and_bic(ds, E)

    # Step 2.1 - forward candidates
    cands1 = forward_candidates(ds, E)
    text21_local = format_step2_1(cands1)
    step2_1_text = run_llm_step(PROMPT_STEP2_1, text21_local)
    save_text('outputs/step_2_1.txt', step2_1_text)

    # Step 2.2 - evaluate candidates
    evals1 = [fit_and_bic(ds, c) for c in cands1]
    text22_local = format_step2_2(ds, evals1)
    step2_2_text = run_llm_step(PROMPT_STEP2_2, text22_local)
    save_text('outputs/step_2_2.txt', step2_2_text)

    # Step 2.3 - pick best
    best1 = max(evals1, key=lambda r: r.bic) if evals1 else fit_E
    if best1.bic > fit_E.bic:
        E = best1.graph
        fit_E = best1
    text23_local = format_step2_3(fit_E, best1)
    step2_3_text = run_llm_step(PROMPT_STEP2_3, text23_local)
    save_text('outputs/step_2_3.txt', step2_3_text)

    step3_round = 1
    while True:
        cands = forward_candidates(ds, E)

        if not cands:
            text3_local = format_step3(fit_E, updated_flag=False)
            step3_round_text = run_llm_step(PROMPT_STEP3, text3_local)
            save_text(f'outputs/step_3_round{step3_round}.txt', step3_round_text)
            break
        evals = [fit_and_bic(ds, c) for c in cands]
        best = max(evals, key=lambda r: r.bic)
        updated = best.bic > fit_E.bic
        text3_local = format_step3(best if updated else fit_E, updated_flag=updated)
        step3_round_text = run_llm_step(PROMPT_STEP3, text3_local)
        save_text(f'outputs/step_3_round{step3_round}.txt', step3_round_text)

        if updated:
            E = best.graph
            fit_E = best
            step3_round += 1
            continue
        else:
            break

    text3_local = format_step3(fit_E)
    step3_final_text = run_llm_step(PROMPT_STEP3, text3_local)
    save_text('outputs/step_3.txt', step3_final_text)

    j = 1
    changed = False
    while True:
        cands_b = backward_candidates(E)
        if not cands_b:
            text4_local = (
                f"j={j}\n"
                "candidates=[]\n"
                f"best=E*={{ {edges_to_str(E.edges)} }}, BIC*={fit_E.bic:.6g}, updated=no\n"
            )
            step4_text = run_llm_step(PROMPT_STEP4, text4_local)
            save_text(f'outputs/step_4_round{j}.txt', step4_text)
            break

        evals_b = [fit_and_bic(ds, c) for c in cands_b]
        text4_local, updated, best_b = format_step4(j, evals_b, fit_E)
        step4_text = run_llm_step(PROMPT_STEP4, text4_local)
        save_text(f'outputs/step_4_round{j}.txt', step4_text)

        if updated:
            E = best_b.graph
            fit_E = best_b
            changed = True
            j += 1
            continue
        else:
            break

    cands_b2 = backward_candidates(E)
    updated5 = False
    if cands_b2:
        evals_b2 = [fit_and_bic(ds, c) for c in cands_b2]
        best_b2 = max(evals_b2, key=lambda r: r.bic)
        if best_b2.bic > fit_E.bic:
            E = best_b2.graph
            fit_E = best_b2
            updated5 = True
    text5_local = format_step5(fit_E, updated5)
    step5_text = run_llm_step(PROMPT_STEP5, text5_local)
    save_text('outputs/step_5.txt', step5_text)

    final_edges = set(E.edges)
    for t in ds.T:
        for g in ds.G_p:
            final_edges.add((g, t))
    E_final = Graph(sorted(list(final_edges)))
    fit_final = fit_and_bic(ds, E_final)
    text6_local = format_step6(fit_final)
    step6_text = run_llm_step(PROMPT_STEP6, text6_local)
    save_text('outputs/step_6.txt', step6_text)

    result = {
        "E": E_final.edges,
        "BIC": fit_final.bic,
        "beta": fit_final.trait_stats,
        "N": ds.N,
        "D": ds.D,
        "G": ds.G,
        "T": ds.T,
        "G_p": ds.G_p,
        "G_r": ds.G_r,
    }
    save_json('outputs/final_result.json', result)

    print("[done] The complete process has been executed, and the results have been written to outputs/.", flush=True)

###################################################
    step6 = STEP6_PATH if os.path.exists(STEP6_PATH) else "data/step_6.txt"
    bio = BIO_CSV_PATH if os.path.exists(BIO_CSV_PATH) else "data/Bio_Result.csv"
    truef = TRUE_GENE_PATH if os.path.exists(TRUE_GENE_PATH) else "data/CausalGene_148.txt"

    universe = read_universe_from_bio_first_col(bio)
    predicted = parse_predicted_genes_from_step6(step6)
    truth_total = read_true_genes(truef)

    metrics = evaluate(predicted, truth_total, universe)

    print("\n[eval] Candidate vs True Causal Genes", flush=True)
    print(f"[eval] Universe (related genes) from Bio_Result.csv first column = {metrics['universe_size']}", flush=True)
    print(f"[eval] Predicted candidates = {metrics['predicted_size']}", flush=True)
    print(f"[eval] True causal genes total = {metrics['truth_size_total']}", flush=True)
    print(f"[eval] True causal genes in universe (used for metrics) = {metrics['truth_size_in_universe']}", flush=True)
    print(f"[eval] TP={metrics['TP']}, FP={metrics['FP']}, FN={metrics['FN']}, TN={metrics['TN']}", flush=True)
    print(f"[eval] Precision = {metrics['Precision']:.6f}", flush=True)
    print(f"[eval] Recall    = {metrics['Recall']:.6f}", flush=True)
    print(f"[eval] Accuracy  = {metrics['Accuracy']:.6f}", flush=True)
    print(f"[eval] F1        = {metrics['F1']:.6f}", flush=True)
    print(f"[eval] Enrichment= {metrics['Enrichment']:.6f}", flush=True)

    overlap = metrics["overlap_genes"]
    if overlap:
        show_k = min(50, len(overlap))
        print(f"[eval] Overlap genes (showing {show_k}/{len(overlap)}): {overlap[:show_k]}", flush=True)
    else:
        print("[eval] Overlap genes: none", flush=True)


if __name__ == '__main__':
    main()
