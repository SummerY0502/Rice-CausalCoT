#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import sys
from typing import List, Dict, Any, Iterable

import pandas as pd
from openai import OpenAI
from typing import Literal

# =======================================================
MODEL = "gpt-5"
# REASONING_EFFORT = "low"  # minimal/low/medium/high
MAX_OUTPUT_TOKENS = 5000
MAX_RETRIES = 6
BATCH_SIZE = 1

DEFAULT_EXCEL = "../RAG/rapdb_salt_full_overlap_rows.xlsx"
OUTPUT_DIR = "outputs"
REASONING_EFFORT: Literal["minimal", "low", "medium", "high"] = "low"

# =============== JSON fenced block ===============
FENCE_RE = re.compile(r"```(?:json)?\s*(?P<body>[\s\S]*?)\s*```", re.IGNORECASE)


def extract_fenced_json(text: str) -> Any:
    if not text:
        raise ValueError("Empty model output.")
    m = FENCE_RE.search(text)
    if not m:
        raise ValueError("No fenced JSON block found.")
    raw = m.group("body")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raw2 = raw.strip("\ufeff").strip()
        return json.loads(raw2)


# ====================== OpenAI ========================
def make_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Please set OPENAI_API_KEY before running this script.")
    client = OpenAI(
        api_key="YOUR_API_KEY",
        base_url="YOUR_API_URL"
    )
    if not hasattr(client, "responses"):
        raise RuntimeError("Your 'openai' package is too old (no '.responses'). pip install -U openai")
    return client


def call_once(client: OpenAI, system_instruction: str, user_prompt: str) -> str:
    resp = client.responses.create(
        model=MODEL,
        instructions=system_instruction,
        input=user_prompt,
        tools=[{"type": "web_search"}],
        tool_choice="auto",
        reasoning={"effort": REASONING_EFFORT},
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    out = getattr(resp, "output_text", None)
    if out:
        return out

    # 兼容性兜底
    chunks = []
    for item in getattr(resp, "output", []) or []:
        content = getattr(item, "content", None)
        if not content:
            continue
        for c in content:
            t = getattr(c, "type", None) or (isinstance(c, dict) and c.get("type"))
            if t in ("output_text", "text"):
                val = getattr(c, "text", None)
                if isinstance(val, str):
                    chunks.append(val)
                elif isinstance(val, dict):
                    chunks.append(val.get("value", ""))
                elif isinstance(c, dict):
                    tv = c.get("text")
                    if isinstance(tv, dict):
                        chunks.append(tv.get("value", ""))
                    elif isinstance(tv, str):
                        chunks.append(tv)
    if chunks:
        return "".join(chunks)
    return str(resp)


def call_with_retries(client: OpenAI, system_instruction: str, user_prompt: str) -> str:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return call_once(client, system_instruction, user_prompt)
        except Exception as e:
            last_err = e
            wait = min(2 ** attempt, 30)
            print(f"[warn] attempt {attempt}/{MAX_RETRIES} failed: {e} -> retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"OpenAI request failed after {MAX_RETRIES} attempts: {last_err}")


# ====================== Prompt ========================

SYSTEM_INSTRUCTION = """
You are an expert in crop molecular biology, specializing in determining causal relationships between rice (Oryza sativa) genes and salt tolerance traits. Act as a meticulous reviewer and evidence synthesizer.

CRITICAL RULES:
- Keep organism/trait explicit: Oryza sativa/rice + salt/salinity/NaCl.
- Use authoritative QTL/annotation resources, PubMed & Google Scholar. DO NOT hallucinate PMIDs/DOIs.
- Be conservative when evidence is mixed. Process genes strictly in provided order, do NOT skip.
- STRICT OUTPUT: Return ONE fenced JSON block (```json ... ```), per batch, following the JSON CONTRACT; no extra text.
- Inputs per gene come from file fields: {Position}, {GO}, {RAP_ID}, {Gene_Symbol}, {Gene_Name}, {Description}. Use these exact identifiers for database/literature searches.
- Finally, based on the evidence labels of genes  (S1–S5), perform harmonized prior.

Let’s reason step by step:
S1.QTL localization (use gene_position{Position} vs salt-related QTL intervals):
- S1a: Gene position overlaps the reported main-effect rice salt/salinity QTL interval(s) (including the fixed main salt QTL chr1: 11.1–14.6 Mb), supported by credible QTL database records or peer-reviewed mapping studies.
- S1b: Gene position does NOT overlap any credible rice salt/salinity QTL interval.

S2.Functional mechanism analysis:
- Identify functional mechanisms using {GO}/{Description} terms, and authoritative functional databases.
- S2a: Established to function within pivotal regulatory factors or central salt-tolerance modules.
- S2b: Does not satisfy any of the direct evidence standards under S2a.

S3.Pathway analysis:
- Identify salt/salinity-relevant pathways by searching KEGG/Plant Reactome (and other authoritative pathway resources) using {RAP_ID}/{Gene_Symbol}/{Gene_Name} and trait context.
- S3a: Evidence for the presence of components associated with salt tolerance pathways.
- S3b: Lack of pathway components required for S3a and direct evidence.

S4.Homolog/ortholog functional validation:
- Identify homologous genes in model crops/related crops.
- S4a: Homologous gene function influences salt tolerance traits.
- S4b: No qualifying S4a homolog evidence.

S5.Direct experimental evidence in rice:
- Search MUST be done in BOTH PubMed and Google Scholar with strict query patterns such as:
  - "Oryza sativa" AND (salt OR salinity OR NaCl) AND ({Gene_Symbol} OR {RAP_ID} OR {Gene_Name})
- S5a: At least ONE study found that satisfies ALL:
  1. verifiable PMID or DOI;
  2. salt/salinity/NaCl keyword appears in title/abstract;
  3. causal genetic validation performed in rice itself (e.g., complementation, knockout/knockdown, overexpression, precise allele replacement/editing) that directly changes rice salt tolerance phenotype.
- S5b: If PubMed AND Google Scholar do not yield any study meeting ALL S5a conditions, or evidence is only expression association/transcriptomics/QTL inference/heterologous system.

- Harmonized Prior:
  1. Statistics on evidence labels in steps S1-S5;
  2. Conservative: If S5a or (S1a and S2a and S3a and S4a) is satisfied, Conservative=Yes; otherwise, Conservative=No;
  3. Exploratory: If S1a or S2a or S3a or S4a or S5a is satisfied, Exploratory=Yes; otherwise, Exploratory=No.

JSON CONTRACT (single batch):
{
  "batch_meta": {"batch_index": <int>, "num_genes": <int>},
  "genes": [
    {
      "rap_id": "...", "gene_symbol": "...", "gene_name": "...",
      "trait_name": "...", "description": "...",
      "position": "...", "go": "...",
      "S1": {"label": "S1a|S1b", "reason": "..."},
      "S2": {"label": "S2a|S2b", "reason": "..."},
      "S3": {"label": "S3a|S3b", "reason": "..."},
      "S4": {"label": "S4a|S4b", "reason": "..."},
      "S5": {"label": "S5a|S5b", "reason": "...", "pmids": [], "dois": []},
      "Conservative": "Yes|No", "reason": "..."},
      "Exploratory": "Yes|No", "reason": "..."},
      "summary": "...",
      "sources": [{"title":"...","url":"...","year":2021}]
    }
  ]
}

"""

USER_PROMPT_TEMPLATE = """
You are an expert in crop molecular biology, specializing in determining causal relationships between rice genes and salt tolerance traits.
Using the TOP226_info.xlsx dataset rows provided below, sequentially evaluate ALL gene–salt tolerance trait pairs in strict file order.
Process EXACTLY these {N} genes in this round and do NOT omit any. Fill unknowns using your knowledge base or references; be rigorous and avoid hallucinations.

Task (S1–S5) exactly as specified by the JSON CONTRACT in the system instruction.

Genes in this batch (N={N}, maintain order):
{GENE_BULLETS}
"""


def build_gene_bullets(rows: List[Dict[str, Any]]) -> str:
    bullets = []
    for i, r in enumerate(rows, 1):
        bullets.append(
            f"{i}. {r.get('RAP_ID', '')} | {r.get('Gene_Symbol', '')} | {r.get('Gene_Name', '')}\n"
            f"   - Position: {r.get('Position', '')}\n"
            f"   - GO: {r.get('GO', '')}\n"
            f"   - Description: {r.get('Description', '')}\n"
            f"   - Trait_Name: {r.get('Trait_Name', '')}\n"
            f"   - Trait_Description: {r.get('Trait_Description', '')}"
        )
    return "\n".join(bullets)


# ==============================================================
REQUIRED_COLUMNS = [
    "RAP_ID", "Gene_Symbol", "Gene_Name", "Description",
    "GO", "Position", "Trait_Name", "Trait_Description"
]


def read_excel_rows(excel_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel not found: {excel_path}")
    df = pd.read_excel(excel_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Excel is missing required columns: {missing}")
    for c in REQUIRED_COLUMNS:
        df[c] = df[c].fillna("").astype(str)
    return df[REQUIRED_COLUMNS].to_dict(orient="records")


def chunk_iter(it: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(it), size):
        yield it[i:i + size]


# ================================================================
def run_click_to_run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        rows = read_excel_rows(DEFAULT_EXCEL)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

    total = len(rows)
    print(f"[info] Loaded {total} genes from {DEFAULT_EXCEL}", file=sys.stderr)

    client = make_client()

    all_genes: List[Dict[str, Any]] = []
    batches_meta: List[Dict[str, Any]] = []
    batch_index = 0

    for batch in chunk_iter(rows, BATCH_SIZE):
        batch_index += 1
        gene_bullets = build_gene_bullets(batch)
        user_prompt = USER_PROMPT_TEMPLATE.format(N=len(batch), GENE_BULLETS=gene_bullets)

        print(f"[info] Processing batch {batch_index} with {len(batch)} genes ...", file=sys.stderr)
        raw_text = call_with_retries(client, SYSTEM_INSTRUCTION, user_prompt)

        raw_out_path = os.path.join(OUTPUT_DIR, f"batch_{batch_index:03d}_raw.txt")
        with open(raw_out_path, "w", encoding="utf-8") as f:
            f.write(raw_text)
        print(f"[info] Saved raw output -> {raw_out_path}", file=sys.stderr)

        try:
            json_obj = extract_fenced_json(raw_text)
        except Exception as e:
            err_path = os.path.join(OUTPUT_DIR, f"batch_{batch_index:03d}_parse_error.txt")
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(str(e))
            print(f"[warn] Failed to parse fenced JSON for batch {batch_index}: {e}", file=sys.stderr)
            continue

        genes = json_obj.get("genes") or []
        all_genes.extend(genes)
        meta = json_obj.get("batch_meta") or {"batch_index": batch_index, "num_genes": len(genes)}
        batches_meta.append(meta)

        if len(genes) != len(batch):
            print(f"[warn] Batch {batch_index}: expected {len(batch)} genes, got {len(genes)}", file=sys.stderr)

    total_genes = len(all_genes)
    result = {
        "meta": {
            "input_excel": DEFAULT_EXCEL,
            "total_batches": len(batches_meta),
            "total_genes": total_genes,
            "batches": batches_meta
        },
        "genes": all_genes
    }

    sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2))
    sys.stdout.flush()
    print(f"\n[success] Printed aggregated JSON with {total_genes} genes to stdout", file=sys.stderr)


if __name__ == "__main__":
    run_click_to_run()
