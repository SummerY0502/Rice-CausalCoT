#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import re
import json
import time
import math
import sys
import csv
from typing import Any, Dict, Optional, Tuple, Literal

import pandas as pd
from openai import OpenAI

# ==================================================
MODEL = "gpt-5"
MAX_OUTPUT_TOKENS = 2000
MAX_RETRIES = 6
REASONING_EFFORT: Literal["minimal", "low", "medium", "high"] = "low"

# 你的文件路径（按需改成你本地实际路径）
STEP6_PATH = "outputs/step_6.txt"
BIO_RESULT_CSV_PATH = "data/Bio_Result.csv"

TARGET_GENE = "OS06G0665500"
OUTPUT_CSV = "gene_causal_explanation.csv"

# =============== JSON fenced block 抽取 ===============
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

# ====================================================================
def make_client() -> OpenAI:

    API_KEY = "YOUR_API_KEY"
    BASE_URL = "YOUR_API_URL"

    if not API_KEY or API_KEY.startswith("xxxx"):
        raise RuntimeError("Please enter the correct API_KEY in the script (or modify it to read from an environment variable).")
    if not BASE_URL or BASE_URL.startswith("xxxx"):
        raise RuntimeError("Please enter the correct BASE_URL in the script.")

    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )
    if not hasattr(client, "responses"):
        raise RuntimeError("Your ‘openai’ package is outdated (lacks .responses). Please run pip install -U openai.")
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

    # 兼容性兜底：从 resp.output 拼 text
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

# ====================== 本地文件信息提取 ======================
def read_text_file(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    for enc in ("utf-8", "utf-8-sig", "gbk", "latin1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("Unable to read step_6.txt using common encodings.", b"", 0, 1, "decode failed")

def extract_beta_from_step6(step6_text: str, gene_id: str) -> float:

    m = re.search(rf"'{re.escape(gene_id)}:(-?\d+(?:\.\d+)?)'", step6_text)
    if m:
        return float(m.group(1))

    # 兜底：不带引号也试试
    m2 = re.search(rf"{re.escape(gene_id)}:(-?\d+(?:\.\d+)?)", step6_text)
    if m2:
        return float(m2.group(1))

    raise ValueError(f"The β value for gene {gene_id} was not found in step_6.txt.")

def read_bio_result_row(csv_path: str, gene_id: str) -> Tuple[str, str, str, str]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = None
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030", "latin1"):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except UnicodeDecodeError as e:
            last_err = e
            continue
    if df is None:
        raise last_err or RuntimeError("Failed to read Bio_Result.csv")

    required = ["RAP_ID", "S2_keyword", "S3_keywords", "S4_note", "S5_keyinfo"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Bio_Result.csv is missing columns: {missing};Current columns: {list(df.columns)}")

    hit = df[df["RAP_ID"].astype(str).str.strip() == gene_id]
    if hit.empty:
        raise ValueError(f"No row found in Bio_Result.csv where RAP_ID == {gene_id}.")

    row = hit.iloc[0]
    s2 = str(row.get("S2_keyword", "")).strip()
    s3 = str(row.get("S3_keywords", "")).strip()
    s4 = str(row.get("S4_note", "")).strip()
    s5 = str(row.get("S5_keyinfo", "")).strip()
    return s2, s3, s4, s5

# ==================================================================
SYSTEM_INSTRUCTION = """
You are a bioinformatics expert. You must strictly follow the “Let's reason step by step” procedure provided by the user to complete the inference.
Important: Do not fabricate non-existent document content; inferences must be based solely on the genetic information fields and beta values provided by the user.
Strict Output: Return only a fenced JSON (```json ... ```) without any additional text.

JSON Output Format:
{
  "gene_id": "OS06G0665500",
  "beta": -5.01938,
  "or": 0.0066,
  "mechanism_module": "xxx",
  "causal_explanation": "xxx"
}
"""

def build_user_prompt(
    gene_id: str,
    beta: float,
    or_value: float,
    s2_keyword: str,
    s3_keywords: str,
    s4_note: str,
    s5_keyinfo: str
) -> str:

    return f"""You are a bioinformatics expert. In the step_6.txt file, locate the {gene_id} gene and its corresponding causal effect β. Output the final causal explanation using the prompt provided below. Save the gene ID, mechanism module, and causal explanation in a .csv file.
Let’s reason step by step:
1. Can you locate the {gene_id} gene and its corresponding causal effect β based on the Step6.txt file?
2. Can the [gene information] (molecular mechanism S2_keyword, signaling pathway S3_keywords, homologous gene function S4_note, key information from literature S5_keyinfo) for the {gene_id} gene in Bio_Result.csv be retrieved?
3. Can we infer the most relevant [mechanism module] for the gene based on the potential mechanisms linking {gene_id} and rice salt tolerance? If multiple [mechanism modules] are involved, should we prioritize information from S5_keyinfo to identify the most relevant [mechanism module]?
For example: If the gene [gene information] includes antioxidant/ROS scavenging, then it can be inferred that the [mechanism module] is the ROS scavenging/antioxidant network.
4. Can we infer the specific biological effects of each gene on rice salt tolerance based on its [gene information] and [mechanistic modules]?
For example: If a gene's [gene information] is ROS clearance and the redox pathway, and its [mechanism module] is ROS clearance/antioxidant network, then it can be inferred that the gene's [biological effect] is to enhance antioxidant capacity/ROS clearance capacity, reduce ROS accumulation and oxidative damage under salt stress, thereby improving rice salt tolerance.
5. Can a causal biological interpretation be provided based on the biological effects of the genes and their β values in the Step6.txt file?
Output Example 1 (Fill in relevant content based on the above results):
The direct causal effect of [Gene Name] is OR = exp(β) = [Numerical Value]. A [Numerical Value] (>1|<1) indicates increased expression, enhancing|weakening salt tolerance. Its function corresponds to that of [Mechanism Module], specifically exerting the [Biological Effect], thereby contributing to rice salt tolerance.

【The information extracted from the local file is as follows (please infer based on this; do not fabricate):】
- step_6.txt: β for {gene_id} = {beta}
- calculated by the program OR = exp(β) = {or_value}

- Bio_Result.csv ({gene_id} rows):
  - S2_keyword = {s2_keyword}
  - S3_keywords = {s3_keywords}
  - S4_note     = {s4_note}
  - S5_keyinfo  = {s5_keyinfo}

【Requirements】
- The output must be fenced JSON (```json ... ```), containing the fields gene_id, beta, or, mechanism_module, causal_explanation.
- The causal explanation must strictly adhere to the sentence structure and logic of “Output Example 1.”
"""

# =========================================================
def write_result_csv(path: str, gene_id: str, mechanism_module: str, causal_explanation: str) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["Gene_ID", "Mechanism_Module", "Causal_Explanation"])
        writer.writeheader()
        writer.writerow({
            "Gene_ID": gene_id,
            "Mechanism_Module": mechanism_module,
            "Causal_Explanation": causal_explanation
        })

    print(f"\n[Gene_ID] {gene_id}")
    print(f"[Mechanism_Module] {mechanism_module}")
    print(f"[Causal_Explanation] {causal_explanation}")

# ====================== MAIN ======================
def main():
    client = make_client()

    step6_text = read_text_file(STEP6_PATH)
    beta = extract_beta_from_step6(step6_text, TARGET_GENE)

    or_value = math.exp(beta)

    s2, s3, s4, s5 = read_bio_result_row(BIO_RESULT_CSV_PATH, TARGET_GENE)

    user_prompt = build_user_prompt(
        gene_id=TARGET_GENE,
        beta=beta,
        or_value=round(or_value, 6),
        s2_keyword=s2,
        s3_keywords=s3,
        s4_note=s4,
        s5_keyinfo=s5
    )

    raw = call_with_retries(client, SYSTEM_INSTRUCTION, user_prompt)

    obj = extract_fenced_json(raw)
    gene_id = obj.get("gene_id", TARGET_GENE)
    mechanism_module = obj.get("mechanism_module", "").strip()
    causal_explanation = obj.get("causal_explanation", "").strip()

    if not mechanism_module or not causal_explanation:
        raise ValueError(f"Model output JSON fields are incomplete: {obj}")

    write_result_csv(OUTPUT_CSV, gene_id, mechanism_module, causal_explanation)
    print(f"[success] Output to {OUTPUT_CSV}", file=sys.stderr)


if __name__ == "__main__":
    main()
