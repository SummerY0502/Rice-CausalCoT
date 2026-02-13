#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import difflib
import gzip
import io
import os
import sys
from typing import Tuple, List

try:
    import pandas as pd
except ImportError:
    print("Pandas is required. Please install it first: pip install pandas", file=sys.stderr)
    sys.exit(1)

# ==================
INPUT_ALL_EXPR = "Expression_gene_all.csv"
TARGET_GENE = "OS06G0665500"                          # OS01G0104100  OS01G0100100 OS06G0665500
RIGHT_MATRIX  = "../Expression_salt_AS_Gene.csv"
INTERMEDIATE_QUERY = "querysample.csv"
OUTPUT_FILE  = "../merged_querysample_plus_INDICA.csv"
WRITE_INTERMEDIATE = False
DEFAULT_ENCODING = "utf-8"


def open_maybe_gzip(path: str, mode: str = "rt", encoding: str = DEFAULT_ENCODING):
    if path.endswith(".gz"):
        return gzip.open(path, mode=mode, encoding=encoding, newline="")
    return open(path, mode=mode, encoding=encoding, newline="")


def sniff_delimiter_and_header(path: str, sample_size: int = 65536) -> Tuple[str, List[str]]:
    with open_maybe_gzip(path, "rt") as fh:
        sample = fh.read(sample_size)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";"])
            delimiter = dialect.delimiter
        except csv.Error:
            if sample.count("\t") >= max(sample.count(","), sample.count(";")):
                delimiter = "\t"
            elif sample.count(";") >= sample.count(","):
                delimiter = ";"
            else:
                delimiter = ","

        fh.seek(0)
        first_line = fh.readline()
        header = next(csv.reader(io.StringIO(first_line), delimiter=delimiter))

    header = [h.strip() for h in header]
    return delimiter, header


def sniff_delimiter(path: str, sample_size: int = 65536) -> str:
    with open_maybe_gzip(path, "rt") as fh:
        sample = fh.read(sample_size)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";"])
            return dialect.delimiter
        except csv.Error:
            if sample.count("\t") >= max(sample.count(","), sample.count(";")):
                return "\t"
            if sample.count(";") >= sample.count(","):
                return ";"
            return ","


def read_table(path: str) -> pd.DataFrame:
    sep = sniff_delimiter(path)
    try:
        return pd.read_csv(
            path,
            sep=sep,
            header=0,
            compression="infer",
            encoding=DEFAULT_ENCODING,
        )
    except Exception as e:
        print(f"Read failed: {path}\nError: {e}", file=sys.stderr)
        sys.exit(2)


def extract_target_gene_column(expr_path: str, gene: str) -> pd.DataFrame:
    if not os.path.exists(expr_path):
        print(f"Input file not found: {expr_path}", file=sys.stderr)
        sys.exit(1)

    try:
        delimiter, header = sniff_delimiter_and_header(expr_path)
    except Exception as e:
        print(f"Failed to automatically recognize delimiters/headers: {e}", file=sys.stderr)
        sys.exit(1)

    if gene not in header:
        candidates = difflib.get_close_matches(gene, header, n=10, cutoff=0.6)
        msg = f"Gene {gene} not found in the header."
        if candidates:
            msg += "\nSimilar candidates: " + ", ".join(candidates)
        print(msg, file=sys.stderr)
        sys.exit(2)

    try:
        df = pd.read_csv(
            expr_path,
            sep=delimiter,
            header=0,
            usecols=[gene],
            compression="infer",
            encoding=DEFAULT_ENCODING,
        )
    except Exception as e:
        print(f"Failed to read target gene sequence: {e}", file=sys.stderr)
        sys.exit(3)

    return df


def main():

    if not os.path.exists(RIGHT_MATRIX):
        print(f"The matrix file to be concatenated cannot be found: {RIGHT_MATRIX}", file=sys.stderr)
        sys.exit(1)
    df_right = read_table(RIGHT_MATRIX)

    if TARGET_GENE in df_right.columns:
        print(f"The matrix on the right already contains the gene columns. {TARGET_GENE}")
        try:
            df_right.to_csv(OUTPUT_FILE, index=False)
        except Exception as e:
            print(f"Write about failure: {e}", file=sys.stderr)
            sys.exit(5)
        print(f"Output complete: {OUTPUT_FILE}\nNumber of samples: {df_right.shape[0]}  Number of columns: {df_right.shape[1]}")
        return

    df_query = extract_target_gene_column(INPUT_ALL_EXPR, TARGET_GENE)

    if df_query.shape[0] != df_right.shape[0]:
        print(
            "The number of lines (sample count) in the two files does not match, preventing column-by-column concatenation: \n"
            f"- querysample({TARGET_GENE}) Number of rows: {df_query.shape[0]}\n"
            f"- {RIGHT_MATRIX} Number of rows: {df_right.shape[0]}",
            file=sys.stderr,
        )
        sys.exit(4)

    if WRITE_INTERMEDIATE:
        try:
            df_query.to_csv(INTERMEDIATE_QUERY, index=False)
        except Exception as e:
            print(f"Failed to write intermediate file: {e}", file=sys.stderr)
            sys.exit(4)

    merged = pd.concat(
        [df_query.reset_index(drop=True), df_right.reset_index(drop=True)],
        axis=1
    )

    try:
        merged.to_csv(OUTPUT_FILE, index=False)
    except Exception as e:
        print(f"Write about failure: {e}", file=sys.stderr)
        sys.exit(5)

    print(
        "Merge completed: {} New gene columns: {} Number of columns on the right: {} Total number of columns: {} Number of samples: {}".format(
            OUTPUT_FILE, TARGET_GENE, df_right.shape[1], merged.shape[1], merged.shape[0]
        )
    )


if __name__ == "__main__":
    main()
