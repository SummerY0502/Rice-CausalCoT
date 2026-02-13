from pathlib import Path
import sys
import pandas as pd

CSV_FILE = Path("rapdb_salt_salinity_nacl_osmotic stress_abiotic stress_full.csv")
LIST_FILE = Path("gene_list_column.txt")
FILTERED_XLSX = Path("rapdb_salt_full_overlap_rows.xlsx")
OVERLAP_IDS_TXT = Path("overlap_ids.txt")

def read_id_list(path: Path) -> set[str]:
    ids = set()
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tok = s.split()[0]
            ids.add(tok.upper())
    return ids

def main():
    if not CSV_FILE.exists():
        print(f"CSV file not found: {CSV_FILE}"); sys.exit(1)
    if not LIST_FILE.exists():
        print(f"Gene list not found: {LIST_FILE}"); sys.exit(1)

    list_ids_upper = read_id_list(LIST_FILE)
    df = pd.read_csv(CSV_FILE, dtype=str)
    if "RAP_ID" not in df.columns:
        print("The ‘RAP_ID’ column was not found in the CSV file."); sys.exit(1)

    df["_RAP_ID_UPPER"] = df["RAP_ID"].astype(str).str.strip().str.upper()
    mask = df["_RAP_ID_UPPER"].isin(list_ids_upper)

    filtered = df.loc[mask].copy()

    filtered["_NONEMPTY"] = filtered.drop(columns=["_RAP_ID_UPPER"]).notna().sum(axis=1)

    filtered = (
        filtered.sort_index()
        .sort_values(["_RAP_ID_UPPER", "_NONEMPTY"], ascending=[True, False])
        .drop_duplicates(subset=["_RAP_ID_UPPER"], keep="first")
    )

    overlap_ids = sorted(filtered["_RAP_ID_UPPER"].unique())

    filtered = filtered.drop(columns=["_RAP_ID_UPPER", "_NONEMPTY"])

    filtered["Trait_Name"] = "salt tolerance"
    filtered["Trait_Description"] = "Tolerance to the high salt content in the growth medium"

    FILTERED_XLSX.parent.mkdir(parents=True, exist_ok=True)

    if "Index" in filtered.columns:
        filtered = filtered.drop(columns=["Index"])

    filtered = filtered.reset_index(drop=True)
    filtered.insert(0, "Index", range(1, len(filtered) + 1))
    # ==========================================
    filtered.to_excel(FILTERED_XLSX, index=False)

    with OVERLAP_IDS_TXT.open("w", encoding="utf-8") as f:
        for gid in overlap_ids:
            f.write(gid + "\n")

    print(f"Total number of rows in CSV: {len(df)}")
    print(f"Number of gene lists: {len(list_ids_upper)}")
    print(f"Number of intersections: {len(overlap_ids)}")
    print(f"The filtered XLSX file has been generated: {FILTERED_XLSX.resolve()} (Number of rows: {len(filtered)})")
    print(f"The list of intersection IDs has been generated: {OVERLAP_IDS_TXT.resolve()}")

if __name__ == "__main__":
    main()
