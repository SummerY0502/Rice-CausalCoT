import asyncio, re, csv, sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

RAP_ID_RE  = re.compile(r"Os\d{2}g\d{7}", re.I)
HITS_RE    = re.compile(r"\bof\s+(\d+)\s+hits\b", re.I)

DESIRED_RESULTS = "lid,tid,desc,pos,rgss,cgss,cgs,rgns,cgns,cgn,go"

# ===============================
def set_query(url: str, **updates) -> str:
    u = urlparse(url)
    q = {k: v[0] if isinstance(v, list) else v for k, v in parse_qs(u.query).items()}
    q.update({k: str(v) for k, v in updates.items() if v is not None})
    return urlunparse((u.scheme, u.netloc, u.path, u.params, urlencode(q, safe=","), u.fragment))

def normalize_rap_id(val: str) -> str:
    if not val:
        return ""
    m = RAP_ID_RE.search(val)
    return m.group(0).upper() if m else ""

def split_semicolon(x: str):
    if not x:
        return []
    return [t.strip() for t in x.split(";") if t and t.strip()]

def uniq_join(values):
    seen, out = set(), []
    for v in values:
        if not v:
            continue
        v = re.sub(r"\s+", " ", v.strip())
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return " ; ".join(out)

def merge_record(dst: dict, src: dict):

    def merge_semicolon_field(key):
        dst[key] = uniq_join(split_semicolon(dst.get(key, "")) + split_semicolon(src.get(key, "")))

    for k in ("Transcript_ID", "Gene_Symbol", "Gene_Name", "GO"):
        merge_semicolon_field(k)

    if not dst.get("Description") and src.get("Description"):
        dst["Description"] = src["Description"]
    if not dst.get("Position") and src.get("Position"):
        dst["Position"] = src["Position"]

def deduplicate_and_merge(rows):

    order = []
    by_id = {}
    counts = {}

    for r in rows:
        rid = normalize_rap_id(r.get("RAP_ID") or "")
        if not rid:
            rid = normalize_rap_id(r.get("Transcript_ID") or "")
        if not rid:
            continue

        r["RAP_ID"] = rid

        if rid not in counts:
            counts[rid] = 0
        counts[rid] += 1

        if rid not in by_id:
            base = {
                "RAP_ID": rid,
                "Transcript_ID": r.get("Transcript_ID", ""),
                "Description":   r.get("Description", ""),
                "Position":      r.get("Position", ""),
                "Gene_Symbol":   r.get("Gene_Symbol", ""),
                "Gene_Name":     r.get("Gene_Name", ""),
                "GO":            r.get("GO", ""),
            }
            by_id[rid] = base
            order.append(rid)
        else:
            merge_record(by_id[rid], r)

    dup_stats = {rid: c for rid, c in counts.items() if c > 1}
    merged_rows = [by_id[rid] for rid in order]
    return merged_rows, dup_stats

# =============================
async def wait_results_ready(page, timeout_ms=90_000):
    for sel in ["table.BasicTable tbody",
                ".PaginationInformation .paginationInformationText",
                "a[href*='/locus/?name=Os']"]:
        try:
            await page.wait_for_selector(sel, state="visible", timeout=timeout_ms)
            return
        except PWTimeout:
            pass
    raise PWTimeout("results not ready")

async def force_set_page_size_100(page) -> bool:
    selectors = [
        'select[name="nrow"]', 'form select[name="nrow"]',
        'select:has(option[value="100"])', '.PaginationInformation select', 'select',
    ]
    for sel in selectors:
        try:
            if await page.locator(sel).count():
                await page.select_option(sel, value="100")
                try:
                    await page.wait_for_load_state("networkidle", timeout=15_000)
                except Exception:
                    pass
                await wait_results_ready(page, 60_000)
                rows = await page.locator("table.BasicTable tbody tr").count()
                if rows >= 50:
                    return True
        except Exception:
            continue
    # 兜底：直接改 URL
    url100 = set_query(page.url, nrow=100, offset=0)
    if url100 != page.url:
        await page.goto(url100, wait_until="domcontentloaded")
        await wait_results_ready(page, 60_000)
        rows = await page.locator("table.BasicTable tbody tr").count()
        if rows >= 50:
            return True
    return False

async def parse_current_page(page):

    def clean_join(texts):
        texts = [re.sub(r"\s+", " ", (t or "").strip()) for t in texts if t and str(t).strip()]
        seen, out = set(), []
        for t in texts:
            if t not in seen:
                seen.add(t); out.append(t)
        return " ; ".join(out)

    async def cell_text_by_class(row, cls_selector: str) -> str:
        cell = row.locator(f"td.{cls_selector}")
        if not await cell.count():
            return ""
        links = await cell.locator("a").all_text_contents()
        return clean_join(links) if links else clean_join([await cell.inner_text()])

    async def merge_cells_by_classes(row, classes) -> str:
        vals = []
        for cls in classes:
            v = await cell_text_by_class(row, cls)
            if v:
                vals.extend([x.strip() for x in v.split(";") if x.strip()])
        return clean_join(vals)

    out = []
    rows = page.locator("table.BasicTable tbody tr")
    n = await rows.count()
    for i in range(n):
        row = rows.nth(i)

        # RAP_ID（若混杂则挑第一个匹配 Os##g#######）
        rap_id = await cell_text_by_class(row, "lid")
        if rap_id and not RAP_ID_RE.fullmatch(rap_id):
            lid_links = await row.locator("td.lid a").all_text_contents()
            rap_id2 = next((t.strip() for t in lid_links if RAP_ID_RE.fullmatch(t.strip())), "")
            rap_id = rap_id2 or rap_id

        gene_symbol = await merge_cells_by_classes(row, ["rgss", "cgss", "cgs"])
        gene_name   = await merge_cells_by_classes(row, ["rgns", "cgns", "cgn"])

        rec = {
            "RAP_ID":        rap_id,
            "Transcript_ID": await cell_text_by_class(row, "tid"),
            "Description":   await cell_text_by_class(row, "desc"),
            "Position":      await cell_text_by_class(row, "pos"),
            "Gene_Symbol":   gene_symbol,
            "Gene_Name":     gene_name,
            "GO":            "",
        }

        go_cell = row.locator('td[class*="go"]')
        if await go_cell.count():
            links = await go_cell.locator("a").all_text_contents()
            rec["GO"] = clean_join(links) or clean_join([await go_cell.inner_text()])

        out.append(rec)
    return out

async def crawl(keyword="salt", headless=True):
    all_rows = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        ctx = await browser.new_context(
            ignore_https_errors=True,
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0 Safari/537.36"),
            viewport={"width": 1366, "height": 900},
            locale="en-US",
        )
        page = await ctx.new_page()
        page.set_default_navigation_timeout(120_000)
        page.set_default_timeout(60_000)

        await page.goto("https://rapdb.dna.affrc.go.jp/search/", wait_until="domcontentloaded")
        await page.fill('form.formsearch02 input[name="q"]', keyword)
        async with page.expect_navigation(wait_until="domcontentloaded"):
            await page.click('form.formsearch02 input[type="submit"]')
        await wait_results_ready(page, 120_000)

        url_cols = set_query(page.url, results=DESIRED_RESULTS, offset=0)
        if url_cols != page.url:
            await page.goto(url_cols, wait_until="domcontentloaded")
            await wait_results_ready(page, 60_000)
        ok100 = await force_set_page_size_100(page)

        total_hits = 0
        try:
            t = await page.locator(".PaginationInformation .paginationInformationText").inner_text()
            m = HITS_RE.search(t);  total_hits = int(m.group(1)) if m else 0
        except Exception:
            pass
        rows_now = await page.locator("table.BasicTable tbody tr").count()
        page_size = 100 if ok100 else max(rows_now, 10)

        page_rows = await parse_current_page(page)
        all_rows.extend(page_rows)
        print(f"[{keyword}] [Page 1] rows≈{rows_now}, per_page={page_size}, got={len(page_rows)}, acc={len(all_rows)}, total_hits≈{total_hits or 'unknown'}")

        offset = page_size
        base_url = set_query(page.url, nrow=page_size, results=DESIRED_RESULTS, offset=None)
        loops = 0
        max_loops = 300
        while loops < max_loops:
            if total_hits and offset >= total_hits:
                break
            next_url = set_query(base_url, offset=offset)
            await page.goto(next_url, wait_until="domcontentloaded")
            try:
                await wait_results_ready(page, 90_000)
            except PWTimeout:
                break

            page_rows = await parse_current_page(page)
            if not page_rows:
                break
            all_rows.extend(page_rows)
            page_idx = offset // page_size + 1
            print(f"[{keyword}] [Page {page_idx}] got={len(page_rows)}, acc={len(all_rows)}")

            if not total_hits and len(page_rows) < page_size:
                break
            offset += page_size
            loops += 1

        await browser.close()

    return all_rows

async def crawl_many(keywords, headless=True):

    all_rows = []
    for kw in keywords:
        try:
            data = await crawl(kw, headless=headless)
            print(f"[{kw}] collected {len(data)} rows")
            all_rows.extend(data)
        except Exception as e:
            print(f"[{kw}] error: {e!r}")

    merged_rows, dup_stats = deduplicate_and_merge(all_rows)
    print(f"[MERGED] {len(merged_rows)} unique RAP_ID from {len(keywords)} keywords, duplicates={len(dup_stats)}")
    return merged_rows, dup_stats

# =============================
if __name__ == "__main__":

    argv_keywords = sys.argv[1:] or ["salt", "Salinity", "NaCl", "osmotic stress", "abiotic stress"]

    merged, dup_stats = asyncio.run(crawl_many(argv_keywords, headless=True))

    cols = ["Index", "RAP_ID", "Transcript_ID", "Description", "Position", "Gene_Symbol", "Gene_Name", "GO"]
    tag = "_".join([k.replace("/", "_") for k in argv_keywords]).lower()
    csv_name = f"rapdb_{tag}_full.csv"
    # ids_name = f"rapdb_{tag}_ids.txt"
    # dup_name = f"rapdb_{tag}_duplicates.txt"

    if merged:

        with open(csv_name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for idx, row in enumerate(merged, start=1):
                row_out = {"Index": idx}
                row_out.update({
                    "RAP_ID":        row.get("RAP_ID", ""),
                    "Transcript_ID": row.get("Transcript_ID", ""),
                    "Description":   row.get("Description", ""),
                    "Position":      row.get("Position", ""),
                    "Gene_Symbol":   row.get("Gene_Symbol", ""),
                    "Gene_Name":     row.get("Gene_Name", ""),
                    "GO":            row.get("GO", ""),
                })
                w.writerow(row_out)


