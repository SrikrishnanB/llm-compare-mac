"""
sentence_compare_poc.py -- Multi-type message comparison: load -> pair -> LLM judge -> report.

Reads SOD, EOD and Previsit CSV files from docs/Hyde/ and docs/SF/,
consolidates into a single DataFrame keyed by (dsr_id, date, source),
pairs rows between sources, and sends each pair to qwen3:14b for scoring.

Folder layout:
    docs/
      Hyde/   <- dsr_message_start_of_day_*.csv
               dsr_message_end_of_day_*.csv
               dsr_message_previsit_*.csv
      SF/     <- same file patterns

Usage
-----
    python sentence_compare_poc.py
    python sentence_compare_poc.py --docs /path/to/docs
    python sentence_compare_poc.py --model qwen3:30b --workers 6
    python sentence_compare_poc.py --out results.csv
"""

import argparse
import csv
import json
import sys
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ollama
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL      = "qwen3:8b"
DEFAULT_WORKERS    = 4
DEFAULT_BATCH_SIZE = 4
DEFAULT_DOCS_DIR   = "docs"

SOURCE_HYDE = "hyde"
SOURCE_SF   = "sf"

FILE_GLOBS = {
    "sod":      "dsr_message_start_of_day_*.csv",
    "eod":      "dsr_message_end_of_day_*.csv",
    "previsit": "dsr_message_previsit_*.csv",
}

_DSR_COLS = {"message_date", "message_html", "dsr_mavic_id", "dsr_code", "dsr_name"}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class LLMScore(BaseModel):
    template_match_pct: int = Field(..., ge=0, le=100)
    data_match_pct:     int = Field(..., ge=0, le=100)
    template_note:      str = Field(default="")
    data_note:          str = Field(default="")


class BatchResult(BaseModel):
    results: list[LLMScore]


class ScoredRow(BaseModel):
    msg_type:           str
    dsr_id:             str
    dsr_code:           str
    dsr_name:           str
    date:               str
    outlet_mavic_id:    str
    outlet_name:        str
    template_match_pct: int
    data_match_pct:     int
    template_note:      str
    data_note:          str


class OutletDiff(BaseModel):
    dsr_id:            str
    dsr_name:          str
    date:              str
    common_count:      int
    only_hyde_count:   int
    only_sf_count:     int
    only_hyde_outlets: list[str]
    only_sf_outlets:   list[str]


class TypeSummary(BaseModel):
    count:            int
    avg_template_pct: float
    avg_data_pct:     float


class RunSummary(BaseModel):
    by_type: dict[str, TypeSummary]
    overall: TypeSummary


# ---------------------------------------------------------------------------
# SECTION 1: Data loading -> consolidated DataFrame
# ---------------------------------------------------------------------------

def _find_subfolder(base: Path, name: str) -> Path | None:
    for child in base.iterdir():
        if child.is_dir() and child.name.lower() == name:
            return child
    return None


def _load_type_from_folder(folder: Path, msg_type: str, source_label: str) -> pd.DataFrame:
    files = sorted(folder.glob(FILE_GLOBS[msg_type]))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, dtype=str, keep_default_na=False)
            missing = _DSR_COLS - set(df.columns)
            if missing:
                print(f"[warn] {f.name}: missing columns {missing} -- skipped.", flush=True)
                continue
            df["source"]      = source_label
            df["msg_type"]    = msg_type
            df["source_file"] = f.name
            frames.append(df)
        except Exception as exc:
            print(f"[warn] Failed to read {f.name}: {exc}", flush=True)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_consolidated_df(docs_dir: str = DEFAULT_DOCS_DIR) -> pd.DataFrame:
    """
    Load all message types from both source folders. Returns one row per
    (dsr_id, date, source) with columns:
        dsr_id, dsr_code, dsr_name, date, source,
        sod_html, eod_html,
        previsit_outlets  -- list[dict] {outlet_mavic_id, outlet_code, outlet_name, html}
    """
    base = Path(docs_dir)
    hyde_dir = _find_subfolder(base, SOURCE_HYDE)
    sf_dir   = _find_subfolder(base, SOURCE_SF)

    if hyde_dir is None:
        print(f"[error] No 'Hyde' folder found inside {base.resolve()}")
        sys.exit(1)
    if sf_dir is None:
        print(f"[error] No 'SF' folder found inside {base.resolve()}")
        sys.exit(1)

    raw_frames = []
    for source_label, folder in [(SOURCE_HYDE, hyde_dir), (SOURCE_SF, sf_dir)]:
        for msg_type in FILE_GLOBS:
            df = _load_type_from_folder(folder, msg_type, source_label)
            if not df.empty:
                raw_frames.append(df)
                print(f"[info] {source_label.upper()} / {msg_type}: {len(df)} rows loaded.", flush=True)

    if not raw_frames:
        print("[error] No data found in any folder.")
        sys.exit(1)

    raw = pd.concat(raw_frames, ignore_index=True)
    raw["dsr_id"] = raw["dsr_mavic_id"].str.strip()
    raw["date"]   = raw["message_date"].str.strip()
    raw["html"]   = raw["message_html"].str.strip()
    for col in ("dsr_code", "dsr_name", "source"):
        raw[col] = raw[col].str.strip()

    records: dict[tuple, dict] = {}

    for _, row in raw.iterrows():
        key = (row["dsr_id"], row["date"], row["source"])
        if key not in records:
            records[key] = {
                "dsr_id":           row["dsr_id"],
                "dsr_code":         row["dsr_code"],
                "dsr_name":         row["dsr_name"],
                "date":             row["date"],
                "source":           row["source"],
                "sod_html":         "",
                "eod_html":         "",
                "previsit_outlets": [],
            }

        rec = records[key]

        if row["msg_type"] == "sod":
            if not rec["sod_html"]:
                rec["sod_html"] = row["html"]
            else:
                print(f"[warn] Duplicate SOD for {key} -- keeping first.", flush=True)

        elif row["msg_type"] == "eod":
            if not rec["eod_html"]:
                rec["eod_html"] = row["html"]
            else:
                print(f"[warn] Duplicate EOD for {key} -- keeping first.", flush=True)

        elif row["msg_type"] == "previsit":
            outlet_id = row.get("outlet_mavic_id", "").strip()
            if not outlet_id:
                print(f"[warn] Previsit row for {key} has no outlet_mavic_id -- skipped.", flush=True)
                continue
            existing_ids = {o["outlet_mavic_id"] for o in rec["previsit_outlets"]}
            if outlet_id not in existing_ids:
                rec["previsit_outlets"].append({
                    "outlet_mavic_id": outlet_id,
                    "outlet_code":     row.get("outlet_code", "").strip(),
                    "outlet_name":     row.get("outlet_name", "").strip(),
                    "html":            row["html"],
                })

    consolidated = pd.DataFrame(list(records.values()))

    coverage     = consolidated.groupby(["dsr_id", "date"])["source"].nunique()
    paired_count = (coverage == 2).sum()
    orphan_count = (coverage == 1).sum()
    if orphan_count:
        print(f"[info] {orphan_count} (dsr_id, date) group(s) have data in only one source -- excluded from comparison.", flush=True)
    print(f"[info] {paired_count} (dsr_id, date) group(s) have data in both sources.", flush=True)

    return consolidated


# ---------------------------------------------------------------------------
# SECTION 2: Build comparison pairs
# ---------------------------------------------------------------------------

def build_comparison_pairs(df: pd.DataFrame) -> tuple[list[dict], list[OutletDiff]]:
    """
    For each (dsr_id, date) present in both sources:
      - One SOD pair (if both have sod_html)
      - One EOD pair (if both have eod_html)
      - N previsit pairs (one per outlet_mavic_id matched in both)
      - One OutletDiff record per (dsr_id, date) with any previsit data
    """
    pairs:        list[dict]       = []
    outlet_diffs: list[OutletDiff] = []

    for (dsr_id, date), group in df.groupby(["dsr_id", "date"]):
        sources = set(group["source"].tolist())
        if SOURCE_HYDE not in sources or SOURCE_SF not in sources:
            continue

        hyde_row = group[group["source"] == SOURCE_HYDE].iloc[0]
        sf_row   = group[group["source"] == SOURCE_SF].iloc[0]

        base = {
            "dsr_id":   dsr_id,
            "dsr_code": hyde_row["dsr_code"] or sf_row["dsr_code"],
            "dsr_name": hyde_row["dsr_name"] or sf_row["dsr_name"],
            "date":     date,
        }

        # SOD
        if hyde_row["sod_html"] and sf_row["sod_html"]:
            pairs.append({**base, "msg_type": "sod", "outlet_mavic_id": "", "outlet_name": "",
                          "html_a": hyde_row["sod_html"], "html_b": sf_row["sod_html"]})

        # EOD
        if hyde_row["eod_html"] and sf_row["eod_html"]:
            pairs.append({**base, "msg_type": "eod", "outlet_mavic_id": "", "outlet_name": "",
                          "html_a": hyde_row["eod_html"], "html_b": sf_row["eod_html"]})

        # Previsit -- outlet-level matching by outlet_mavic_id
        hyde_outlets = {o["outlet_mavic_id"]: o for o in hyde_row["previsit_outlets"]}
        sf_outlets   = {o["outlet_mavic_id"]: o for o in sf_row["previsit_outlets"]}

        if hyde_outlets or sf_outlets:
            common_ids = set(hyde_outlets) & set(sf_outlets)
            only_hyde  = set(hyde_outlets) - set(sf_outlets)
            only_sf    = set(sf_outlets)   - set(hyde_outlets)

            outlet_diffs.append(OutletDiff(
                dsr_id            = dsr_id,
                dsr_name          = base["dsr_name"],
                date              = date,
                common_count      = len(common_ids),
                only_hyde_count   = len(only_hyde),
                only_sf_count     = len(only_sf),
                only_hyde_outlets = [hyde_outlets[i]["outlet_name"] for i in only_hyde],
                only_sf_outlets   = [sf_outlets[i]["outlet_name"]   for i in only_sf],
            ))

            if only_hyde or only_sf:
                print(f"[info] Previsit ({dsr_id} / {date}): "
                      f"{len(common_ids)} matched, {len(only_hyde)} only-Hyde, {len(only_sf)} only-SF.",
                      flush=True)

            for outlet_id in sorted(common_ids):
                h = hyde_outlets[outlet_id]
                s = sf_outlets[outlet_id]
                pairs.append({**base, "msg_type": "previsit",
                              "outlet_mavic_id": outlet_id, "outlet_name": h["outlet_name"],
                              "html_a": h["html"], "html_b": s["html"]})

    return pairs, outlet_diffs


# ---------------------------------------------------------------------------
# SECTION 3: LLM scoring
# ---------------------------------------------------------------------------

_SCORING_INSTRUCTIONS = """\
You are comparing pairs of sales briefing messages sent to the same field sales executive.
Message A is from system Hyde. Message B is from system SF/Saathi.
Both messages are in Hindi or a mix of Hindi and English.

For EACH pair score similarity on TWO dimensions:

DIMENSION 1 -- TEMPLATE MATCH (template_match_pct, integer 0-100):
  How similar is the MESSAGE STRUCTURE and PHRASING?
  Consider: template sections, sentence patterns, language tone, number and ordering of bullet points.
  Do NOT factor in actual data values (numbers, product names, store names, person names).
  100 = identical structure/phrasing, 0 = completely different structure.

DIMENSION 2 -- DATA MATCH (data_match_pct, integer 0-100):
  How similar is the ACTUAL BUSINESS DATA?
  Consider: target numbers (bills, lines), product names and discounts, store/outlet names,
            any monetary values (INR amounts).
  Do NOT factor in phrasing or template structure.
  Ignore person/DSR names -- these are expected to differ between systems.
  100 = all data values are identical, 0 = no data values match."""


def _build_batch_prompt(chunk: list[dict]) -> str:
    n = len(chunk)
    parts = [_SCORING_INSTRUCTIONS, f"\nYou have {n} pair(s) to score.\n"]
    for i, pair in enumerate(chunk, start=1):
        parts.append(f"\n--- PAIR {i} ---")
        parts.append(f"\nMessage A -- Hyde (HTML):\n{pair['html_a']}")
        parts.append(f"\nMessage B -- SF/Saathi (HTML):\n{pair['html_b']}")
        parts.append(f"\n--- END PAIR {i} ---")
    parts.append(
        f'\n\nReply ONLY with valid JSON -- no extra text, no markdown fences, no explanation.\n'
        f'Return a JSON object with a "results" array containing exactly {n} score object(s),\n'
        f'one per pair in order:\n'
        f'{{\n'
        f'  "results": [\n'
        f'    {{\n'
        f'      "template_match_pct": <integer 0-100>,\n'
        f'      "data_match_pct": <integer 0-100>,\n'
        f'      "template_note": "<one sentence in English>",\n'
        f'      "data_note": "<one sentence in English>"\n'
        f'    }}\n'
        f'  ]\n'
        f'}}'
    )
    return "\n".join(parts)


def check_model(model: str) -> None:
    try:
        available = [m.model for m in ollama.list().models]
        normalised = [m.split(":")[0] + ":" + m.split(":")[1] if ":" in m else m + ":latest"
                      for m in available]
        needle = model if ":" in model else model + ":latest"
        if needle not in normalised and model not in available:
            print(f"[error] Model '{model}' is not pulled locally.")
            print(f"        Run:  ollama pull {model}")
            sys.exit(1)
    except Exception as exc:
        print(f"[error] Cannot reach Ollama: {exc}")
        print("        Is Ollama running?  Start it with:  ollama serve")
        sys.exit(1)


def _call_llm_batch(chunk: list[dict], model: str) -> list[LLMScore | None]:
    """Send N pairs in one Ollama call; returns list of N scores (None on failure)."""
    n = len(chunk)
    prompt = _build_batch_prompt(chunk)
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format=BatchResult.model_json_schema(),
            think=False,
        )
        batch = BatchResult.model_validate_json(response.message.content)
        scores: list[LLMScore | None] = list(batch.results)
        if len(scores) != n:
            print(f"\n[warn] LLM returned {len(scores)} scores for {n} pairs -- padding/trimming.",
                  flush=True)
            while len(scores) < n:
                scores.append(None)
            scores = scores[:n]
        return scores
    except ValidationError as exc:
        print(f"\n[warn] Pydantic validation failed -- {exc.errors()[0]['msg']}", flush=True)
        return [None] * n
    except json.JSONDecodeError as exc:
        print(f"\n[warn] LLM returned non-JSON: {exc}", flush=True)
        return [None] * n
    except Exception as exc:
        print(f"\n[warn] Ollama error: {exc}", flush=True)
        return [None] * n


def score_pairs(pairs: list[dict], model: str, workers: int,
                batch_size: int = 1) -> list[ScoredRow]:
    total   = len(pairs)
    results: list[ScoredRow | None] = [None] * total
    skipped = 0
    counter = 0
    lock    = threading.Lock()

    chunks = [
        (i, pairs[i : i + batch_size])
        for i in range(0, total, batch_size)
    ]

    def _process_chunk(start: int, chunk: list[dict]) -> tuple[int, list[ScoredRow | None]]:
        scores = _call_llm_batch(chunk, model)
        scored_rows: list[ScoredRow | None] = []
        for pair, score in zip(chunk, scores):
            if score is None:
                scored_rows.append(None)
            else:
                scored_rows.append(ScoredRow(
                    msg_type           = pair["msg_type"],
                    dsr_id             = pair["dsr_id"],
                    dsr_code           = pair.get("dsr_code", ""),
                    dsr_name           = pair.get("dsr_name", ""),
                    date               = pair["date"],
                    outlet_mavic_id    = pair.get("outlet_mavic_id", ""),
                    outlet_name        = pair.get("outlet_name", ""),
                    template_match_pct = score.template_match_pct,
                    data_match_pct     = score.data_match_pct,
                    template_note      = score.template_note,
                    data_note          = score.data_note,
                ))
        return start, scored_rows

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_chunk, start, chunk): start
                   for start, chunk in chunks}
        for future in as_completed(futures):
            start, scored_chunk = future.result()
            with lock:
                for j, scored_row in enumerate(scored_chunk):
                    idx  = start + j
                    pair = pairs[idx]
                    counter += 1
                    label = pair["outlet_name"][:15] if pair.get("outlet_name") else pair["msg_type"].upper()
                    print(f"\r  [{counter:>{len(str(total))}}/{total}] "
                          f"{pair['dsr_id']}  {pair['date']}  {label} ...",
                          end="", flush=True)
                    if scored_row is None:
                        print(f"\n[warn] Pair {idx+1} skipped "
                              f"({pair['dsr_id']} / {pair['date']} / {pair['msg_type']}).",
                              flush=True)
                        skipped += 1
                    else:
                        results[idx] = scored_row

    print()
    if skipped:
        print(f"[info] {skipped} pair(s) skipped due to LLM errors.", flush=True)
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# SECTION 4: Summary & output
# ---------------------------------------------------------------------------

def build_summary(scored: list[ScoredRow]) -> RunSummary:
    by_type: dict[str, list[ScoredRow]] = {}
    for row in scored:
        by_type.setdefault(row.msg_type, []).append(row)

    type_summaries: dict[str, TypeSummary] = {}
    all_template: list[int] = []
    all_data:     list[int] = []

    for t, t_rows in sorted(by_type.items()):
        tmpl = [r.template_match_pct for r in t_rows]
        data = [r.data_match_pct     for r in t_rows]
        type_summaries[t] = TypeSummary(
            count            = len(t_rows),
            avg_template_pct = round(sum(tmpl) / len(tmpl), 1),
            avg_data_pct     = round(sum(data) / len(data), 1),
        )
        all_template.extend(tmpl)
        all_data.extend(data)

    overall = TypeSummary(
        count            = len(scored),
        avg_template_pct = round(sum(all_template) / len(all_template), 1) if all_template else 0.0,
        avg_data_pct     = round(sum(all_data)     / len(all_data),     1) if all_data     else 0.0,
    )
    return RunSummary(by_type=type_summaries, overall=overall)


def print_results(
    scored: list[ScoredRow],
    summary: RunSummary,
    outlet_diffs: list[OutletDiff],
    model: str,
) -> None:
    header = (f"{'TYPE':<10} {'DSR ID':<14} {'DSR NAME':<20} {'DATE':<12} "
              f"{'OUTLET':<20} {'TMPL %':>7} {'DATA %':>7}")
    print(f"\n{header}")
    print("-" * 90)

    for row in scored:
        outlet_label = row.outlet_name[:18] if row.outlet_name else "-"
        print(f"{row.msg_type:<10} {row.dsr_id:<14} {row.dsr_name[:18]:<20} "
              f"{row.date:<12} {outlet_label:<20} "
              f"{row.template_match_pct:>6}% {row.data_match_pct:>6}%")
        indent = " " * 12
        if row.template_note:
            print(f"{indent}template : {row.template_note}")
        if row.data_note:
            print(f"{indent}data     : {row.data_note}")

    if outlet_diffs:
        print(f"\n{'---' * 23}")
        print("  PREVISIT OUTLET MATCHING  (programmatic, by outlet_mavic_id)")
        print(f"{'---' * 23}")
        print(f"  {'DSR ID':<14} {'DATE':<12} {'COMMON':>8} {'ONLY HYDE':>10} {'ONLY SF':>8}")
        print(f"  {'--'*7} {'--'*6} {'--'*4} {'--'*5} {'--'*4}")
        for od in outlet_diffs:
            print(f"  {od.dsr_id:<14} {od.date:<12} {od.common_count:>8} "
                  f"{od.only_hyde_count:>10} {od.only_sf_count:>8}")
            if od.only_hyde_outlets:
                print(f"    only-Hyde : {', '.join(od.only_hyde_outlets)}")
            if od.only_sf_outlets:
                print(f"    only-SF   : {', '.join(od.only_sf_outlets)}")

    print("\n" + "=" * 60)
    print(f"  LLM COMPARISON SUMMARY  (model: {model})")
    print("=" * 60)
    print(f"  {'TYPE':<12} {'PAIRS':>6}  {'TEMPLATE AVG':>13}  {'DATA AVG':>9}")
    print("-" * 60)

    for key, info in summary.by_type.items():
        print(f"  {key.upper():<12} {info.count:>6}  "
              f"{info.avg_template_pct:>12.1f}%  {info.avg_data_pct:>8.1f}%")

    print("-" * 60)
    o = summary.overall
    print(f"  {'OVERALL':<12} {o.count:>6}  "
          f"{o.avg_template_pct:>12.1f}%  {o.avg_data_pct:>8.1f}%")
    print("=" * 60 + "\n")

    t = o.avg_template_pct
    d = o.avg_data_pct
    t_label = "consistent" if t >= 90 else ("drifting" if t >= 70 else "diverged")
    d_label = "accurate"   if d >= 90 else ("degraded" if d >= 70 else "unreliable")
    print(f"  Template : {t:.1f}%  -> {t_label}")
    print(f"  Data     : {d:.1f}%  -> {d_label}")
    print()


def write_output_csv(
    scored: list[ScoredRow],
    outlet_diffs: list[OutletDiff],
    path: str,
) -> None:
    fieldnames = ["msg_type", "dsr_id", "dsr_code", "dsr_name", "date",
                  "outlet_mavic_id", "outlet_name",
                  "template_match_pct", "data_match_pct", "template_note", "data_note"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in scored:
            writer.writerow(row.model_dump())
    print(f"[info] Scored results written to {path}", flush=True)

    if outlet_diffs:
        diff_path = Path(path).with_name(Path(path).stem + "_outlet_diff.csv")
        diff_fields = ["dsr_id", "dsr_name", "date", "common_count",
                       "only_hyde_count", "only_sf_count",
                       "only_hyde_outlets", "only_sf_outlets"]
        with open(diff_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=diff_fields)
            writer.writeheader()
            for od in outlet_diffs:
                d = od.model_dump()
                d["only_hyde_outlets"] = "; ".join(d["only_hyde_outlets"])
                d["only_sf_outlets"]   = "; ".join(d["only_sf_outlets"])
                writer.writerow(d)
        print(f"[info] Outlet diff written to {diff_path}", flush=True)


# ---------------------------------------------------------------------------
# SECTION 5: Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-type message comparison: load docs/ -> pair -> LLM judge -> report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python sentence_compare_poc.py
              python sentence_compare_poc.py --docs /data/docs --model qwen3:30b
              python sentence_compare_poc.py --workers 6 --out results.csv
        """),
    )
    parser.add_argument("--docs",       default=DEFAULT_DOCS_DIR,   help=f"Parent folder with Hyde/ and SF/ (default: {DEFAULT_DOCS_DIR})")
    parser.add_argument("--model",      default=DEFAULT_MODEL,      help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--workers",    default=DEFAULT_WORKERS,    type=int, help=f"Parallel Ollama calls (default: {DEFAULT_WORKERS})")
    parser.add_argument("--batch-size", default=DEFAULT_BATCH_SIZE, type=int, dest="batch_size",
                        help=f"Pairs per LLM call (default: {DEFAULT_BATCH_SIZE}; reduces prompt-token overhead)")
    parser.add_argument("--type",       default=None,               dest="msg_type",
                        choices=["sod", "eod", "previsit"],
                        help="Run only one message type (default: all three, SOD first)")
    parser.add_argument("--out",        default=None,               help="Base output CSV path; type suffix appended (e.g. results.csv -> results_sod.csv)")
    args = parser.parse_args()

    if not Path(args.docs).exists():
        print(f"[error] docs folder not found: {Path(args.docs).resolve()}")
        sys.exit(1)

    if args.workers < 1 or args.workers > 8:
        print(f"[error] --workers must be between 1 and 8 (got {args.workers})")
        sys.exit(1)

    if args.batch_size < 1 or args.batch_size > 8:
        print(f"[error] --batch-size must be between 1 and 8 (got {args.batch_size})")
        sys.exit(1)

    print(f"[info] Loading files from {Path(args.docs).resolve()} ...")
    consolidated = build_consolidated_df(docs_dir=args.docs)

    pairs, outlet_diffs = build_comparison_pairs(consolidated)

    if not pairs:
        print("[error] No comparison pairs found. Check that both Hyde/ and SF/ have matching files.")
        sys.exit(1)

    # Split by type; apply --type filter
    by_type: dict[str, list[dict]] = {"sod": [], "eod": [], "previsit": []}
    for p in pairs:
        by_type[p["msg_type"]].append(p)

    active_types = (
        [(args.msg_type, by_type[args.msg_type])]
        if args.msg_type
        else [(t, by_type[t]) for t in ("sod", "eod", "previsit") if by_type[t]]
    )

    sod_n      = len(by_type["sod"])
    eod_n      = len(by_type["eod"])
    previsit_n = len(by_type["previsit"])
    print(f"[info] {len(pairs)} pairs total ({sod_n} SOD, {eod_n} EOD, {previsit_n} previsit outlets).")
    if args.msg_type:
        print(f"[info] --type {args.msg_type}: processing only {len(by_type[args.msg_type])} pairs.")

    check_model(args.model)

    # Output path helper
    def _type_out_path(type_name: str) -> str | None:
        if not args.out:
            return None
        p = Path(args.out)
        return str(p.with_name(p.stem + f"_{type_name}" + p.suffix))

    all_scored: list[ScoredRow] = []

    for type_name, type_pairs in active_types:
        if not type_pairs:
            print(f"[info] No pairs for {type_name.upper()} -- skipping.", flush=True)
            continue

        n_chunks = (len(type_pairs) + args.batch_size - 1) // args.batch_size
        print(f"\n[info] ── {type_name.upper()} ({len(type_pairs)} pairs, "
              f"workers={args.workers}, batch_size={args.batch_size}, chunks={n_chunks}) ──",
              flush=True)

        scored = score_pairs(type_pairs, model=args.model, workers=args.workers,
                             batch_size=args.batch_size)

        if not scored:
            print(f"[warn] All {type_name.upper()} pairs failed LLM comparison.", flush=True)
            continue

        diffs_for_type = outlet_diffs if type_name == "previsit" else []
        summary = build_summary(scored)
        print_results(scored, summary, diffs_for_type, model=args.model)

        out_path = _type_out_path(type_name)
        if out_path:
            write_output_csv(scored, diffs_for_type, out_path)

        all_scored.extend(scored)

    if not all_scored:
        print("[error] All pairs failed LLM comparison.")
        sys.exit(1)

    # Overall summary when more than one type ran
    if len(active_types) > 1:
        overall_summary = build_summary(all_scored)
        print("\n" + "=" * 60)
        print(f"  OVERALL SUMMARY — ALL TYPES  (model: {args.model})")
        print("=" * 60)
        print(f"  {'TYPE':<12} {'PAIRS':>6}  {'TEMPLATE AVG':>13}  {'DATA AVG':>9}")
        print("-" * 60)
        for key, info in overall_summary.by_type.items():
            print(f"  {key.upper():<12} {info.count:>6}  "
                  f"{info.avg_template_pct:>12.1f}%  {info.avg_data_pct:>8.1f}%")
        print("-" * 60)
        o = overall_summary.overall
        print(f"  {'OVERALL':<12} {o.count:>6}  "
              f"{o.avg_template_pct:>12.1f}%  {o.avg_data_pct:>8.1f}%")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()