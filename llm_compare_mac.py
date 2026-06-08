"""
llm_compare_mac.py — Mac-optimised LLM comparator using Qwen3:14b via Ollama.

Identical input/output contract to llm_compare.py, with three Mac-specific
improvements:

  1. Bigger model (qwen3:14b default) — full Metal GPU on Apple Silicon;
     reliable Hindi + bilingual instruction-following at 14b.
  2. Parallel execution — ThreadPoolExecutor sends N pairs concurrently;
     Ollama handles multiple Metal inference sessions efficiently.
  3. Raw HTML sent to LLM — <li> structure helps the model separate template
     bullets from data values without stripping context.

Usage
-----
    python llm_compare_mac.py input.csv
    python llm_compare_mac.py input.csv --model qwen3:30b
    python llm_compare_mac.py input.csv --workers 6
    python llm_compare_mac.py input.csv --out results.csv

Input CSV schema:
    type, dsr_id, outlet_id, date, html_a, html_b

Supported types: sod, previsit, eod
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
from pydantic import BaseModel, Field, ValidationError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL   = "qwen3:14b"
DEFAULT_WORKERS = 4
SUPPORTED_TYPES = {"sod", "previsit", "eod"}
REQUIRED_COLUMNS = {"type", "dsr_id", "date", "html_a", "html_b"}

# ---------------------------------------------------------------------------
# Pydantic models  (same shape as llm_compare.py — downstream-compatible)
# ---------------------------------------------------------------------------

class LLMScore(BaseModel):
    template_match_pct: int = Field(..., ge=0, le=100)
    data_match_pct:     int = Field(..., ge=0, le=100)
    template_note:      str = Field(default="")
    data_note:          str = Field(default="")


class ScoredRow(BaseModel):
    type:               str
    dsr_id:             str
    outlet_id:          str
    date:               str
    template_match_pct: int
    data_match_pct:     int
    template_note:      str
    data_note:          str


class TypeSummary(BaseModel):
    count:            int
    avg_template_pct: float
    avg_data_pct:     float


class RunSummary(BaseModel):
    by_type: dict[str, TypeSummary]
    overall: TypeSummary


# ---------------------------------------------------------------------------
# LLM prompt  — raw HTML passed so <li> structure gives the model context
# ---------------------------------------------------------------------------

_PROMPT = """\
You are comparing two sales briefing messages sent to the same field sales executive.
Message A is from system Saathi. Message B is from system SFDC/Hyde.
Both messages are in Hindi or a mix of Hindi and English.

Score the similarity on TWO separate dimensions and provide a one-sentence note for each.

DIMENSION 1 — TEMPLATE MATCH (template_match_pct, integer 0-100):
  How similar is the MESSAGE STRUCTURE and PHRASING?
  Consider: template sections, sentence patterns, language tone, number and ordering of bullet points.
  Do NOT factor in the actual data values (numbers, product names, store names, person name).
  100 = identical structure/phrasing, 0 = completely different structure.

DIMENSION 2 — DATA MATCH (data_match_pct, integer 0-100):
  How similar is the ACTUAL BUSINESS DATA?
  Consider: target numbers (bills, lines), product names and discounts, store/outlet names,
            any monetary values (INR amounts).
  Do NOT factor in phrasing or template structure.
  Ignore person/DSR names — these are expected to differ between systems.
  100 = all data values are identical, 0 = no data values match.

--- Message A (HTML) ---
{html_a}

--- Message B (HTML) ---
{html_b}
--- End ---

Reply ONLY with valid JSON — no extra text, no markdown fences, no explanation:
{{
  "template_match_pct": <integer 0-100>,
  "data_match_pct": <integer 0-100>,
  "template_note": "<one sentence in English>",
  "data_note": "<one sentence in English>"
}}"""


# ---------------------------------------------------------------------------
# Startup model check
# ---------------------------------------------------------------------------

def check_model(model: str) -> None:
    """Exit with a helpful message if the model is not pulled locally."""
    try:
        available = [m.model for m in ollama.list().models]
        # Ollama appends :latest when no tag is given; normalise for comparison
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


# ---------------------------------------------------------------------------
# LLM call — returns None on any failure
# ---------------------------------------------------------------------------

def _call_llm(html_a: str, html_b: str, model: str) -> LLMScore | None:
    prompt = _PROMPT.format(html_a=html_a, html_b=html_b)
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format=LLMScore.model_json_schema(),
            think=False,
        )
        return LLMScore.model_validate_json(response.message.content)
    except ValidationError as exc:
        print(f"\n[warn] Pydantic validation failed — {exc.errors()[0]['msg']}", flush=True)
        return None
    except json.JSONDecodeError as exc:
        print(f"\n[warn] LLM returned non-JSON: {exc}", flush=True)
        return None
    except Exception as exc:
        print(f"\n[warn] Ollama error: {exc}", flush=True)
        return None


# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------

def read_input_csv(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {missing}")
        for i, row in enumerate(reader, start=2):
            msg_type = row["type"].strip().lower()
            if msg_type not in SUPPORTED_TYPES:
                print(f"[warn] Row {i}: unknown type '{row['type']}' — skipped.", flush=True)
                continue
            rows.append({
                "type":      msg_type,
                "dsr_id":    row["dsr_id"].strip(),
                "outlet_id": row.get("outlet_id", "").strip(),
                "date":      row["date"].strip(),
                "html_a":    row["html_a"],
                "html_b":    row["html_b"],
            })
    return rows


# ---------------------------------------------------------------------------
# Parallel scoring pipeline
# ---------------------------------------------------------------------------

def score_rows(rows: list[dict], model: str, workers: int) -> list[ScoredRow]:
    total    = len(rows)
    results  = [None] * total       # pre-allocated — preserve order
    skipped  = 0
    counter  = 0
    lock     = threading.Lock()

    def _process(index: int, row: dict) -> tuple[int, ScoredRow | None]:
        score = _call_llm(row["html_a"], row["html_b"], model)
        if score is None:
            return index, None
        return index, ScoredRow(
            type               = row["type"],
            dsr_id             = row["dsr_id"],
            outlet_id          = row["outlet_id"],
            date               = row["date"],
            template_match_pct = score.template_match_pct,
            data_match_pct     = score.data_match_pct,
            template_note      = score.template_note,
            data_note          = score.data_note,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process, i, row): i for i, row in enumerate(rows)}
        for future in as_completed(futures):
            idx, scored_row = future.result()
            row = rows[idx]
            with lock:
                nonlocal_counter = counter  # read before increment for display
                counter += 1
                print(f"\r  [{counter:>{len(str(total))}}/{total}] "
                      f"{row['dsr_id']}  {row['date']} ...", end="", flush=True)
            if scored_row is None:
                print(f"\n[warn] Pair {idx + 1} ({row['dsr_id']} / {row['date']}) skipped.",
                      flush=True)
                skipped += 1
            else:
                results[idx] = scored_row

    print()
    if skipped:
        print(f"[info] {skipped} pair(s) skipped due to LLM errors.", flush=True)

    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary(scored: list[ScoredRow]) -> RunSummary:
    by_type: dict[str, list[ScoredRow]] = {}
    for row in scored:
        by_type.setdefault(row.type, []).append(row)

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


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_results(scored: list[ScoredRow], summary: RunSummary, model: str) -> None:
    header = f"{'TYPE':<10} {'DSR':<12} {'OUTLET':<14} {'DATE':<12} {'TMPL %':>7} {'DATA %':>7}"
    print(f"\n{header}")
    print("-" * 65)

    for row in scored:
        print(
            f"{row.type:<10} {row.dsr_id:<12} {row.outlet_id or '-':<14} "
            f"{row.date:<12} {row.template_match_pct:>6}% {row.data_match_pct:>6}%"
        )
        indent = " " * 12
        if row.template_note:
            print(f"{indent}template : {row.template_note}")
        if row.data_note:
            print(f"{indent}data     : {row.data_note}")

    print("\n" + "=" * 58)
    print(f"  LLM COMPARISON SUMMARY  (model: {model})")
    print("=" * 58)
    print(f"  {'TYPE':<12} {'PAIRS':>6}  {'TEMPLATE AVG':>13}  {'DATA AVG':>9}")
    print("-" * 58)

    for key, info in summary.by_type.items():
        print(f"  {key.upper():<12} {info.count:>6}  "
              f"{info.avg_template_pct:>12.1f}%  {info.avg_data_pct:>8.1f}%")

    print("-" * 58)
    o = summary.overall
    print(f"  {'OVERALL':<12} {o.count:>6}  "
          f"{o.avg_template_pct:>12.1f}%  {o.avg_data_pct:>8.1f}%")
    print("=" * 58 + "\n")

    t = o.avg_template_pct
    d = o.avg_data_pct
    t_label = "consistent" if t >= 90 else ("drifting"  if t >= 70 else "diverged")
    d_label = "accurate"   if d >= 90 else ("degraded"  if d >= 70 else "unreliable")
    print(f"  Template : {t:.1f}%  → {t_label}")
    print(f"  Data     : {d:.1f}%  → {d_label}")
    print()


# ---------------------------------------------------------------------------
# Optional CSV output
# ---------------------------------------------------------------------------

def write_output_csv(scored: list[ScoredRow], path: str) -> None:
    fieldnames = ["type", "dsr_id", "outlet_id", "date",
                  "template_match_pct", "data_match_pct",
                  "template_note", "data_note"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in scored:
            writer.writerow(row.model_dump())
    print(f"[info] Results written to {path}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mac-optimised LLM message comparator (Ollama + Metal GPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python llm_compare_mac.py input.csv
              python llm_compare_mac.py input.csv --model qwen3:30b
              python llm_compare_mac.py input.csv --workers 6 --out results.csv
        """),
    )
    parser.add_argument("input",   help="Input CSV file (type, dsr_id, outlet_id, date, html_a, html_b)")
    parser.add_argument("--model",   default=DEFAULT_MODEL,   help=f"Ollama model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--workers", default=DEFAULT_WORKERS, type=int, help=f"Parallel Ollama calls (default: {DEFAULT_WORKERS}, max recommended: 8)")
    parser.add_argument("--out",     default=None,            help="Optional output CSV path for scored results")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"[error] File not found: {args.input}")
        sys.exit(1)

    if args.workers < 1 or args.workers > 8:
        print(f"[error] --workers must be between 1 and 8 (got {args.workers})")
        sys.exit(1)

    # Verify model is available before starting
    check_model(args.model)

    print(f"[info] Reading {args.input} ...")
    rows = read_input_csv(args.input)
    if not rows:
        print("[error] No valid rows found in input CSV.")
        sys.exit(1)

    print(f"[info] {len(rows)} pairs loaded. Sending to {args.model} "
          f"via Ollama (workers={args.workers}) ...")
    scored = score_rows(rows, model=args.model, workers=args.workers)

    if not scored:
        print("[error] All rows failed LLM comparison.")
        sys.exit(1)

    summary = build_summary(scored)
    print_results(scored, summary, model=args.model)

    if args.out:
        write_output_csv(scored, args.out)


if __name__ == "__main__":
    main()
