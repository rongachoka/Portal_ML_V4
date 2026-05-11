"""
morning_runner.py
=================
Daily morning automation for Portal Pharmacy ML pipeline.

Steps:
    1  Pre-flight check — scan SharePoint downloads for new files today,
                          detect date format in each branch's latest sales CSV
    2  Archive old CSVs — keep only the most recent sales_reports CSV per branch,
                          move all older ones to archive/{branch_code}/
    3  Run pipeline     — execute run_pipeline_copy.py via portal_venv, capture output
    4  Write summary    — write morning_summary.txt with per-branch table + pipeline result

Run:
    python morning_runner.py

Register with Windows Task Scheduler (run once as Administrator):
    python morning_runner.py --setup-scheduler
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
from datetime import datetime, date
from pathlib import Path
from io import StringIO

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  (derived from __file__ — no hardcoded credentials)
# ─────────────────────────────────────────────────────────────────────────────

ROOT_DIR        = Path(__file__).resolve().parent
SP_DOWNLOADS    = ROOT_DIR / "data" / "01_raw" / "sharepoint_downloads"
ARCHIVE_ROOT    = SP_DOWNLOADS / "archive"
PROCESSED_DIR   = ROOT_DIR / "data" / "03_processed"
SOCIAL_SALES    = PROCESSED_DIR / "sales_attribution" / "social_sales_direct.csv"
SUMMARY_FILE    = ROOT_DIR / "morning_summary.txt"
PIPELINE_SCRIPT = ROOT_DIR / "run_pipeline_copy.py"
VENV_PYTHON     = Path(r"D:\Documents\Portal ML Analys\Portal_ML\portal_venv\Scripts\python.exe")

# ─────────────────────────────────────────────────────────────────────────────
# BRANCH CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BRANCHES = {
    "ABC":          "abc",
    "Centurion 2R": "c2r",
    "Galleria":     "galleria",
    "Milele":       "milele",
    "Portal 2R":    "p2r",
    "Portal CBD":   "cbd",
}

# Candidate date column names in the sales CSV (checked in order)
DATE_COLUMNS = ["Date Sold", "date_sold", "Date_Sold", "Sale_Date", "sale_date",
                "Transaction Date", "Date"]

# (strptime format, human label) — probed in order
DATE_FORMATS = [
    ("%d/%m/%Y %H:%M:%S",    "DD/MM/YYYY HH:MM:SS"),
    ("%Y-%m-%d %H:%M:%S",    "YYYY-MM-DD HH:MM:SS"),
    ("%m/%d/%Y %I:%M:%S %p", "M/D/YYYY HH:MM:SS AM/PM"),
    ("%d/%m/%Y",             "DD/MM/YYYY"),
    ("%Y-%m-%d",             "YYYY-MM-DD"),
    ("%d-%b-%Y %I:%M:%S %p", "DD-MMM-YYYY HH:MM:SS AM/PM"),
    ("%d-%b-%y %I:%M:%S %p", "DD-MMM-YY HH:MM:SS AM/PM"),
    ("%d-%b-%Y",             "DD-MMM-YYYY"),
    ("%d-%b-%y",             "DD-MMM-YY"),
    ("%m/%d/%Y",             "MM/DD/YYYY"),
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _detect_date_format(date_str: str) -> str:
    """Return a human-readable label for the date format in date_str."""
    cleaned = str(date_str).strip()
    for fmt, label in DATE_FORMATS:
        try:
            datetime.strptime(cleaned, fmt)
            return label
        except ValueError:
            continue
    return f"Unknown  (sample: {cleaned[:25]})"


def _read_date_format_from_csv(csv_path: Path) -> tuple[str, str]:
    """
    Open csv_path, find the first recognised date column, and return
    (column_name, format_label).  Returns ("—", "no date column found")
    if nothing matches.
    """
    try:
        with open(csv_path, newline="", encoding="utf-8-sig", errors="replace") as fh:
            reader = csv.DictReader(fh)
            headers = reader.fieldnames or []
            col = next((c for c in DATE_COLUMNS if c in headers), None)
            if col is None:
                return "—", "no date column found"
            for row in reader:
                val = row.get(col, "").strip()
                if val:
                    return col, _detect_date_format(val)
        return col, "no rows with data"
    except Exception as exc:
        return "—", f"read error: {exc}"


def _was_modified_today(path: Path) -> bool:
    """True if path's mtime is today."""
    try:
        return date.fromtimestamp(path.stat().st_mtime) == date.today()
    except OSError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — PRE-FLIGHT CHECK
# ─────────────────────────────────────────────────────────────────────────────

def step1_preflight() -> dict:
    """
    For each branch: detect whether new files arrived today in sales_reports
    and cashier_reports, then read the date format from the latest sales CSV.

    Returns a dict keyed by branch name with preflight findings.
    """
    _log("STEP 1 — Pre-flight check")
    results = {}

    for branch, code in BRANCHES.items():
        branch_dir = SP_DOWNLOADS / branch
        sales_dir   = branch_dir / "sales_reports"
        cashier_dir = branch_dir / "cashier_reports"

        # ── Sales reports ─────────────────────────────────────────────────────
        sales_csvs = sorted(
            sales_dir.glob("*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ) if sales_dir.exists() else []

        latest_sales    = sales_csvs[0] if sales_csvs else None
        new_sales_today = _was_modified_today(latest_sales) if latest_sales else False
        total_sales     = len(sales_csvs)

        if latest_sales:
            date_col, date_fmt = _read_date_format_from_csv(latest_sales)
        else:
            date_col, date_fmt = "—", "no sales CSV found"

        # ── Cashier reports ───────────────────────────────────────────────────
        cashier_files = sorted(
            [p for p in cashier_dir.iterdir() if p.is_file()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ) if cashier_dir.exists() else []

        latest_cashier    = cashier_files[0] if cashier_files else None
        new_cashier_today = _was_modified_today(latest_cashier) if latest_cashier else False

        results[branch] = {
            "code":             code,
            "new_sales_today":  new_sales_today,
            "new_cashier_today":new_cashier_today,
            "total_sales_csvs": total_sales,
            "latest_sales":     latest_sales.name if latest_sales else "—",
            "date_col":         date_col,
            "date_fmt":         date_fmt,
        }

        status_s = "YES" if new_sales_today  else "no"
        status_c = "YES" if new_cashier_today else "no"
        _log(
            f"  {branch:<14} sales_new={status_s}  cashier_new={status_c}"
            f"  date_fmt={date_fmt}  total_sales_CSVs={total_sales}"
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — ARCHIVE OLD SALES REPORTS
# ─────────────────────────────────────────────────────────────────────────────

def step2_archive(preflight: dict) -> dict:
    """
    For each branch's sales_reports folder, keep only the single most recent
    CSV file and move all others to ARCHIVE_ROOT/{branch_code}/.

    Returns a dict keyed by branch name with archive results.
    """
    _log("STEP 2 — Archive old sales reports")
    results = {}

    for branch, code in BRANCHES.items():
        sales_dir   = SP_DOWNLOADS / branch / "sales_reports"
        archive_dir = ARCHIVE_ROOT / code
        archive_dir.mkdir(parents=True, exist_ok=True)

        sales_csvs = sorted(
            sales_dir.glob("*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ) if sales_dir.exists() else []

        if len(sales_csvs) <= 1:
            results[branch] = {"archived": 0, "kept": sales_csvs[0].name if sales_csvs else "—"}
            _log(f"  {branch:<14} nothing to archive (only {len(sales_csvs)} CSV)")
            continue

        kept   = sales_csvs[0]
        to_move = sales_csvs[1:]
        moved  = 0
        errors = []

        for f in to_move:
            dest = archive_dir / f.name
            # Avoid overwriting an existing archive file with the same name
            if dest.exists():
                stem = f.stem
                ext  = f.suffix
                dest = archive_dir / f"{stem}_dup{ext}"
            try:
                shutil.move(str(f), str(dest))
                moved += 1
            except Exception as exc:
                errors.append(f"{f.name}: {exc}")

        results[branch] = {
            "archived": moved,
            "kept":     kept.name,
            "errors":   errors,
        }
        _log(f"  {branch:<14} kept={kept.name}  archived={moved}  errors={len(errors)}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — RUN MORNING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def step3_run_pipeline() -> dict:
    """
    Execute run_pipeline_copy.py via the portal_venv Python interpreter.
    Captures all stdout + stderr. Returns a dict with success flag, output,
    elapsed seconds, and error message if any.
    """
    _log("STEP 3 — Running morning pipeline (run_pipeline_copy.py)…")

    python = VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)
    if not VENV_PYTHON.exists():
        _log(f"  WARNING: portal_venv not found at {VENV_PYTHON} — using {python}")

    t_start = datetime.now()
    try:
        proc = subprocess.run(
            [str(python), str(PIPELINE_SCRIPT)],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            timeout=7200,           # 2-hour hard ceiling
        )
        elapsed = (datetime.now() - t_start).total_seconds()
        success = proc.returncode == 0
        output  = (proc.stdout or "") + (proc.stderr or "")

        if success:
            _log(f"  Pipeline finished successfully in {elapsed:.0f}s")
        else:
            tail = output.strip().splitlines()[-10:]
            _log(f"  Pipeline FAILED (exit code {proc.returncode})")
            for line in tail:
                _log(f"    {line}")

        return {
            "success":   success,
            "returncode":proc.returncode,
            "elapsed_s": elapsed,
            "output":    output,
            "error":     None if success else f"exit code {proc.returncode}",
        }

    except subprocess.TimeoutExpired:
        elapsed = (datetime.now() - t_start).total_seconds()
        _log(f"  Pipeline TIMED OUT after {elapsed:.0f}s")
        return {"success": False, "returncode": -1, "elapsed_s": elapsed,
                "output": "", "error": "Timed out after 2 hours"}
    except Exception as exc:
        elapsed = (datetime.now() - t_start).total_seconds()
        _log(f"  Pipeline ERROR: {exc}")
        return {"success": False, "returncode": -1, "elapsed_s": elapsed,
                "output": "", "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — WRITE MORNING SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def _read_social_sales_stats() -> dict:
    """Read row count and revenue total from social_sales_direct.csv."""
    if not SOCIAL_SALES.exists():
        return {"rows": 0, "revenue": 0.0, "error": "file not found"}
    try:
        rows = 0
        revenue = 0.0
        with open(SOCIAL_SALES, newline="", encoding="utf-8-sig", errors="replace") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows += 1
                for col in ("Total (Tax Ex)", "total_tax_ex", "Revenue", "Amount"):
                    raw = row.get(col, "").strip()
                    if raw:
                        try:
                            revenue += float(raw.replace(",", ""))
                        except ValueError:
                            pass
                        break
        return {"rows": rows, "revenue": revenue, "error": None}
    except Exception as exc:
        return {"rows": 0, "revenue": 0.0, "error": str(exc)}


def step4_write_summary(
    run_ts: str,
    preflight: dict,
    archive_results: dict,
    pipeline: dict,
) -> None:
    """Write morning_summary.txt to ROOT_DIR."""
    _log("STEP 4 — Writing morning summary")

    social = _read_social_sales_stats() if pipeline["success"] else None

    lines = []
    lines.append("=" * 72)
    lines.append("PORTAL PHARMACY — MORNING RUN SUMMARY")
    lines.append(f"Run time : {run_ts}")
    lines.append("=" * 72)

    # ── Per-branch table ──────────────────────────────────────────────────────
    lines.append("")
    lines.append("PRE-FLIGHT CHECK")
    lines.append("-" * 72)
    header = f"{'Branch':<16} {'New Sales':>10} {'New Cashier':>12} {'Old Archived':>13}  Date Format"
    lines.append(header)
    lines.append("-" * 72)

    for branch in BRANCHES:
        pf  = preflight.get(branch, {})
        arc = archive_results.get(branch, {})
        new_s  = "YES" if pf.get("new_sales_today")   else "no"
        new_c  = "YES" if pf.get("new_cashier_today")  else "no"
        n_arc  = arc.get("archived", 0)
        d_fmt  = pf.get("date_fmt", "—")
        lines.append(
            f"{branch:<16} {new_s:>10} {new_c:>12} {n_arc:>13}  {d_fmt}"
        )

    lines.append("-" * 72)
    lines.append("")

    # ── Archive detail (errors only) ──────────────────────────────────────────
    any_errors = any(arc.get("errors") for arc in archive_results.values())
    if any_errors:
        lines.append("ARCHIVE ERRORS")
        lines.append("-" * 72)
        for branch, arc in archive_results.items():
            for err in arc.get("errors", []):
                lines.append(f"  {branch}: {err}")
        lines.append("")

    # ── Pipeline result ───────────────────────────────────────────────────────
    lines.append("PIPELINE RESULT")
    lines.append("-" * 72)
    elapsed_min = pipeline["elapsed_s"] / 60
    if pipeline["success"]:
        lines.append(f"  Status  : SUCCESS")
        lines.append(f"  Duration: {elapsed_min:.1f} min")
    else:
        lines.append(f"  Status  : FAILED")
        lines.append(f"  Error   : {pipeline.get('error', 'unknown')}")
        lines.append(f"  Duration: {elapsed_min:.1f} min")
        lines.append("")
        lines.append("  Last 20 lines of pipeline output:")
        tail = (pipeline.get("output") or "").strip().splitlines()[-20:]
        for line in tail:
            lines.append(f"    {line}")
    lines.append("")

    # ── Social sales stats ────────────────────────────────────────────────────
    if social is not None:
        lines.append("SOCIAL SALES OUTPUT  (social_sales_direct.csv)")
        lines.append("-" * 72)
        if social["error"]:
            lines.append(f"  Error reading file: {social['error']}")
        else:
            lines.append(f"  Rows    : {social['rows']:,}")
            lines.append(f"  Revenue : KES {social['revenue']:>14,.2f}")
        lines.append("")

    lines.append("=" * 72)

    summary_text = "\n".join(lines) + "\n"
    SUMMARY_FILE.write_text(summary_text, encoding="utf-8")
    _log(f"  Summary written → {SUMMARY_FILE.name}")
    print()
    print(summary_text)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — TASK SCHEDULER REGISTRATION
# ─────────────────────────────────────────────────────────────────────────────

def setup_scheduler() -> None:
    """
    Register a Windows Task Scheduler entry that runs this script daily at 4:00 AM.
    Must be run as Administrator.
    """
    python  = str(VENV_PYTHON)
    script  = str(Path(__file__).resolve())
    task_name = r"Portal Pharmacy\Morning Runner"

    cmd = [
        "schtasks", "/Create",
        "/TN",  task_name,
        "/TR",  f'"{python}" "{script}"',
        "/SC",  "DAILY",
        "/ST",  "04:00",
        "/RL",  "HIGHEST",
        "/F",
    ]

    print(f"Registering Task Scheduler entry: {task_name}")
    print(f"  Python  : {python}")
    print(f"  Script  : {script}")
    print(f"  Schedule: Daily at 04:00")
    print()

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("SUCCESS: Task registered.")
        print(result.stdout.strip())
    else:
        print(f"FAILED (exit code {result.returncode})")
        print(result.stderr.strip())
        print()
        print("Make sure you are running this script as Administrator.")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Portal Pharmacy morning runner")
    parser.add_argument(
        "--setup-scheduler",
        action="store_true",
        help="Register this script in Windows Task Scheduler (run as Administrator)",
    )
    args = parser.parse_args()

    if args.setup_scheduler:
        setup_scheduler()
        return

    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _log(f"PORTAL PHARMACY MORNING RUNNER — {run_ts}")
    _log(f"Root : {ROOT_DIR}")
    print()

    # Step 1
    preflight = step1_preflight()
    print()

    # Step 2
    archive_results = step2_archive(preflight)
    print()

    # Step 3
    pipeline = step3_run_pipeline()
    print()

    # Step 4
    step4_write_summary(run_ts, preflight, archive_results, pipeline)

    sys.exit(0 if pipeline["success"] else 1)


if __name__ == "__main__":
    main()
