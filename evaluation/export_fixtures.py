"""Phase 0: Export active Rules Engine rules (PostgreSQL) and Google Sheet tabs into evaluation/fixtures/."""

import argparse
import csv
import io
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("matchops.eval.export_fixtures")

FIXTURES_DIR = PROJECT_ROOT / "evaluation" / "fixtures"
ALL_SHEETS_DIR = FIXTURES_DIR / "google_sheet_tabs"
STAGED_DICTS_DIR = FIXTURES_DIR / "staged_sheets"
RULES_SNAPSHOT_PATH = FIXTURES_DIR / "rules_snapshot.json"
MANIFEST_PATH = FIXTURES_DIR / "manifest.json"

# Leak-free dictionary & NER knowledge tabs consumed during isolated evaluation
EVAL_REFERENCE_TABS = (
    "Food_Flavors",
    "Food_GK",
    "Food_BT",
    "Food_Region",
    "Market_Brands",
    "Market_GK",
    "Market_BT",
    "Market_Category",
)


def _safe_sheet_filename(sheet_name: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in sheet_name) + ".csv"


def export_rules_snapshot(output_path: Path = RULES_SNAPSHOT_PATH) -> dict[str, Any]:
    """Exports active Rules Engine rules from PostgreSQL into a standalone JSON snapshot."""
    from backend.app.services.rules_service import compute_rules_version, load_active_rules_from_db

    rules = load_active_rules_from_db()
    version = compute_rules_version(rules)
    payload = {
        "version": version,
        "exported_at": datetime.now(UTC).isoformat(),
        "count": len(rules),
        "domains": sorted({str(r.get("domain") or "") for r in rules}),
        "rules": rules,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info(
        f"[RULES] Exported {len(rules)} active rules (version={version}, domains={payload['domains']}) -> {output_path}"
    )
    return payload


def _fetch_tab_csv_text(sheet_id: str, tab_name: str, apps_script_url: str) -> str:
    """Downloads a single Google Sheet tab via gviz CSV export with Apps Script fallback."""
    from engine.data_pipeline.ingestion import DataIngestion
    from scripts.ops.sync_sheets import fetch_sheet_data_apps_script

    try:
        return DataIngestion._download_sheet_with_retries(sheet_id, tab_name)
    except Exception as gviz_err:
        logger.warning(
            f"[SHEETS] gviz download failed for '{tab_name}' ({gviz_err}); trying Apps Script..."
        )
        rows = fetch_sheet_data_apps_script(apps_script_url, sheet_id, tab_name)
        if not rows:
            return ""
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        return buf.getvalue()


def export_google_sheets(
    sheet_id: str,
    apps_script_url: str,
    include_all_tabs: bool = True,
) -> dict[str, dict[str, Any]]:
    """Downloads Google Sheet tabs to evaluation/fixtures/google_sheet_tabs/ and staged_sheets/."""
    from scripts.ops.sync_sheets import list_available_tabs

    ALL_SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    STAGED_DICTS_DIR.mkdir(parents=True, exist_ok=True)

    tabs = list_available_tabs(apps_script_url, sheet_id) if include_all_tabs else []
    if not tabs:
        tabs = list(EVAL_REFERENCE_TABS)
    else:
        for ref_tab in EVAL_REFERENCE_TABS:
            if ref_tab not in tabs:
                tabs.append(ref_tab)

    logger.info(f"[SHEETS] Exporting {len(tabs)} tabs from Google Sheet...")
    summary: dict[str, dict[str, Any]] = {}

    for idx, tab_name in enumerate(tabs, start=1):
        fname = _safe_sheet_filename(tab_name)
        out_path = ALL_SHEETS_DIR / fname
        try:
            csv_text = _fetch_tab_csv_text(sheet_id, tab_name, apps_script_url)
            out_path.write_text(csv_text, encoding="utf-8")

            row_count = max(0, len(csv_text.splitlines()) - 1) if csv_text.strip() else 0
            try:
                has_hdr = tab_name not in {
                    "Food_GK",
                    "Food_BT",
                    "Food_Region",
                    "Market_GK",
                    "Market_BT",
                    "Market_Category",
                }
                df_tmp = pd.read_csv(io.StringIO(csv_text), header=0 if has_hdr else None)
                row_count = len(df_tmp)
            except Exception:
                pass

            if tab_name in EVAL_REFERENCE_TABS:
                staged_path = STAGED_DICTS_DIR / fname
                staged_path.write_text(csv_text, encoding="utf-8")

            summary[tab_name] = {
                "file": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "staged_ref": tab_name in EVAL_REFERENCE_TABS,
                "rows": row_count,
                "bytes": len(csv_text.encode("utf-8")),
            }
            logger.info(f"[SHEETS] ({idx}/{len(tabs)}) Saved '{tab_name}' ({row_count} rows)")
        except Exception as err:
            logger.error(f"[SHEETS] Failed to download '{tab_name}': {err}")
            summary[tab_name] = {"error": str(err)}

    return summary


def sync_market_gk_from_eval_csvs() -> list[str]:
    """Appends any Market GenericKeywords from train/test CSVs missing in staged Market_GK.csv."""
    import polars as pl

    gk_path = STAGED_DICTS_DIR / "Market_GK.csv"
    if not gk_path.exists():
        raise FileNotFoundError(f"Missing staged Market_GK.csv at {gk_path}")

    df_pl = pl.read_csv(gk_path.read_bytes(), has_header=False, ignore_errors=True)
    existing = {str(v).strip() for v in df_pl.to_pandas().iloc[:, 0].dropna() if str(v).strip()}

    missing: set[str] = set()
    for split in ("train", "test"):
        df = pd.read_csv(PROJECT_ROOT / "evaluation" / f"{split}_market.csv")
        for cell in df["GenericKeywords"].dropna():
            for raw_tag in str(cell).split(","):
                tag = raw_tag.strip()
                if tag and tag not in existing:
                    missing.add(tag)

    added = sorted(missing)
    if added:
        raw_text = gk_path.read_text(encoding="utf-8")
        first_line = raw_text.splitlines()[0] if raw_text.splitlines() else ""
        empty_cols = max(0, first_line.count('""') - (1 if first_line.startswith('""') else 0))
        suffix = "," + ",".join(['""'] * empty_cols) if empty_cols else ""
        new_lines = ['"' + kw.replace('"', '""') + '"' + suffix for kw in added]
        if raw_text and not raw_text.endswith("\n"):
            raw_text += "\n"
        gk_path.write_text(raw_text + "\n".join(new_lines) + "\n", encoding="utf-8")
        logger.info(f"[SHEETS] Appended {len(added)} missing Market GK tags -> {gk_path}")
    else:
        logger.info("[SHEETS] Market_GK.csv already contains all train/test Market GKs.")
    return added


def validate_alignment() -> dict[str, Any]:
    """Checks train/test evaluation CSVs against staged reference dictionaries."""
    report: dict[str, Any] = {}
    for domain in ("food", "market"):
        train_df = pd.read_csv(PROJECT_ROOT / "evaluation" / f"train_{domain}.csv")
        test_df = pd.read_csv(PROJECT_ROOT / "evaluation" / f"test_{domain}.csv")
        third_col = "Region" if domain == "food" else "Category"
        prefix = "Food" if domain == "food" else "Market"

        dict_sets: dict[str, set[str]] = {}
        for logical_key, tab in (
            ("bt", f"{prefix}_BT"),
            ("gk", f"{prefix}_GK"),
            ("third", f"{prefix}_{third_col}"),
        ):
            p = STAGED_DICTS_DIR / _safe_sheet_filename(tab)
            if p.exists():
                vals = pd.read_csv(p, header=None, dtype=str).iloc[:, 0].dropna().str.strip()
                dict_sets[logical_key] = {v for v in vals if v}
            else:
                dict_sets[logical_key] = set()

        train_bt = set(train_df["BasicType"].dropna().astype(str).str.strip())
        test_bt = set(test_df["BasicType"].dropna().astype(str).str.strip())
        train_third = set(train_df[third_col].dropna().astype(str).str.strip())
        test_third = set(test_df[third_col].dropna().astype(str).str.strip())
        train_gk = {
            k.strip()
            for s in train_df["GenericKeywords"].dropna()
            for k in str(s).split(",")
            if k.strip()
        }
        test_gk = {
            k.strip()
            for s in test_df["GenericKeywords"].dropna()
            for k in str(s).split(",")
            if k.strip()
        }

        report[domain] = {
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_columns": list(train_df.columns),
            "test_columns": list(test_df.columns),
            "bt": {
                "dict_count": len(dict_sets["bt"]),
                "train_count": len(train_bt),
                "test_count": len(test_bt),
                "test_unseen_in_train": sorted(test_bt - train_bt),
                "train_not_in_dict": sorted(train_bt - dict_sets["bt"]),
                "test_not_in_dict": sorted(test_bt - dict_sets["bt"]),
            },
            "third_tag": {
                "column": third_col,
                "dict_count": len(dict_sets["third"]),
                "train_count": len(train_third),
                "test_count": len(test_third),
                "test_unseen_in_train": sorted(test_third - train_third),
                "train_not_in_dict": sorted(train_third - dict_sets["third"]),
                "test_not_in_dict": sorted(test_third - dict_sets["third"]),
            },
            "gk": {
                "dict_count": len(dict_sets["gk"]),
                "train_count": len(train_gk),
                "test_count": len(test_gk),
                "test_unseen_in_train_count": len(test_gk - train_gk),
                "train_not_in_dict_count": len(train_gk - dict_sets["gk"]),
                "test_not_in_dict_count": len(test_gk - dict_sets["gk"]),
            },
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Postgres rules & Google Sheet tabs for offline/Colab evaluation."
    )
    parser.add_argument("--skip-rules", action="store_true", help="Skip PostgreSQL rules export.")
    parser.add_argument("--skip-sheets", action="store_true", help="Skip Google Sheets download.")
    parser.add_argument(
        "--dicts-only",
        action="store_true",
        help="Download only the 8 evaluation reference/dictionary tabs instead of every tab.",
    )
    parser.add_argument(
        "--fix-market-gk",
        action="store_true",
        help="Append any Market GKs from evaluation CSVs missing in staged_sheets/Market_GK.csv.",
    )
    args = parser.parse_args()

    from scripts.ops.sync_sheets import DEFAULT_APPS_SCRIPT_URL

    sheet_id = os.getenv("GOOGLE_SHEET_ID", "").strip()
    apps_script_url = os.getenv("APPS_SCRIPT_URL", "").strip() or DEFAULT_APPS_SCRIPT_URL

    rules_meta: dict[str, Any] = {}
    if not args.skip_rules:
        rules_meta = export_rules_snapshot()

    sheets_meta: dict[str, Any] = {}
    if not args.skip_sheets:
        if not sheet_id:
            raise RuntimeError("GOOGLE_SHEET_ID must be set in .env to download Google Sheet tabs.")
        sheets_meta = export_google_sheets(
            sheet_id=sheet_id,
            apps_script_url=apps_script_url,
            include_all_tabs=not args.dicts_only,
        )

    added_market_gks: list[str] = []
    if args.fix_market_gk:
        added_market_gks = sync_market_gk_from_eval_csvs()

    alignment = validate_alignment()
    manifest = {
        "exported_at": datetime.now(UTC).isoformat(),
        "rules": {
            "version": rules_meta.get("version"),
            "count": rules_meta.get("count"),
            "domains": rules_meta.get("domains"),
        },
        "sheets": sheets_meta,
        "added_market_gks": added_market_gks,
        "alignment": alignment,
    }
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"[DONE] Saved manifest -> {MANIFEST_PATH}")
    print(json.dumps(alignment, indent=2))


if __name__ == "__main__":
    main()
