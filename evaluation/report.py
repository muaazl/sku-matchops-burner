"""Aggregates completed evaluation runs into ranked CSV tables and a Markdown summary report."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("matchops.eval.report")


def _df_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```\n" + df.to_string(index=False) + "\n```"


def _collect_run_summaries(results_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    if not results_dir.exists():
        return summaries
    for summary_path in sorted(results_dir.glob("*/summary.json")):
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            data["_run_dir"] = str(summary_path.parent)
            summaries.append(data)
        except Exception as err:
            logger.warning(f"[REPORT] Skipping unreadable summary {summary_path}: {err}")
    return summaries


def generate_reports(work_dir: Path | str) -> dict[str, Path]:
    """Builds master CSV leaderboards and Markdown summary from completed summary.json files."""
    wdir = Path(work_dir)
    results_dir = wdir / "results"
    reports_dir = wdir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    summaries = _collect_run_summaries(results_dir)
    if not summaries:
        logger.warning(f"[REPORT] No completed run summaries found under {results_dir}.")
        return {}

    e2e_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []

    seen_stage_b: set[tuple[str, str, str]] = set()

    for s in summaries:
        meta = s.get("meta", {})
        domain = meta.get("domain", "")
        emb = meta.get("embedding_model", "")
        rerank = meta.get("reranker_model", "")
        prec = meta.get("precision", "")
        timings = s.get("timings_s", {})
        n_test = meta.get("test_rows", 0)

        stage_b_key = (domain, emb, prec)
        if stage_b_key not in seen_stage_b:
            seen_stage_b.add(stage_b_key)
            emb_only = s.get("embedding_standalone", {})
            for ret_mode, ret_m in (emb_only.get("catalog_retrieval") or {}).items():
                retrieval_rows.append(
                    {
                        "domain": domain,
                        "embedding_model": emb,
                        "precision": prec,
                        "space": "catalog",
                        "retrieval_mode": ret_mode,
                        "hit@1_pct": ret_m.get("hit@1_pct"),
                        "hit@3_pct": ret_m.get("hit@3_pct"),
                        "hit@5_pct": ret_m.get("hit@5_pct"),
                        "hit@10_pct": ret_m.get("hit@10_pct"),
                        "precision@1_pct": ret_m.get("precision@1_pct"),
                        "precision@3_pct": ret_m.get("precision@3_pct"),
                        "recall@3_pct": ret_m.get("recall@3_pct"),
                        "capped_recall@3_pct": ret_m.get("capped_recall@3_pct"),
                        "f1@3_pct": ret_m.get("f1@3_pct"),
                        "mrr": ret_m.get("mrr"),
                        "map@10": ret_m.get("map@10"),
                        "ndcg@3": ret_m.get("ndcg@3"),
                        "ndcg@10": ret_m.get("ndcg@10"),
                    }
                )
            tag_ret = emb_only.get("gk_tag_retrieval") or {}
            if tag_ret:
                retrieval_rows.append(
                    {
                        "domain": domain,
                        "embedding_model": emb,
                        "precision": prec,
                        "space": "gk_tags",
                        "retrieval_mode": "hybrid_rrf",
                        "hit@1_pct": tag_ret.get("hit@1_pct"),
                        "hit@3_pct": tag_ret.get("hit@3_pct"),
                        "hit@5_pct": tag_ret.get("hit@5_pct"),
                        "hit@10_pct": tag_ret.get("hit@10_pct"),
                        "precision@1_pct": tag_ret.get("precision@1_pct"),
                        "precision@3_pct": tag_ret.get("precision@3_pct"),
                        "recall@3_pct": tag_ret.get("recall@3_pct"),
                        "capped_recall@3_pct": tag_ret.get("capped_recall@3_pct"),
                        "f1@3_pct": tag_ret.get("f1@3_pct"),
                        "mrr": tag_ret.get("mrr"),
                        "map@10": tag_ret.get("map@10"),
                        "ndcg@3": tag_ret.get("ndcg@3"),
                        "ndcg@10": tag_ret.get("ndcg@10"),
                    }
                )

            for head_name, hm in (s.get("head_standalone") or {}).items():
                bt_m = hm.get("bt", {})
                th_m = hm.get("third_tag", {})
                head_rows.append(
                    {
                        "domain": domain,
                        "embedding_model": emb,
                        "precision": prec,
                        "head": head_name,
                        "bt_top1_pct": bt_m.get("accuracy_pct"),
                        "bt_top3_pct": bt_m.get("top3_acc_pct"),
                        "bt_top5_pct": bt_m.get("top5_acc_pct"),
                        "bt_macro_f1_pct": bt_m.get("macro_f1_pct"),
                        "bt_ece": bt_m.get("ece"),
                        "bt_auroc": bt_m.get("confidence_auroc"),
                        "third_top1_pct": th_m.get("accuracy_pct"),
                        "third_top3_pct": th_m.get("top3_acc_pct"),
                        "third_macro_f1_pct": th_m.get("macro_f1_pct"),
                    }
                )

        rerank_diag = s.get("reranker_diagnostics", {})
        cat_diag = rerank_diag.get("catalog", {})
        gk_diag = rerank_diag.get("gk_tags", {})

        for run_key, rm in (s.get("runs") or {}).items():
            parts = run_key.split("__", 2)
            mode = parts[0] if len(parts) > 0 else ""
            head = parts[1] if len(parts) > 1 else ""
            abl = parts[2] if len(parts) > 2 else ""

            bt_m = rm.get("bt", {})
            th_m = rm.get("third_tag", {})
            gk_m = rm.get("gk", {})
            joint_m = rm.get("joint", {})
            row = {
                "domain": domain,
                "embedding_model": emb,
                "reranker_model": rerank,
                "precision": prec,
                "emb_backend": meta.get("emb_backend", ""),
                "rerank_backend": meta.get("rerank_backend", ""),
                "mode": mode,
                "bt_head": head,
                "ablation": abl,
                "test_rows": n_test,
                "bt_acc_pct": bt_m.get("accuracy_pct"),
                "bt_macro_f1_pct": bt_m.get("macro_f1_pct"),
                "bt_seen_acc_pct": ((bt_m.get("by_name_seen") or {}).get("seen_name") or {}).get(
                    "accuracy_pct"
                ),
                "bt_unseen_acc_pct": ((bt_m.get("by_name_seen") or {}).get("unseen_name") or {}).get(
                    "accuracy_pct"
                ),
                "third_tag_acc_pct": th_m.get("accuracy_pct"),
                "third_tag_macro_f1_pct": th_m.get("macro_f1_pct"),
                "gk_sample_f1_pct": gk_m.get("sample_f1_pct"),
                "gk_sample_precision_pct": gk_m.get("sample_precision_pct"),
                "gk_sample_recall_pct": gk_m.get("sample_recall_pct"),
                "gk_jaccard_pct": gk_m.get("sample_jaccard_pct"),
                "gk_exact_pct": gk_m.get("exact_set_match_pct"),
                "gk_micro_f1_pct": gk_m.get("micro_f1_pct"),
                "all3_exact_pct": joint_m.get("all_3_tags_exact_pct"),
                "bt_and_third_exact_pct": joint_m.get("bt_and_third_exact_pct"),
                "cat_pre_hit@1_pct": (cat_diag.get("before_rerank") or {}).get("hit@1_pct"),
                "cat_post_hit@1_pct": (cat_diag.get("after_rerank") or {}).get("hit@1_pct"),
                "cat_post_hit@3_pct": (cat_diag.get("after_rerank") or {}).get("hit@3_pct"),
                "cat_lift_hit@1_pct": (cat_diag.get("lift") or {}).get("hit@1_pct"),
                "cat_post_ndcg@3": (cat_diag.get("after_rerank") or {}).get("ndcg@3"),
                "cat_pair_auroc": (cat_diag.get("pair_scores") or {}).get("pairwise_auroc"),
                "gk_post_hit@3_pct": (gk_diag.get("after_rerank") or {}).get("hit@3_pct"),
                "gk_post_ndcg@3": (gk_diag.get("after_rerank") or {}).get("ndcg@3"),
                "escalation_pct": rm.get("escalation_pct"),
                "rerank_time_s": timings.get("rerank_scoring_s"),
                "total_combo_s": timings.get("total_combo_s"),
            }
            ablation_rows.append(row)
            if abl in ("templates_on__rules_on", "templates_on__rules_off") and mode in (
                "pipeline",
                "matcher",
                "classifier",
            ):
                if mode == "matcher" or abl == "templates_on__rules_on":
                    e2e_rows.append(row)

    out_paths: dict[str, Path] = {}
    for name, rows in (
        ("leaderboard.csv", e2e_rows),
        ("ablations_all.csv", ablation_rows),
        ("retrieval_and_embedding.csv", retrieval_rows),
        ("bt_heads_standalone.csv", head_rows),
    ):
        df = pd.DataFrame(rows)
        if not df.empty and "all3_exact_pct" in df.columns:
            df = df.sort_values(
                by=["domain", "mode", "all3_exact_pct", "bt_acc_pct"],
                ascending=[True, True, False, False],
            )
        p = reports_dir / name
        df.to_csv(p, index=False, encoding="utf-8")
        out_paths[name] = p

    # Build concise Markdown summary
    md_lines = [
        "# SKU MatchOps Evaluation Report",
        "",
        f"Aggregated **{len(summaries)}** completed `(domain, embedding, reranker, precision)` combinations.",
        "",
    ]

    e2e_df = pd.DataFrame(e2e_rows)
    if not e2e_df.empty:
        pipe_df = e2e_df[e2e_df["mode"] == "pipeline"]
        for dom in sorted(pipe_df["domain"].unique()):
            sub = pipe_df[pipe_df["domain"] == dom].sort_values(
                by=["all3_exact_pct", "bt_acc_pct"], ascending=False
            )
            md_lines.extend([f"## Best End-to-End Pipeline (`{dom.upper()}`)", ""])
            cols = [
                "embedding_model",
                "reranker_model",
                "precision",
                "bt_head",
                "bt_acc_pct",
                "third_tag_acc_pct",
                "gk_sample_f1_pct",
                "all3_exact_pct",
                "cat_post_hit@3_pct",
                "cat_lift_hit@1_pct",
            ]
            avail_cols = [c for c in cols if c in sub.columns]
            md_lines.append(_df_to_markdown(sub[avail_cols].head(12)))
            md_lines.append("")

    ret_df = pd.DataFrame(retrieval_rows)
    if not ret_df.empty:
        md_lines.extend(["## Standalone Retrieval (`Hit@3`, `MRR`, `NDCG@3`, `NDCG@10`)", ""])
        md_lines.append(
            _df_to_markdown(
                ret_df.sort_values(
                    by=["domain", "space", "ndcg@3"], ascending=[True, True, False]
                ).head(24)
            )
        )
        md_lines.append("")

    head_df = pd.DataFrame(head_rows)
    if not head_df.empty:
        md_lines.extend(["## Standalone BT & Third-Tag Classifier Heads", ""])
        md_lines.append(
            _df_to_markdown(head_df.sort_values(by=["domain", "bt_top1_pct"], ascending=[True, False]))
        )
        md_lines.append("")

    md_path = reports_dir / "report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    out_paths["report.md"] = md_path
    logger.info(f"[REPORT] Generated reports in {reports_dir}")
    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate SKU MatchOps evaluation results.")
    parser.add_argument(
        "--work-dir",
        default=str(Path(__file__).resolve().parent / "runs"),
        help="Evaluation work directory containing results/*/summary.json",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    paths = generate_reports(args.work_dir)
    for k, p in paths.items():
        print(f"{k}: {p}")


if __name__ == "__main__":
    main()
