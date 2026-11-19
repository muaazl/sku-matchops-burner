"""Comprehensive classification, multi-label, retrieval, reranking, calibration, and ablation metrics."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

RANK_KS = (1, 3, 5, 10, 30)


def norm_label(val: Any) -> str:
    """Normalizes a tag string for case/whitespace-insensitive exact comparison."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    s = str(val).strip().lower()
    return "" if s in ("", "nan", "none", "<na>", "null") else s


def parse_gk_list(raw: Any) -> list[str]:
    """Parses comma-separated or list-formatted generic keywords preserving order, deduplicated."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    items = raw if isinstance(raw, (list, tuple, set)) else str(raw).split(",")
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        s = str(item or "").strip()
        low = s.lower()
        if not low or low in ("nan", "none", "<na>", "null") or low in seen:
            continue
        seen.add(low)
        out.append(s)
    return out


def parse_gk_set(raw: Any) -> set[str]:
    return {x.lower() for x in parse_gk_list(raw)}


def bucket_status(status: str) -> str:
    """Normalizes matcher/classifier statuses into AUTO / REVIEW / LOW buckets."""
    s = norm_label(status)
    if s in ("auto", "high confidence", "high", "exact text match"):
        return "AUTO"
    if s in ("review", "medium confidence", "medium"):
        return "REVIEW"
    return "LOW"


def _safe_pct(num: float, den: float) -> float:
    return round(100.0 * float(num) / float(den), 4) if den > 0 else 0.0


def compute_ece_and_brier(y_correct: np.ndarray, confs: np.ndarray, n_bins: int = 10) -> tuple[float, float]:
    """Computes Expected Calibration Error (ECE) and Brier score for confidence estimates."""
    if len(y_correct) == 0:
        return 0.0, 0.0
    c = np.clip(np.asarray(confs, dtype=np.float64), 0.0, 1.0)
    y = np.asarray(y_correct, dtype=np.float64)
    brier = float(np.mean((c - y) ** 2))

    ece = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y)
    for idx in range(n_bins):
        lo, hi = bins[idx], bins[idx + 1]
        mask = (c >= lo) & (c <= hi if idx == n_bins - 1 else c < hi)
        if np.any(mask):
            acc_bin = float(np.mean(y[mask]))
            conf_bin = float(np.mean(c[mask]))
            ece += (float(np.sum(mask)) / n) * abs(acc_bin - conf_bin)
    return round(float(ece), 5), round(brier, 5)


def compute_multiclass_metrics(
    gt_labels: list[str],
    pred_labels: list[str],
    confidences: list[float] | None = None,
    statuses: list[str] | None = None,
    sources: list[str] | None = None,
    topk_preds: list[list[str]] | None = None,
    train_class_counts: dict[str, int] | None = None,
    seen_sku_mask: list[bool] | None = None,
) -> dict[str, Any]:
    """Computes rich single-label classification metrics, calibration, slices, and error analysis."""
    n = len(gt_labels)
    if n == 0:
        return {"count": 0}

    gt_norm = [norm_label(x) for x in gt_labels]
    pred_norm = [norm_label(x) for x in pred_labels]
    correct = np.array([g == p and g != "" for g, p in zip(gt_norm, pred_norm)], dtype=np.int64)

    acc = accuracy_score(gt_norm, pred_norm) * 100.0
    with np.errstate(all="ignore"):
        bal_acc = balanced_accuracy_score(gt_norm, pred_norm) * 100.0
    macro_f1 = f1_score(gt_norm, pred_norm, average="macro", zero_division=0) * 100.0
    micro_f1 = f1_score(gt_norm, pred_norm, average="micro", zero_division=0) * 100.0
    weighted_f1 = f1_score(gt_norm, pred_norm, average="weighted", zero_division=0) * 100.0

    out: dict[str, Any] = {
        "count": n,
        "correct_count": int(np.sum(correct)),
        "accuracy_pct": round(float(acc), 4),
        "balanced_accuracy_pct": round(float(bal_acc), 4),
        "macro_f1_pct": round(float(macro_f1), 4),
        "micro_f1_pct": round(float(micro_f1), 4),
        "weighted_f1_pct": round(float(weighted_f1), 4),
        "empty_pred_count": int(sum(1 for p in pred_norm if not p)),
    }

    if topk_preds is not None and len(topk_preds) == n:
        for k in (1, 3, 5):
            hits = sum(
                1
                for g, preds in zip(gt_norm, topk_preds)
                if g and g in [norm_label(p) for p in preds[:k]]
            )
            out[f"top{k}_acc_pct"] = _safe_pct(hits, n)

    if confidences is not None and len(confidences) == n:
        confs_arr = np.asarray([float(c or 0.0) for c in confidences], dtype=np.float64)
        ece, brier = compute_ece_and_brier(correct, confs_arr)
        out["ece"] = ece
        out["brier_score"] = brier
        if len(np.unique(correct)) == 2:
            out["confidence_auroc"] = round(float(roc_auc_score(correct, confs_arr)), 5)
            out["confidence_prauc"] = round(float(average_precision_score(correct, confs_arr)), 5)
        else:
            out["confidence_auroc"] = None
            out["confidence_prauc"] = None

    if statuses is not None and len(statuses) == n:
        by_status: dict[str, dict[str, Any]] = {}
        buckets = [bucket_status(s) for s in statuses]
        for b in ("AUTO", "REVIEW", "LOW"):
            idxs = [i for i, bv in enumerate(buckets) if bv == b]
            b_corr = int(sum(correct[i] for i in idxs))
            by_status[b] = {
                "count": len(idxs),
                "coverage_pct": _safe_pct(len(idxs), n),
                "accuracy_pct": _safe_pct(b_corr, len(idxs)),
            }
        out["by_status"] = by_status

    if sources is not None and len(sources) == n:
        by_source: dict[str, dict[str, Any]] = {}
        src_groups: dict[str, list[int]] = {}
        for i, src in enumerate(sources):
            src_key = str(src or "unknown").strip() or "unknown"
            src_groups.setdefault(src_key, []).append(i)
        for src_key, idxs in sorted(src_groups.items(), key=lambda x: -len(x[1])):
            s_corr = int(sum(correct[i] for i in idxs))
            by_source[src_key] = {
                "count": len(idxs),
                "share_pct": _safe_pct(len(idxs), n),
                "accuracy_pct": _safe_pct(s_corr, len(idxs)),
            }
        out["by_source"] = by_source

    if train_class_counts is not None:
        counts_norm = {norm_label(k): int(v) for k, v in train_class_counts.items()}
        tiers = {"warm_ge_15": [], "few_shot_1_to_14": [], "zero_shot_0": []}
        for i, g in enumerate(gt_norm):
            cnt = counts_norm.get(g, 0)
            if cnt >= 15:
                tiers["warm_ge_15"].append(i)
            elif cnt >= 1:
                tiers["few_shot_1_to_14"].append(i)
            else:
                tiers["zero_shot_0"].append(i)
        out["by_frequency_tier"] = {
            tier: {
                "count": len(idxs),
                "accuracy_pct": _safe_pct(sum(correct[i] for i in idxs), len(idxs)),
                "macro_f1_pct": round(
                    float(
                        f1_score(
                            [gt_norm[i] for i in idxs],
                            [pred_norm[i] for i in idxs],
                            average="macro",
                            zero_division=0,
                        )
                        * 100.0
                    ),
                    4,
                )
                if idxs
                else 0.0,
            }
            for tier, idxs in tiers.items()
        }

    if seen_sku_mask is not None and len(seen_sku_mask) == n:
        seen_idx = [i for i, flag in enumerate(seen_sku_mask) if flag]
        unseen_idx = [i for i, flag in enumerate(seen_sku_mask) if not flag]
        out["by_name_seen"] = {
            "seen_name": {
                "count": len(seen_idx),
                "accuracy_pct": _safe_pct(sum(correct[i] for i in seen_idx), len(seen_idx)),
            },
            "unseen_name": {
                "count": len(unseen_idx),
                "accuracy_pct": _safe_pct(sum(correct[i] for i in unseen_idx), len(unseen_idx)),
            },
        }

    # Error diagnostics: top 20 confusions and worst classes (support >= 5)
    confusions = Counter(
        (gt_norm[i], pred_norm[i]) for i in range(n) if gt_norm[i] != pred_norm[i]
    )
    out["top_confusions"] = [
        {"gt": g, "pred": p or "<empty>", "count": c} for (g, p), c in confusions.most_common(20)
    ]

    gt_counts = Counter(gt_norm)
    pred_counts = Counter(pred_norm)
    tp_counts = Counter(gt_norm[i] for i in range(n) if gt_norm[i] == pred_norm[i])
    class_rows = []
    for lbl, supp in gt_counts.items():
        if supp < 5:
            continue
        tp = tp_counts[lbl]
        p_den = pred_counts[lbl]
        prec = tp / p_den if p_den > 0 else 0.0
        rec = tp / supp if supp > 0 else 0.0
        f1_c = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        class_rows.append(
            {
                "label": lbl,
                "support": supp,
                "precision_pct": round(prec * 100.0, 2),
                "recall_pct": round(rec * 100.0, 2),
                "f1_pct": round(f1_c * 100.0, 2),
            }
        )
    class_rows.sort(key=lambda r: (r["f1_pct"], -r["support"]))
    out["worst_classes"] = class_rows[:15]
    return out


def compute_gk_metrics(
    gt_gk_raw: list[Any],
    pred_gk_raw: list[Any],
    train_seen_gks: set[str] | None = None,
    statuses: list[str] | None = None,
) -> dict[str, Any]:
    """Computes multi-label GenericKeywords metrics over non-empty GT rows + null-GT stats."""
    n_total = len(gt_gk_raw)
    gt_sets = [parse_gk_set(x) for x in gt_gk_raw]
    pred_sets = [parse_gk_set(x) for x in pred_gk_raw]

    null_gt_indices = [i for i, gs in enumerate(gt_sets) if not gs]
    eval_indices = [i for i, gs in enumerate(gt_sets) if gs]

    null_empty_preds = sum(1 for i in null_gt_indices if not pred_sets[i])
    if not eval_indices:
        return {
            "total_rows": n_total,
            "eval_rows": 0,
            "null_gt_rows": len(null_gt_indices),
            "null_gt_predicted_empty_pct": _safe_pct(null_empty_preds, len(null_gt_indices)),
        }

    precs, recs, f1s, jaccards, exacts = [], [], [], [], []
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0

    tag_tp: Counter[str] = Counter()
    tag_fp: Counter[str] = Counter()
    tag_fn: Counter[str] = Counter()

    seen_gt_total = 0
    seen_gt_hit = 0
    unseen_gt_total = 0
    unseen_gt_hit = 0

    for i in eval_indices:
        gs = gt_sets[i]
        ps = pred_sets[i]
        inter = gs & ps
        union = gs | ps

        tp = len(inter)
        fp = len(ps - gs)
        fn = len(gs - ps)
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

        p_i = tp / len(ps) if ps else 0.0
        r_i = tp / len(gs)
        f1_i = (2 * p_i * r_i / (p_i + r_i)) if (p_i + r_i) > 0 else 0.0
        jac_i = len(inter) / len(union) if union else 0.0

        precs.append(p_i)
        recs.append(r_i)
        f1s.append(f1_i)
        jaccards.append(jac_i)
        exacts.append(1.0 if gs == ps else 0.0)

        for t in inter:
            tag_tp[t] += 1
        for t in ps - gs:
            tag_fp[t] += 1
        for t in gs - ps:
            tag_fn[t] += 1

        if train_seen_gks is not None:
            for t in gs:
                if t in train_seen_gks:
                    seen_gt_total += 1
                    if t in ps:
                        seen_gt_hit += 1
                else:
                    unseen_gt_total += 1
                    if t in ps:
                        unseen_gt_hit += 1

    micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

    all_gt_tags = set(tag_tp.keys()) | set(tag_fn.keys())
    tag_f1s = []
    for t in all_gt_tags:
        tp = tag_tp[t]
        fp = tag_fp[t]
        fn = tag_fn[t]
        denom = 2 * tp + fp + fn
        tag_f1s.append((2 * tp / denom) if denom > 0 else 0.0)

    out: dict[str, Any] = {
        "total_rows": n_total,
        "eval_rows": len(eval_indices),
        "null_gt_rows": len(null_gt_indices),
        "null_gt_predicted_empty_pct": _safe_pct(null_empty_preds, len(null_gt_indices)),
        "exact_set_match_pct": round(float(np.mean(exacts) * 100.0), 4),
        "sample_precision_pct": round(float(np.mean(precs) * 100.0), 4),
        "sample_recall_pct": round(float(np.mean(recs) * 100.0), 4),
        "sample_f1_pct": round(float(np.mean(f1s) * 100.0), 4),
        "sample_jaccard_pct": round(float(np.mean(jaccards) * 100.0), 4),
        "micro_precision_pct": round(micro_p * 100.0, 4),
        "micro_recall_pct": round(micro_r * 100.0, 4),
        "micro_f1_pct": round(micro_f1 * 100.0, 4),
        "macro_f1_gt_tags_pct": round(float(np.mean(tag_f1s) * 100.0), 4) if tag_f1s else 0.0,
        "avg_gt_tags_per_sku": round(float(np.mean([len(gt_sets[i]) for i in eval_indices])), 3),
        "avg_pred_tags_per_sku": round(float(np.mean([len(pred_sets[i]) for i in eval_indices])), 3),
    }

    if train_seen_gks is not None:
        out["seen_gk_tag_recall_pct"] = _safe_pct(seen_gt_hit, seen_gt_total)
        out["unseen_gk_tag_recall_pct"] = _safe_pct(unseen_gt_hit, unseen_gt_total)
        out["seen_gk_gt_instances"] = seen_gt_total
        out["unseen_gk_gt_instances"] = unseen_gt_total

    if statuses is not None and len(statuses) == n_total:
        by_status: dict[str, dict[str, Any]] = {}
        buckets = [bucket_status(s) for s in statuses]
        for b in ("AUTO", "REVIEW", "LOW"):
            b_idx = [i for i in eval_indices if buckets[i] == b]
            by_status[b] = {
                "count": len(b_idx),
                "coverage_pct": _safe_pct(len(b_idx), len(eval_indices)),
                "sample_f1_pct": round(
                    float(
                        np.mean(
                            [
                                (
                                    2 * len(gt_sets[i] & pred_sets[i]) / (len(gt_sets[i]) + len(pred_sets[i]))
                                )
                                if (gt_sets[i] or pred_sets[i])
                                else 0.0
                                for i in b_idx
                            ]
                        )
                        * 100.0
                    ),
                    4,
                )
                if b_idx
                else 0.0,
            }
        out["by_status"] = by_status

    return out


def compute_joint_metrics(
    gt_bt: list[str],
    pred_bt: list[str],
    gt_third: list[str],
    pred_third: list[str],
    gt_gk: list[Any],
    pred_gk: list[Any],
) -> dict[str, Any]:
    """Computes joint end-to-end SKU exact match across BT, Third-Tag, and GenericKeywords."""
    n = len(gt_bt)
    if n == 0:
        return {"count": 0}
    bt_ok = [norm_label(g) == norm_label(p) and norm_label(g) != "" for g, p in zip(gt_bt, pred_bt)]
    third_ok = [norm_label(g) == norm_label(p) and norm_label(g) != "" for g, p in zip(gt_third, pred_third)]
    bt_third_ok = sum(1 for b, t in zip(bt_ok, third_ok) if b and t)

    gk_eval_idx = [i for i, raw in enumerate(gt_gk) if parse_gk_set(raw)]
    all3_exact = sum(
        1
        for i in gk_eval_idx
        if bt_ok[i] and third_ok[i] and parse_gk_set(gt_gk[i]) == parse_gk_set(pred_gk[i])
    )
    return {
        "count": n,
        "bt_and_third_exact_pct": _safe_pct(bt_third_ok, n),
        "all_3_tags_exact_pct": _safe_pct(all3_exact, len(gk_eval_idx)),
        "all_3_eval_rows": len(gk_eval_idx),
    }


def _dcg(relevances: list[float], k: int) -> float:
    return sum((2.0**rel - 1.0) / math.log2(rank + 2) for rank, rel in enumerate(relevances[:k]))


def compute_ranking_metrics(
    binary_rels_per_query: list[list[int]],
    total_relevant_per_query: list[int],
    graded_rels_per_query: list[list[float]] | None = None,
    ideal_graded_per_query: list[list[float]] | None = None,
    ks: tuple[int, ...] = RANK_KS,
) -> dict[str, Any]:
    """Computes Hit@K, Precision@K, Recall@K, CappedRecall@K, F1@K, MRR, MAP@K, NDCG@K."""
    n = len(binary_rels_per_query)
    if n == 0:
        return {"query_count": 0}

    valid_idx = [i for i, tot in enumerate(total_relevant_per_query) if tot > 0]
    m = len(valid_idx)
    if m == 0:
        return {"query_count": n, "evaluable_queries": 0}

    mrr_sum = 0.0
    hits = dict.fromkeys(ks, 0)
    prec_sum = dict.fromkeys(ks, 0.0)
    rec_sum = dict.fromkeys(ks, 0.0)
    capped_rec_sum = dict.fromkeys(ks, 0.0)
    f1_sum = dict.fromkeys(ks, 0.0)
    ap_sum = dict.fromkeys(ks, 0.0)
    ndcg_sum = dict.fromkeys(ks, 0.0)

    for i in valid_idx:
        brels = binary_rels_per_query[i]
        tot_rel = total_relevant_per_query[i]
        grels = graded_rels_per_query[i] if graded_rels_per_query is not None else [float(r) for r in brels]
        ideal_grels = (
            ideal_graded_per_query[i]
            if ideal_graded_per_query is not None
            else sorted(grels, reverse=True) + [0.0] * max(0, max(ks) - len(grels))
        )

        # MRR (over full returned list)
        first_pos = next((rank + 1 for rank, r in enumerate(brels) if r > 0), None)
        if first_pos is not None:
            mrr_sum += 1.0 / first_pos

        for k in ks:
            top_b = brels[:k]
            rel_in_k = sum(1 for r in top_b if r > 0)
            if rel_in_k > 0:
                hits[k] += 1
            p_k = rel_in_k / k
            r_k = rel_in_k / tot_rel
            cr_k = rel_in_k / min(k, tot_rel)
            f1_k = (2 * p_k * cr_k / (p_k + cr_k)) if (p_k + cr_k) > 0 else 0.0

            prec_sum[k] += p_k
            rec_sum[k] += r_k
            capped_rec_sum[k] += cr_k
            f1_sum[k] += f1_k

            # AP@K
            cum_rel = 0
            ap_k = 0.0
            for r_idx, rel in enumerate(top_b):
                if rel > 0:
                    cum_rel += 1
                    ap_k += cum_rel / (r_idx + 1)
            ap_sum[k] += ap_k / min(k, tot_rel)

            # NDCG@K
            idcg_k = _dcg(ideal_grels, k)
            ndcg_sum[k] += (_dcg(grels, k) / idcg_k) if idcg_k > 0 else 0.0

    metrics: dict[str, Any] = {
        "query_count": n,
        "evaluable_queries": m,
        "mrr": round(mrr_sum / m, 5),
    }
    for k in ks:
        metrics[f"hit@{k}_pct"] = _safe_pct(hits[k], m)
        metrics[f"precision@{k}_pct"] = round(100.0 * prec_sum[k] / m, 4)
        metrics[f"recall@{k}_pct"] = round(100.0 * rec_sum[k] / m, 4)
        metrics[f"capped_recall@{k}_pct"] = round(100.0 * capped_rec_sum[k] / m, 4)
        metrics[f"f1@{k}_pct"] = round(100.0 * f1_sum[k] / m, 4)
        metrics[f"map@{k}"] = round(ap_sum[k] / m, 5)
        metrics[f"ndcg@{k}"] = round(ndcg_sum[k] / m, 5)
    return metrics


def compute_reranker_diagnostics(
    pre_binary_rels: list[list[int]],
    post_binary_rels: list[list[int]],
    total_relevant_per_query: list[int],
    pair_labels: list[int],
    pair_scores: list[float],
    pre_graded_rels: list[list[float]] | None = None,
    post_graded_rels: list[list[float]] | None = None,
    ideal_graded_per_query: list[list[float]] | None = None,
    top1_scores: list[float] | None = None,
    top1_correct: list[int] | None = None,
) -> dict[str, Any]:
    """Measures reranker lift over pre-rerank retrieval + pairwise ROC/PR + threshold sweep."""
    before = compute_ranking_metrics(
        pre_binary_rels,
        total_relevant_per_query,
        graded_rels_per_query=pre_graded_rels,
        ideal_graded_per_query=ideal_graded_per_query,
    )
    after = compute_ranking_metrics(
        post_binary_rels,
        total_relevant_per_query,
        graded_rels_per_query=post_graded_rels,
        ideal_graded_per_query=ideal_graded_per_query,
    )

    wins = 0
    losses = 0
    ties = 0
    for pre_b, post_b in zip(pre_binary_rels, post_binary_rels):
        r_pre = next((1.0 / (idx + 1) for idx, v in enumerate(pre_b) if v > 0), 0.0)
        r_post = next((1.0 / (idx + 1) for idx, v in enumerate(post_b) if v > 0), 0.0)
        if r_post > r_pre + 1e-9:
            wins += 1
        elif r_post < r_pre - 1e-9:
            losses += 1
        else:
            ties += 1

    lift = {
        key: round(after[key] - before[key], 4)
        for key in ("hit@1_pct", "hit@3_pct", "precision@1_pct", "precision@3_pct", "mrr", "ndcg@3", "ndcg@10")
        if key in before and key in after
    }

    pair_stats: dict[str, Any] = {"pair_count": len(pair_labels)}
    if pair_labels and len(pair_labels) == len(pair_scores):
        y_arr = np.asarray(pair_labels, dtype=np.int64)
        s_arr = np.asarray(pair_scores, dtype=np.float64)
        pos = s_arr[y_arr == 1]
        neg = s_arr[y_arr == 0]
        if len(pos) > 0:
            pair_stats["pos_mean"] = round(float(np.mean(pos)), 5)
            pair_stats["pos_median"] = round(float(np.median(pos)), 5)
        if len(neg) > 0:
            pair_stats["neg_mean"] = round(float(np.mean(neg)), 5)
            pair_stats["neg_median"] = round(float(np.median(neg)), 5)
        if len(np.unique(y_arr)) == 2:
            pair_stats["pairwise_auroc"] = round(float(roc_auc_score(y_arr, s_arr)), 5)
            pair_stats["pairwise_prauc"] = round(float(average_precision_score(y_arr, s_arr)), 5)

    # Optimal score threshold sweep on Top-1 decisions (or pairwise labels)
    thr_sweep: list[dict[str, Any]] = []
    if top1_scores is not None and top1_correct is not None and len(top1_scores) == len(top1_correct) and top1_scores:
        sc = np.asarray(top1_scores, dtype=np.float64)
        ok = np.asarray(top1_correct, dtype=np.int64)
        for thr in np.round(np.arange(0.15, 0.91, 0.05), 2):
            accepted = sc >= thr
            n_acc = int(np.sum(accepted))
            tp = int(np.sum(accepted & (ok == 1)))
            prec = tp / n_acc if n_acc > 0 else 0.0
            rec = tp / len(ok) if len(ok) > 0 else 0.0
            f1_v = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            thr_sweep.append(
                {
                    "threshold": float(thr),
                    "coverage_pct": _safe_pct(n_acc, len(ok)),
                    "precision_pct": round(prec * 100.0, 2),
                    "recall_pct": round(rec * 100.0, 2),
                    "f1_pct": round(f1_v * 100.0, 2),
                }
            )
    best_thr = max(thr_sweep, key=lambda r: r["f1_pct"]) if thr_sweep else None

    return {
        "before_rerank": before,
        "after_rerank": after,
        "lift": lift,
        "win_tie_loss": {"wins": wins, "ties": ties, "losses": losses},
        "pair_scores": pair_stats,
        "best_top1_threshold": best_thr,
        "threshold_sweep": thr_sweep,
    }


def catalog_candidate_relevance(
    gt_bt: str,
    gt_third: str,
    gt_gk_set: set[str],
    cand_row: dict[str, Any],
    third_col: str,
) -> tuple[int, float]:
    """Returns (binary_rel, graded_rel) for a retrieved catalog candidate."""
    c_bt = norm_label(cand_row.get("basictype", cand_row.get("BasicType", "")))
    c_third = norm_label(cand_row.get(third_col, cand_row.get("region", cand_row.get("category", ""))))
    c_gk = parse_gk_set(cand_row.get("Generic keywords", cand_row.get("GenericKeywords", "")))

    same_bt = bool(gt_bt and c_bt == gt_bt)
    same_third = bool(gt_third and c_third == gt_third)
    gk_jac = (len(gt_gk_set & c_gk) / len(gt_gk_set | c_gk)) if (gt_gk_set and c_gk) else 0.0

    if same_bt and same_third and gk_jac >= 0.5:
        graded = 3.0
    elif same_bt:
        graded = 2.0
    elif same_third and gk_jac > 0.0:
        graded = 1.0
    else:
        graded = 0.0

    return (1 if same_bt else 0), graded


def build_per_sku_df(
    test_df: pd.DataFrame,
    third_col_csv: str,
    predictions: list[dict[str, Any]],
    extra_cols: dict[str, list[Any]] | None = None,
) -> pd.DataFrame:
    """Constructs a per-SKU evaluation DataFrame with per-tag accuracy percentages & diagnostics."""
    rows: list[dict[str, Any]] = []
    for idx, (_, gt_row) in enumerate(test_df.iterrows()):
        pred = predictions[idx] if idx < len(predictions) else {}
        gt_bt = str(gt_row.get("BasicType", "") or "").strip()
        gt_third = str(gt_row.get(third_col_csv, "") or "").strip()
        gt_gk_raw = gt_row.get("GenericKeywords", "")
        gt_gk_set = parse_gk_set(gt_gk_raw)

        p_bt = str(pred.get("suggested_bt", "") or "").strip()
        p_third = str(pred.get("suggested_region", "") or "").strip()
        p_gk_raw = pred.get("suggested_gk", "")
        p_gk_set = parse_gk_set(p_gk_raw)

        bt_match = int(norm_label(gt_bt) == norm_label(p_bt) and norm_label(gt_bt) != "")
        third_match = int(norm_label(gt_third) == norm_label(p_third) and norm_label(gt_third) != "")

        if gt_gk_set:
            inter = len(gt_gk_set & p_gk_set)
            union = len(gt_gk_set | p_gk_set)
            gk_prec = round(100.0 * inter / len(p_gk_set), 2) if p_gk_set else 0.0
            gk_rec = round(100.0 * inter / len(gt_gk_set), 2)
            gk_f1 = round((2 * gk_prec * gk_rec / (gk_prec + gk_rec)), 2) if (gk_prec + gk_rec) > 0 else 0.0
            gk_jac = round(100.0 * inter / union, 2) if union > 0 else 0.0
            gk_exact = int(gt_gk_set == p_gk_set)
            all3 = int(bt_match == 1 and third_match == 1 and gk_exact == 1)
        else:
            gk_prec = gk_rec = gk_f1 = gk_jac = np.nan
            gk_exact = np.nan
            all3 = np.nan

        rec: dict[str, Any] = {
            "sku_index": idx,
            "name": str(gt_row.get("Name", "") or ""),
            "seller_category": str(gt_row.get("SellerCategory", "") or ""),
            "gt_bt": gt_bt,
            "pred_bt": p_bt,
            "bt_acc_pct": 100.0 * bt_match,
            "gt_third_tag": gt_third,
            "pred_third_tag": p_third,
            "third_tag_acc_pct": 100.0 * third_match,
            "gt_gk": "" if pd.isna(gt_gk_raw) else str(gt_gk_raw),
            "pred_gk": str(p_gk_raw),
            "gk_precision_pct": gk_prec,
            "gk_recall_pct": gk_rec,
            "gk_f1_pct": gk_f1,
            "gk_jaccard_pct": gk_jac,
            "gk_exact_pct": 100.0 * gk_exact if not pd.isna(gk_exact) else np.nan,
            "all3_exact_pct": 100.0 * all3 if not pd.isna(all3) else np.nan,
            "score_or_bt_conf": pred.get("score") if "score" in pred else pred.get("bt_confidence"),
            "status": str(pred.get("status") or pred.get("bt_status") or ""),
            "pipeline_source": str(pred.get("pipeline_source") or pred.get("bt_source") or ""),
            "escalated": bool(pred.get("escalated", False)),
            "matched_catalog_name": str(pred.get("matched_catalog_name", "") or ""),
            "rules_applied": str(pred.get("rules_applied", "") or ""),
        }
        if extra_cols:
            for col_name, col_vals in extra_cols.items():
                rec[col_name] = col_vals[idx] if idx < len(col_vals) else None
        rows.append(rec)

    return pd.DataFrame(rows)
