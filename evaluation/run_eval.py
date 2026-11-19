"""Checkpoint-resumable, leak-free evaluation harness for SKU MatchOps (Colab & local)."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import config
from engine.classification.classifier import TaxonomyClassifier
from engine.classification.loader import (
    augment_bt_gk_map_with_training,
    build_bt_third_tag_map_from_catalog,
    build_descriptions,
    build_umbrella_from_training,
)
from engine.classification.tagger import compute_weighted_confidence, get_strategy
from engine.core import resource_loader
from engine.data_pipeline.cache_manager import (
    CacheManager,
    calculate_row_hash,
    clean_price,
)
from engine.data_pipeline.ingestion import DataIngestion
from engine.data_pipeline.vector_store import VectorStore
from engine.matching.logic_gates import LogicGates
from engine.matching.matcher import SKUMatcher
from engine.nlp.ner_engine import NEREngine
from engine.nlp.text_cleaner import TextPipeline
from engine.pipeline.processor import _apply_template_tag_enrichment, process_request
from engine.rules_engine import run_rules_engine
from engine.rules_engine.loader import set_rules_from_dicts
from engine.templates import template_suggest
from evaluation import metrics as M
from evaluation.model_manager import (
    EMBEDDING_MODELS,
    PRECISION_MODES,
    RERANKER_MODELS,
    EvalEmbeddingEngine,
    export_arcface_precisions,
)
from evaluation.report import generate_reports

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("matchops.eval.run")

EVAL_DIR = PROJECT_ROOT / "evaluation"
STAGED_FIXTURES_DIR = EVAL_DIR / "fixtures" / "staged_sheets"
RULES_SNAPSHOT_PATH = EVAL_DIR / "fixtures" / "rules_snapshot.json"


def _slug(name: str) -> str:
    return config.model_id_to_folder_name(name)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    joblib.dump(payload, tmp)
    os.replace(tmp, path)


def _combo_id(
    domain: str,
    emb_slug: str,
    rerank_slug: str,
    prec: str,
    max_test_rows: int,
    n_test: int,
) -> str:
    cid = f"{domain}__{emb_slug}__{rerank_slug}__{prec}"
    if max_test_rows > 0:
        cid += f"__n{n_test}"
    return cid


def _load_registry(work_dir: Path) -> dict[str, Any]:
    reg_path = work_dir / "checkpoint_registry.json"
    if reg_path.exists():
        try:
            return json.loads(reg_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"stage_a": {}, "stage_b": {}, "stage_c": {}}


def _save_registry(work_dir: Path, registry: dict[str, Any]) -> None:
    _atomic_write_json(work_dir / "checkpoint_registry.json", registry)


def _stratified_sample_df(
    df: pd.DataFrame, max_rows: int, seed: int = 42, col: str = "BasicType"
) -> pd.DataFrame:
    """Selects a deterministic stratified slice preserving class proportions."""
    if max_rows <= 0 or len(df) <= max_rows:
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    indices_by_cls: dict[str, list[int]] = {}
    for idx, val in enumerate(df[col].fillna("").astype(str).tolist()):
        indices_by_cls.setdefault(val, []).append(idx)

    selected: list[int] = []
    # Proportional allocation with deterministic floor + remainder sampling
    classes = sorted(indices_by_cls.keys())
    quotas = {}
    fracs = []
    for c in classes:
        exact = len(indices_by_cls[c]) * max_rows / len(df)
        q = int(exact)
        quotas[c] = q
        fracs.append((exact - q, c))
    rem = max_rows - sum(quotas.values())
    for _, c in sorted(fracs, reverse=True)[:rem]:
        quotas[c] += 1

    for c in classes:
        pool = list(indices_by_cls[c])
        rng.shuffle(pool)
        selected.extend(pool[: quotas[c]])
    selected.sort()
    return df.iloc[selected].reset_index(drop=True)


def _prepare_train_catalog_df(raw_df: pd.DataFrame, domain: str) -> pd.DataFrame:
    """Formats evaluation/train_{domain}.csv identically to DataIngestion.load_catalog_from_sheets."""
    df = raw_df.copy()
    col_map = (
        config.CATALOG_COL_MAP_FOOD
        if domain == config.DOMAIN_FOOD
        else config.CATALOG_COL_MAP_MARKET
    )
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df = df[df["Name"].notna() & (df["Name"].astype(str).str.strip() != "")].reset_index(drop=True)

    raw_names = df["Name"].fillna("").astype(str).str.strip().str.lower()
    occ = raw_names.groupby(raw_names).cumcount() + 1
    df["db_uid"] = raw_names + "#occ_" + occ.astype(str)

    clean_texts, clean_no_weights, weights, tok_counts, row_hashes = [], [], [], [], []
    for row in df.to_dict(orient="records"):
        ct = TextPipeline.normalize_final(TextPipeline.standardize_units(str(row.get("Name", ""))))
        clean_texts.append(ct)
        clean_no_weights.append(TextPipeline.strip_weights(ct))
        weights.append(TextPipeline.extract_weight_feature(ct))
        tok_counts.append(len(ct.split()))
        row_hashes.append(calculate_row_hash(row, domain=domain))

    df["clean_text"] = clean_texts
    df["clean_no_weights"] = clean_no_weights
    df["weight_val"] = weights
    df["token_count"] = tok_counts
    df["row_hash"] = row_hashes
    if "Price" in df.columns:
        df["price"] = df["Price"].apply(clean_price)
    return df


_BRAND_COL_MAP = {
    "aliases": "Aliases",
    "is_weak": "Is_Weak",
    "is_meat": "Is_Meat",
    "is_vegetable": "Is_Vegetable",
    "is_seafood": "Is_Seafood",
}


def _load_staged_brands_df(domain: str) -> pd.DataFrame:
    orig_staging = config.STAGING_DIR
    config.STAGING_DIR = STAGED_FIXTURES_DIR
    try:
        brand_sheet = "Food_Flavors" if domain == config.DOMAIN_FOOD else "Market_Brands"
        df = DataIngestion._fetch_sheet_as_csv("EVAL_LOCAL", brand_sheet)
        col_map = {
            **_BRAND_COL_MAP,
            "name": "Flavor Name" if domain == config.DOMAIN_FOOD else "Brand Name",
        }
        df = df.rename(columns=col_map)
        for col in ("Is_Weak", "Is_Meat", "Is_Vegetable", "Is_Seafood"):
            if col in df.columns and not pd.api.types.is_bool_dtype(df[col]):
                df[col] = (
                    df[col]
                    .fillna(False)
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin(("true", "1", "1.0", "yes", "y"))
                )
        return df
    finally:
        config.STAGING_DIR = orig_staging


def _load_staged_domain_knowledge(domain: str) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Loads Food_Flavors / Market_Brands and taxonomy dictionaries strictly from fixtures."""
    orig_staging = config.STAGING_DIR
    orig_cache = config.CACHE_DIR
    config.STAGING_DIR = STAGED_FIXTURES_DIR
    try:
        brands_df = _load_staged_brands_df(domain)

        third_key = config.get_third_tag_col(domain)
        third_tab = "Region" if domain == config.DOMAIN_FOOD else "Category"
        suffix = "Food" if domain == config.DOMAIN_FOOD else "Market"
        dicts: dict[str, list[str]] = {}
        for k, tab in (("gk", f"{suffix}_GK"), ("bt", f"{suffix}_BT"), (third_key, f"{suffix}_{third_tab}")):
            df_d = DataIngestion._fetch_sheet_as_csv("EVAL_LOCAL", tab, header=None, dtype=str)
            vals = df_d.iloc[:, 0].dropna().str.strip().tolist()
            # Deduplicate preserving order
            dicts[k] = list(dict.fromkeys(v for v in vals if v))
        return brands_df, dicts
    finally:
        config.STAGING_DIR = orig_staging
        config.CACHE_DIR = orig_cache


def _build_test_skus(test_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Builds leak-free incoming SKU dicts from test CSV (no target labels, no Brand/Flavor)."""
    skus: list[dict[str, Any]] = []
    for _, row in test_df.iterrows():
        desc = row.get("Description", "")
        cat = row.get("SellerCategory", "")
        p_val = clean_price(row.get("Price", 0.0))
        skus.append(
            {
                "name": str(row.get("Name", "") or "").strip(),
                "description": "" if pd.isna(desc) else str(desc).strip(),
                "category": "" if pd.isna(cat) else str(cat).strip(),
                "price": float(p_val) if p_val is not None else 0.0,
            }
        )
    return skus


class CachedNEREngine(NEREngine):
    """Wraps NEREngine with a persistent text -> entities dict so GLiNER runs only once per domain."""

    def __init__(self, base_ner: NEREngine, entity_cache: dict[str, dict[str, set[str]]], cache_path: Path):
        self.domain = base_ner.domain
        self.target_label = base_ner.target_label
        self.labels = base_ner.labels
        self.model = base_ner.model
        self.brand_mapping = base_ner.brand_mapping
        self.weak_brands = base_ner.weak_brands
        self.meat_flavors = base_ner.meat_flavors
        self.vegetable_flavors = base_ner.vegetable_flavors
        self.seafood_flavors = base_ner.seafood_flavors
        self.sorted_aliases = base_ner.sorted_aliases
        self.combined_pattern = base_ner.combined_pattern
        self._get_dict_entities = base_ner._get_dict_entities
        self._ner_cache = base_ner._ner_cache
        self._ner_cache_max = base_ner._ner_cache_max
        self._cache_lock = base_ner._cache_lock
        self._base_ner = base_ner
        self._entity_cache = entity_cache
        self._cache_path = cache_path
        self._dirty = False

    def extract_entities(self, text: str) -> dict[str, set[str]]:
        if text in self._entity_cache:
            return {k: set(v) for k, v in self._entity_cache[text].items()}
        res = self._base_ner.extract_entities(text)
        self._entity_cache[text] = {k: set(v) for k, v in res.items()}
        self._dirty = True
        return res

    def batch_extract_entities(self, texts: list[str], batch_size: int = 32) -> list[dict[str, set[str]]]:
        missing_idx = [i for i, t in enumerate(texts) if t not in self._entity_cache]
        if missing_idx:
            miss_texts = [texts[i] for i in missing_idx]
            computed = self._base_ner.batch_extract_entities(miss_texts, batch_size=batch_size)
            for t, ent in zip(miss_texts, computed):
                self._entity_cache[t] = {k: set(v) for k, v in ent.items()}
            self._dirty = True
        return [{k: set(v) for k, v in self._entity_cache[t].items()} for t in texts]

    def flush(self) -> None:
        if self._dirty:
            _atomic_write_pickle(self._cache_path, self._entity_cache)
            self._dirty = False


def ensure_stage_a_ner(
    domain: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    brands_df: pd.DataFrame,
    work_dir: Path,
) -> tuple[pd.DataFrame, CachedNEREngine]:
    """Extracts & caches GLiNER/alias entities once per domain across all evaluation sweeps."""
    ckpt_dir = work_dir / "checkpoints" / domain
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ner_cache_path = ckpt_dir / "ner_text_entities.pkl"
    train_ents_path = ckpt_dir / "train_catalog_entities.pkl"

    entity_cache: dict[str, dict[str, set[str]]] = (
        joblib.load(ner_cache_path) if ner_cache_path.exists() else {}
    )
    if not Path(config.GLINER_ONNX).exists():
        from scripts.sync import export_gliner

        export_gliner(force=False)
    shared_ner_model = getattr(resource_loader._ner_engine, "model", None)
    base_ner = NEREngine(brands_df, domain=domain, shared_model=shared_ner_model)
    if resource_loader._ner_engine is None:
        resource_loader._ner_engine = base_ner
    cached_ner = CachedNEREngine(base_ner, entity_cache, ner_cache_path)

    # 1. Train catalog entities (use pre-filled Flavor/Brand column when present, GLiNER only on missing)
    if train_ents_path.exists():
        train_entities = joblib.load(train_ents_path)
        if len(train_entities) == len(train_df):
            train_df["entities"] = pd.Series(train_entities, index=train_df.index, dtype=object)
        else:
            train_entities = None
    else:
        train_entities = None

    if train_entities is None:
        target_col = "Brand" if domain == config.DOMAIN_MARKET else "Flavor"
        target_label = cached_ner.target_label
        ent_list: list[dict[str, set[str]] | None] = []
        if target_col in train_df.columns:
            for raw_val in train_df[target_col].tolist():
                val = str(raw_val or "").strip()
                if val and val.lower() not in ("nan", "none", "<na>"):
                    ent_list.append({target_label: {v.strip().lower() for v in val.split(",") if v.strip()}})
                else:
                    ent_list.append(None)
        else:
            ent_list = [None] * len(train_df)

        miss_pos = [i for i, e in enumerate(ent_list) if e is None]
        if miss_pos:
            logger.info(f"[STAGE-A] [{domain.upper()}] Extracting NER for {len(miss_pos)} train rows missing {target_col}...")
            raw_names = train_df["Name"].tolist()
            ner_inputs = [TextPipeline.prep_for_ner(str(raw_names[i])) for i in miss_pos]
            res_ents = cached_ner.batch_extract_entities(ner_inputs, batch_size=config.ONNX_BATCH_SIZE)
            for pos, ent in zip(miss_pos, res_ents):
                ent_list[pos] = ent

        train_df["entities"] = pd.Series(ent_list, index=train_df.index, dtype=object)
        _atomic_write_pickle(train_ents_path, ent_list)

    # 2. Pre-warm test NER cache (exact strings used by processor.py & matcher.py)
    test_skus = _build_test_skus(test_df)
    ner_texts: list[str] = []
    for s in test_skus:
        ner_texts.append(
            TextPipeline.prep_for_ner(
                TextPipeline.build_ner_input(s["name"], s["description"], s["category"])
            )
        )
    missing_test = [t for t in set(ner_texts) if t not in entity_cache]
    if missing_test:
        logger.info(f"[STAGE-A] [{domain.upper()}] Pre-running GLiNER + alias NER on {len(missing_test)} unique test texts...")
        cached_ner.batch_extract_entities(missing_test, batch_size=config.ONNX_BATCH_SIZE)

    # Also warm food post-suffix strings if food domain
    if domain == config.DOMAIN_FOOD:
        extra_food_texts = []
        for s, nt in zip(test_skus, ner_texts):
            fset = cached_ner.extract_entities(nt).get("flavor", set())
            suffix = TextPipeline.resolve_food_mixed_suffix(s["name"], fset, cached_ner)
            eff_name = f"{s['name']} {suffix}" if suffix else s["name"]
            if suffix:
                extra_food_texts.append(
                    TextPipeline.prep_for_ner(
                        TextPipeline.build_ner_input(eff_name, s["description"], s["category"])
                    )
                )
        extra_missing = [t for t in set(extra_food_texts) if t not in entity_cache]
        if extra_missing:
            cached_ner.batch_extract_entities(extra_missing, batch_size=config.ONNX_BATCH_SIZE)

    cached_ner.flush()
    return train_df, cached_ner


def _setup_qdrant_client(qdrant_url: str | None, local_qdrant_dir: Path) -> None:
    from qdrant_client import QdrantClient

    if VectorStore._client is not None:
        try:
            VectorStore._client.close()
        except Exception:
            pass
    VectorStore._checked_collections.clear()
    if qdrant_url:
        config.QDRANT_URL = qdrant_url
        VectorStore._client = QdrantClient(
            url=qdrant_url,
            grpc_port=config.QDRANT_GRPC_PORT,
            prefer_grpc=config.QDRANT_PREFER_GRPC,
            timeout=config.QDRANT_TIMEOUT,
            check_compatibility=False,
        )
    else:
        local_qdrant_dir.mkdir(parents=True, exist_ok=True)
        VectorStore._client = QdrantClient(path=str(local_qdrant_dir))


def _load_rules_for_ablation(enabled: bool) -> None:
    if not enabled or not RULES_SNAPSHOT_PATH.exists():
        set_rules_from_dicts([], version="eval_empty")
        return
    snap = json.loads(RULES_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    set_rules_from_dicts(snap.get("rules", []), version=snap.get("version"))


def _bind_eval_singletons(
    domain: str,
    train_df: pd.DataFrame,
    brands_df: pd.DataFrame,
    dicts: dict[str, list[str]],
    embed_engine: EvalEmbeddingEngine,
    cached_ner: CachedNEREngine,
    classifier: TaxonomyClassifier,
    matcher: SKUMatcher,
) -> None:
    """Routes production resource_loader & DataIngestion lookups strictly to eval train-only objects."""
    DataIngestion._catalog_mem_cache[domain] = train_df
    DataIngestion._brands_mem_cache[domain] = brands_df
    DataIngestion.load_classifier_dictionaries = staticmethod(
        lambda *_a, **_kw: dicts
    )
    template_suggest._catalog_records_cache.pop(domain, None)

    classifier.ner_engine = cached_ner
    resource_loader._embed_engine = embed_engine
    resource_loader._ner_engine = cached_ner
    resource_loader._vector_store = matcher.vector_store
    resource_loader._classifiers[domain] = classifier
    resource_loader._pipelines[domain] = matcher
    resource_loader._domain_ner_engines[domain] = cached_ner
    resource_loader._model_statuses[domain] = {"pipeline": "ready", "classifier": "ready"}
    if domain == config.DOMAIN_MARKET and config.DOMAIN_FOOD not in resource_loader._classifiers:
        if (STAGED_FIXTURES_DIR / "Food_Flavors.csv").exists():
            from types import SimpleNamespace

            food_flavors_df = _load_staged_brands_df(config.DOMAIN_FOOD)
            food_ner = NEREngine(
                food_flavors_df,
                domain=config.DOMAIN_FOOD,
                shared_model=cached_ner.model,
            )
            resource_loader._classifiers[config.DOMAIN_FOOD] = SimpleNamespace(ner_engine=food_ner)
            resource_loader._domain_ner_engines[config.DOMAIN_FOOD] = food_ner
            resource_loader._model_statuses[config.DOMAIN_FOOD] = {"pipeline": "idle", "classifier": "ready"}
    resource_loader.get_classifier = lambda d, force_reset=False: resource_loader._classifiers[d]
    resource_loader.get_pipeline = lambda d, check_for_updates=False, force_sync=False: resource_loader._pipelines[d]
    resource_loader.get_ner_engine = lambda d: resource_loader._domain_ner_engines[d]


def _fit_eval_classifiers_and_arcface(
    domain: str,
    train_df: pd.DataFrame,
    brands_df: pd.DataFrame,
    dicts: dict[str, list[str]],
    embed_engine: EvalEmbeddingEngine,
    cached_ner: CachedNEREngine,
    cache_dir: Path,
    arcface_dir: Path,
    epochs: int = 25,
    arcface_batch_size: int = 64,
) -> tuple[TaxonomyClassifier, dict[str, Any]]:
    """Trains BT/Third/GK LogReg + WarmCentroids + 3-precision ArcFace strictly on train_df."""
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from torch.utils.data import DataLoader, TensorDataset

    from engine.classification.models.arcface_bt import BTArcFaceNet, FocalLoss
    from scripts.ops.colab_train_classifiers import (
        _fit_multiclass_logreg_gpu,
        _fit_ovr_logreg_gpu,
    )

    config.CACHE_DIR = cache_dir
    config.ARCFACE_DIR = arcface_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    arcface_dir.mkdir(parents=True, exist_ok=True)

    dicts_path = cache_dir / f"{domain}_classifier_dicts.json"
    dicts_path.write_text(json.dumps(dicts, indent=2), encoding="utf-8")

    cm = CacheManager(cached_ner, embed_engine)

    def _build_bt_gk(df: pd.DataFrame):
        bt_map = augment_bt_gk_map_with_training(df, {}, dicts.get("gk", []))
        umbrella = build_umbrella_from_training(df)
        third_map = build_bt_third_tag_map_from_catalog(df, domain)
        return bt_map, umbrella, third_map

    bt_gk_cache = cm.get_or_build_bt_gk_cache(train_df, domain, _build_bt_gk)
    descriptions = build_descriptions(
        domain,
        dicts,
        bt_gk_cache["bt_gk_map"],
        bt_gk_cache.get("third_tag_map", {}),
    )
    descriptions["bt_to_gk_umbrella"] = bt_gk_cache["umbrella"]

    # Temporarily force logreg so TaxonomyClassifier trains sklearn heads without failing on missing ArcFace yet
    orig_food_bt, orig_mkt_bt = config.FOOD_BT_MODEL, config.MARKET_BT_MODEL
    config.FOOD_BT_MODEL = "logreg"
    config.MARKET_BT_MODEL = "logreg"

    class EvalTaxonomyClassifier(TaxonomyClassifier):
        def _fit_classifiers(
            self,
            X,
            y_bt,
            y_third_tag,
            y_gk,
            target_col,
            old_bt_clf,
            old_bt_enc,
            old_third_tag_clf,
            old_third_tag_enc,
            LogisticRegression,
            OneVsRestClassifier,
        ):
            if torch.cuda.is_available():
                bt_clf = _fit_multiclass_logreg_gpu(
                    X, y_bt, C=5.0, balanced=False, desc=f"[EVAL-GPU] {domain} BT LogReg"
                )
                third_clf = _fit_multiclass_logreg_gpu(
                    X, y_third_tag, C=5.0, balanced=True, desc=f"[EVAL-GPU] {domain} {target_col} LogReg"
                )
                gk_clf = _fit_ovr_logreg_gpu(X, y_gk, C=5.0, desc=f"[EVAL-GPU] {domain} GK OvR")
                return bt_clf, third_clf, gk_clf
            return super()._fit_classifiers(
                X, y_bt, y_third_tag, y_gk, target_col,
                old_bt_clf, old_bt_enc, old_third_tag_clf, old_third_tag_enc,
                LogisticRegression, OneVsRestClassifier,
            )

    try:
        clf = EvalTaxonomyClassifier(
            embed_engine,
            domain,
            descriptions,
            cache_dir=cache_dir,
            cat_df=train_df,
            brands_df=brands_df,
            force_retrain=False,
        )
    finally:
        config.FOOD_BT_MODEL = orig_food_bt
        config.MARKET_BT_MODEL = orig_mkt_bt

    # Train ArcFace if not already exported for all 3 precisions
    labels_json_path = arcface_dir / f"{domain}_bt_arcface_labels.json"
    arc_paths = {
        "fp32": arcface_dir / f"{domain}_bt_arcface.onnx",
        "fp16": arcface_dir / f"{domain}_bt_arcface_fp16.onnx",
        "int8": arcface_dir / f"{domain}_bt_arcface_int8.onnx",
    }
    if not (labels_json_path.exists() and all(p.exists() for p in arc_paths.values())):
        df_af = train_df.fillna("")
        df_af = df_af[(df_af["basictype"].astype(str).str.strip() != "") & (df_af["Name"].astype(str).str.strip() != "")].reset_index(drop=True)
        cat_col = config.resolve_input_category_col(df_af.columns)
        names_l = df_af["Name"].astype(str).str.strip().tolist()
        descs_l = df_af["Description"].astype(str).str.strip().tolist() if "Description" in df_af.columns else [""] * len(df_af)
        cats_l = df_af[cat_col].astype(str).str.strip().tolist() if cat_col else [""] * len(df_af)

        X_text = clf._embed_weighted_sku_incremental(names_l, descs_l, cats_l).astype(np.float32)
        prices = np.clip(pd.to_numeric(df_af["Price"], errors="coerce").fillna(0.0).values.astype(np.float32).reshape(-1, 1), 0.0, None)
        scaler = StandardScaler()
        scaled_p = scaler.fit_transform(np.log1p(prices)).astype(np.float32)
        X_af = np.hstack([X_text, scaled_p]).astype(np.float32)

        le = LabelEncoder()
        y_af = le.fit_transform(df_af["basictype"].astype(str).str.strip().tolist())
        num_classes = len(le.classes_)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        af_net = BTArcFaceNet(in_features=X_af.shape[1], hidden_features=512, num_classes=num_classes).to(device)
        ds = TensorDataset(torch.from_numpy(X_af), torch.from_numpy(y_af).long())
        dl = DataLoader(ds, batch_size=arcface_batch_size, shuffle=True, drop_last=(len(ds) > arcface_batch_size))
        crit = FocalLoss(gamma=2.0, reduction="mean")
        opt = torch.optim.AdamW(af_net.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=1e-5)

        af_net.train()
        for _ep in range(1, epochs + 1):
            for bx, by in dl:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                loss = crit(af_net(bx, labels=by), by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(af_net.parameters(), max_norm=5.0)
                opt.step()
            sched.step()

        export_arcface_precisions(af_net, X_af, arcface_dir, domain)
        labels_meta = {
            "domain": domain,
            "num_classes": num_classes,
            "classes": le.classes_.tolist(),
            "scale": 30.0,
            "margin": 0.35,
            "input_dim": int(X_af.shape[1]),
            "hidden_dim": 512,
            "embedding_model": config.EMBEDDING_MODEL,
            "price_scaler": {
                "mean": float(scaler.mean_[0]),
                "scale": float(scaler.scale_[0]),
                "var": float(scaler.var_[0]),
            },
        }
        _atomic_write_json(labels_json_path, labels_meta)

    return clf, descriptions


def _set_classifier_head(clf: TaxonomyClassifier, domain: str, head: str, precision: str, arcface_dir: Path) -> None:
    """Switches an already-initialized TaxonomyClassifier cleanly between arcface and logreg."""
    import onnxruntime as ort
    from sklearn.preprocessing import StandardScaler

    clf.bt_model = head
    if head == "logreg":
        return

    labels_path = arcface_dir / f"{domain}_bt_arcface_labels.json"
    suffix_map = {"fp32": ".onnx", "fp16": "_fp16.onnx", "int8": "_int8.onnx"}
    onnx_path = arcface_dir / f"{domain}_bt_arcface{suffix_map[precision]}"
    meta = json.loads(labels_path.read_text(encoding="utf-8"))
    clf._arcface_classes = meta["classes"]
    sc_info = meta["price_scaler"]
    sc = StandardScaler()
    sc.mean_ = np.array([sc_info["mean"]], dtype=np.float32)
    sc.scale_ = np.array([sc_info["scale"]], dtype=np.float32)
    sc.var_ = np.array([sc_info["var"]], dtype=np.float32)
    clf._arcface_price_scaler = sc

    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False
    clf._arcface_session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    clf._arcface_input_name = clf._arcface_session.get_inputs()[0].name


def _sync_eval_qdrant(
    domain: str,
    train_df: pd.DataFrame,
    brands_df: pd.DataFrame,
    dicts: dict[str, list[str]],
    embed_engine: EvalEmbeddingEngine,
    cached_ner: CachedNEREngine,
    cache_dir: Path,
    emb_model: str,
    emb_prec: str,
) -> tuple[SKUMatcher, VectorStore]:
    """Populates eval Qdrant collections (catalog + tags) for current (domain, embedding, precision)."""
    vs = VectorStore()
    cat_col = vs._get_collection_name(domain)
    tag_col = vs._get_tags_collection_name(domain)
    marker_path = cache_dir / "eval_qdrant_synced.json"
    expected_marker = {"domain": domain, "embedding_model": emb_model, "precision": emb_prec, "rows": len(train_df)}

    need_rebuild = True
    if marker_path.exists():
        try:
            prev = json.loads(marker_path.read_text(encoding="utf-8"))
            if prev == expected_marker and vs.client.collection_exists(cat_col) and vs.client.collection_exists(tag_col):
                if vs.client.count(cat_col).count >= len(train_df):
                    need_rebuild = False
        except Exception:
            need_rebuild = True

    if need_rebuild:
        for col in (cat_col, tag_col):
            vs.delete_named_collection(col)
        hashes_f = cache_dir / f"{domain}_row_hashes.json"
        if hashes_f.exists():
            hashes_f.unlink()
        # Pre-seed catalog_metadata.pkl with train entities & NLP cols so CacheManager avoids re-running NER
        _atomic_write_pickle(cache_dir / f"{domain}_catalog_metadata.pkl", train_df)

        third_col = config.get_third_tag_col(domain)
        for tag_key, dict_type in (("gk", "gk"), ("bt", "bt"), (third_col, third_col)):
            tag_list = dicts.get(tag_key, [])
            if tag_list:
                embs = embed_engine.embed_dictionary_incremental(domain, dict_type, tag_list)
                vs.upsert_tags(
                    tags=tag_list,
                    dense_embs=embs["dense"],
                    sparse_embs=embs.get("sparse") or None,
                    bm25_embs=embs.get("bm25") or None,
                    dict_type=dict_type,
                    domain=domain,
                    force=True,
                )

    cm = CacheManager(cached_ner, embed_engine)
    lg = LogicGates(embed_engine, brands_df=brands_df)
    matcher = SKUMatcher(
        train_df,
        brands_df,
        cached_ner,
        embed_engine,
        cm,
        lg,
        domain=domain,
        classifier=None,
        check_for_updates=need_rebuild,
        force_sync=need_rebuild,
    )
    if need_rebuild:
        _atomic_write_json(marker_path, expected_marker)
    return matcher, vs


def _evaluate_standalone_embeddings_and_heads(
    domain: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_skus: list[dict[str, Any]],
    clf: TaxonomyClassifier,
    embed_engine: EvalEmbeddingEngine,
    vs: VectorStore,
    cache_dir: Path,
    arcface_dir: Path,
    emb_prec: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Computes standalone kNN/centroid, head Top-1/3/5, and pre-rerank retrieval metrics."""
    third_csv = "Region" if domain == config.DOMAIN_FOOD else "Category"
    third_col = config.get_third_tag_col(domain)
    gt_bt = test_df["BasicType"].fillna("").astype(str).str.strip().tolist()
    gt_third = test_df[third_csv].fillna("").astype(str).str.strip().tolist()
    gt_gks = [M.parse_gk_set(v) for v in test_df["GenericKeywords"].tolist()]
    train_names_lower = set(train_df["Name"].astype(str).str.strip().str.lower())
    seen_name_mask = [s["name"].strip().lower() in train_names_lower for s in test_skus]
    train_bt_counts = train_df["basictype"].astype(str).str.strip().value_counts().to_dict()

    # Encode test with CLASSIFIER_WEIGHTS and MATCHER_WEIGHTS
    names = [s["name"] for s in test_skus]
    descs = [s["description"] for s in test_skus]
    cats = [s["category"] for s in test_skus]
    prices = [s["price"] for s in test_skus]

    test_clf_emb = embed_engine.embed_weighted_sku(names, descs, cats, weights=config.CLASSIFIER_WEIGHTS)
    clean_names = [TextPipeline.normalize_final(TextPipeline.standardize_units(n)) for n in names]
    test_match_emb = embed_engine.embed_weighted_sku(clean_names, descs, cats, weights=config.MATCHER_WEIGHTS)

    # Load train weighted vectors from cache
    sku_cache = joblib.load(cache_dir / f"{domain}_weighted_skus_cache.pkl")
    sc_col = config.resolve_input_category_col(train_df.columns)
    tr_names = train_df["Name"].astype(str).str.strip().tolist()
    tr_descs = train_df["Description"].fillna("").astype(str).str.strip().tolist() if "Description" in train_df.columns else [""] * len(train_df)
    tr_cats = train_df[sc_col].fillna("").astype(str).str.strip().tolist() if sc_col else [""] * len(train_df)
    X_train = np.vstack([sku_cache[f"{n}||{d}||{c}"] for n, d, c in zip(tr_names, tr_descs, tr_cats)]).astype(np.float32)

    tr_bts = train_df["basictype"].astype(str).str.strip().tolist()
    tr_thirds = train_df[third_col].astype(str).str.strip().tolist()

    # 1. Standalone Embedding k-NN (1-NN & 15-NN)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_t = torch.from_numpy(X_train).to(device)
    te_t = torch.from_numpy(test_clf_emb["dense"].astype(np.float32)).to(device)
    knn1_bt, knn15_bt, knn15_topk_bt, knn15_conf, knn15_third = [], [], [], [], []
    with torch.inference_mode():
        for start in range(0, len(te_t), 256):
            sims = te_t[start : start + 256] @ tr_t.T
            top_vals, top_idx = torch.topk(sims, k=min(15, len(tr_t)), dim=1)
            tv_np = top_vals.cpu().numpy()
            ti_np = top_idx.cpu().numpy()
            for r_vals, r_idxs in zip(tv_np, ti_np):
                knn1_bt.append(tr_bts[r_idxs[0]])
                bt_scores: dict[str, float] = {}
                th_scores: dict[str, float] = {}
                tot_s = 0.0
                for sv, tidx in zip(r_vals, r_idxs):
                    w = max(0.0, float(sv))
                    tot_s += w
                    b = tr_bts[tidx]
                    t = tr_thirds[tidx]
                    bt_scores[b] = bt_scores.get(b, 0.0) + w
                    th_scores[t] = th_scores.get(t, 0.0) + w
                ranked_bts = [k for k, _ in sorted(bt_scores.items(), key=lambda x: -x[1])]
                knn15_bt.append(ranked_bts[0] if ranked_bts else "")
                knn15_topk_bt.append(ranked_bts[:5])
                knn15_conf.append((bt_scores[ranked_bts[0]] / tot_s) if (ranked_bts and tot_s > 0) else 0.0)
                knn15_third.append(max(th_scores.items(), key=lambda x: x[1])[0] if th_scores else "")
    del tr_t, te_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Warm Centroids standalone
    centroid_metrics: dict[str, Any] = {}
    if clf.cold_start_router and clf.cold_start_router.registry.warm_centroids.shape[0] > 0:
        protos = clf.cold_start_router.registry.warm_centroids
        proto_classes = clf.cold_start_router.registry.warm_index_to_class
        probas_c, best_idx_c, conf_c = embed_engine.predict_centroids_temperature(
            test_clf_emb["dense"], protos, tau=config.COLD_START_TAU
        )
        top5_c = np.argsort(-probas_c, axis=1)[:, :5]
        c_preds = [proto_classes[i] for i in best_idx_c]
        c_topk = [[proto_classes[j] for j in row] for row in top5_c]
        centroid_metrics = M.compute_multiclass_metrics(
            gt_bt, c_preds, confidences=conf_c.tolist(), topk_preds=c_topk,
            train_class_counts=train_bt_counts, seen_sku_mask=seen_name_mask,
        )

    # 3. Standalone Catalog Retrieval Ablation (Dense vs BM25 vs Sparse vs Hybrid RRF)
    from qdrant_client.http.models import (
        FieldCondition,
        Filter,
        Fusion,
        FusionQuery,
        MatchValue,
        QueryRequest,
    )

    cat_col = vs._get_collection_name(domain)
    train_bt_norm_counts = Counter_dict(train_df["basictype"].astype(str).str.strip().str.lower().tolist())
    gt_bt_norm = [M.norm_label(x) for x in gt_bt]
    gt_third_norm = [M.norm_label(x) for x in gt_third]

    retrieval_modes = ["hybrid_rrf", "dense_only", "bm25_only"]
    if embed_engine.has_learned_sparse:
        retrieval_modes.append("sparse_only")

    cat_retrieval_metrics: dict[str, Any] = {}
    n_test = len(test_skus)
    for rmode in retrieval_modes:
        reqs = []
        for i in range(n_test):
            d_list = test_match_emb["dense"][i].tolist()
            bm25_sv = vs._parse_sparse_dict(test_match_emb["bm25"][i])
            sp_list = test_match_emb.get("sparse") or []
            sp_sv = vs._parse_sparse_dict(sp_list[i]) if (sp_list and i < len(sp_list)) else None

            if rmode == "hybrid_rrf":
                reqs.append(
                    QueryRequest(
                        prefetch=vs._build_hybrid_prefetch(
                            test_match_emb["dense"][i], test_match_emb["bm25"][i],
                            sp_list[i] if sp_list else None, limit=30,
                        ),
                        query=FusionQuery(fusion=Fusion.RRF),
                        limit=30,
                        with_payload=vs._SEARCH_PAYLOAD,
                    )
                )
            elif rmode == "dense_only":
                reqs.append(QueryRequest(query=d_list, using="dense", limit=30, with_payload=vs._SEARCH_PAYLOAD))
            elif rmode == "bm25_only":
                if bm25_sv.indices:
                    reqs.append(QueryRequest(query=bm25_sv, using="bm25", limit=30, with_payload=vs._SEARCH_PAYLOAD))
                else:
                    reqs.append(QueryRequest(query=d_list, using="dense", limit=1, with_payload=vs._SEARCH_PAYLOAD))
            else:
                if sp_sv and sp_sv.indices:
                    reqs.append(QueryRequest(query=sp_sv, using="sparse", limit=30, with_payload=vs._SEARCH_PAYLOAD))
                else:
                    reqs.append(QueryRequest(query=d_list, using="dense", limit=1, with_payload=vs._SEARCH_PAYLOAD))

        all_hits = []
        for s_i in range(0, len(reqs), config.QDRANT_BATCH_CHUNK_SIZE):
            res_b = vs._call_with_retry(
                f"eval_ret_{rmode}", vs.client.query_batch_points, collection_name=cat_col, requests=reqs[s_i : s_i + config.QDRANT_BATCH_CHUNK_SIZE]
            )
            for r in res_b:
                all_hits.append([vs._hit_to_dict(pt) for pt in r.points])

        b_rels, g_rels, tot_rels, ideals = [], [], [], []
        for i, hits in enumerate(all_hits):
            tot_bt_i = train_bt_norm_counts.get(gt_bt_norm[i], 0)
            tot_rels.append(tot_bt_i)
            b_row, g_row = [], []
            for h in hits:
                b, g = M.catalog_candidate_relevance(gt_bt_norm[i], gt_third_norm[i], gt_gks[i], h, third_col)
                b_row.append(b)
                g_row.append(g)
            b_rels.append(b_row)
            g_rels.append(g_row)
            # Ideal graded ranking given train BT counts
            ideals.append([3.0] * min(30, tot_bt_i) + [0.0] * max(0, 30 - tot_bt_i))

        cat_retrieval_metrics[rmode] = M.compute_ranking_metrics(
            b_rels, tot_rels, graded_rels_per_query=g_rels, ideal_graded_per_query=ideals
        )

    # 4. GK Tag Retrieval (eval_{domain}_tags)
    tag_col = vs._get_tags_collection_name(domain)
    tag_reqs = []
    gk_filter = Filter(must=[FieldCondition(key="dict_type", match=MatchValue(value="gk"))])
    for i in range(n_test):
        sp_list = test_clf_emb.get("sparse") or []
        tag_reqs.append(
            QueryRequest(
                prefetch=vs._build_hybrid_prefetch(
                    test_clf_emb["dense"][i], test_clf_emb["bm25"][i],
                    sp_list[i] if sp_list else None, limit=30,
                ),
                query=FusionQuery(fusion=Fusion.RRF),
                filter=gk_filter,
                limit=30,
                with_payload=["tag"],
            )
        )
    tag_hits_all = []
    for s_i in range(0, len(tag_reqs), config.QDRANT_BATCH_CHUNK_SIZE):
        res_b = vs._call_with_retry(
            "eval_tag_ret", vs.client.query_batch_points, collection_name=tag_col, requests=tag_reqs[s_i : s_i + config.QDRANT_BATCH_CHUNK_SIZE]
        )
        for r in res_b:
            tag_hits_all.append([str((pt.payload or {}).get("tag", "")).strip().lower() for pt in r.points])

    tb_rels, t_tots, t_ideals = [], [], []
    for i, tags_ranked in enumerate(tag_hits_all):
        gs = gt_gks[i]
        t_tots.append(len(gs))
        tb_rels.append([1 if t in gs else 0 for t in tags_ranked])
        t_ideals.append([1.0] * min(30, len(gs)) + [0.0] * max(0, 30 - len(gs)))
    gk_tag_ret_metrics = M.compute_ranking_metrics(
        tb_rels, t_tots, ideal_graded_per_query=t_ideals
    )

    emb_standalone = {
        "knn1_bt": M.compute_multiclass_metrics(gt_bt, knn1_bt, train_class_counts=train_bt_counts, seen_sku_mask=seen_name_mask),
        "knn15_bt": M.compute_multiclass_metrics(gt_bt, knn15_bt, confidences=knn15_conf, topk_preds=knn15_topk_bt, train_class_counts=train_bt_counts, seen_sku_mask=seen_name_mask),
        "knn15_third_tag": M.compute_multiclass_metrics(gt_third, knn15_third, seen_sku_mask=seen_name_mask),
        "warm_centroids_bt": centroid_metrics,
        "catalog_retrieval": cat_retrieval_metrics,
        "gk_tag_retrieval": gk_tag_ret_metrics,
    }

    # 5. Standalone Head Metrics (ArcFace vs LogReg Top-1/3/5 + Third-Tag)
    head_standalone: dict[str, Any] = {}
    # LogReg BT + Third-Tag
    scaled_p_lr = clf._preprocess_prices(prices, is_training=False)
    X_te_lr = np.hstack([test_clf_emb["dense"], scaled_p_lr])
    lr_bt_prob = clf._bt_clf.predict_proba(X_te_lr)
    lr_bt_top5 = np.argsort(-lr_bt_prob, axis=1)[:, :5]
    lr_bt_preds = [clf._bt_enc.classes_[r[0]] for r in lr_bt_top5]
    lr_bt_topk = [[clf._bt_enc.classes_[j] for j in r] for r in lr_bt_top5]
    lr_bt_conf = np.max(lr_bt_prob, axis=1).tolist()

    lr_th_prob = clf._third_tag_clf.predict_proba(X_te_lr)
    lr_th_top5 = np.argsort(-lr_th_prob, axis=1)[:, :5]
    lr_th_preds = [clf._third_tag_enc.classes_[r[0]] for r in lr_th_top5]
    lr_th_topk = [[clf._third_tag_enc.classes_[j] for j in r] for r in lr_th_top5]
    lr_th_conf = np.max(lr_th_prob, axis=1).tolist()

    head_standalone["logreg"] = {
        "bt": M.compute_multiclass_metrics(
            gt_bt, lr_bt_preds, confidences=lr_bt_conf, topk_preds=lr_bt_topk,
            train_class_counts=train_bt_counts, seen_sku_mask=seen_name_mask,
        ),
        "third_tag": M.compute_multiclass_metrics(
            gt_third, lr_th_preds, confidences=lr_th_conf, topk_preds=lr_th_topk, seen_sku_mask=seen_name_mask
        ),
    }

    # ArcFace BT at current emb_prec
    _set_classifier_head(clf, domain, "arcface", emb_prec, arcface_dir)
    af_scaled_p = clf._preprocess_arcface_prices(prices)
    X_te_af = np.hstack([test_clf_emb["dense"], af_scaled_p]).astype(np.float32)
    af_logits_parts = []
    for s_i in range(0, len(X_te_af), config.ONNX_BATCH_SIZE):
        af_logits_parts.append(
            clf._arcface_session.run(None, {clf._arcface_input_name: X_te_af[s_i : s_i + config.ONNX_BATCH_SIZE]})[0]
        )
    af_logits = np.concatenate(af_logits_parts, axis=0)
    exp_l = np.exp(af_logits - np.max(af_logits, axis=-1, keepdims=True))
    af_prob = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
    af_top5 = np.argsort(-af_prob, axis=1)[:, :5]
    af_preds = [clf._arcface_classes[r[0]] for r in af_top5]
    af_topk = [[clf._arcface_classes[j] for j in r] for r in af_top5]
    af_conf = np.max(af_prob, axis=1).tolist()
    head_standalone["arcface"] = {
        "bt": M.compute_multiclass_metrics(
            gt_bt, af_preds, confidences=af_conf, topk_preds=af_topk,
            train_class_counts=train_bt_counts, seen_sku_mask=seen_name_mask,
        ),
        "third_tag": head_standalone["logreg"]["third_tag"],
    }

    encoded_bundle = {
        "test_clf_emb": test_clf_emb,
        "test_match_emb": test_match_emb,
    }
    return emb_standalone, {"head_standalone": head_standalone, **encoded_bundle}


def Counter_dict(items: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for x in items:
        counts[x] = counts.get(x, 0) + 1
    return counts


class RecordingVectorStore(VectorStore):
    """Caches & replays Qdrant searches by query key so Stage C never hits Qdrant redundantly."""

    def __init__(self, base_vs: VectorStore, cache_store: dict[str, Any]):
        self.client = base_vs.client
        self._cache = cache_store
        self._dirty = False

    @staticmethod
    def _hash_vec(arr: np.ndarray) -> str:
        return hashlib.md5(np.asarray(arr, dtype=np.float32).tobytes()).hexdigest()[:16]

    def search_batch_from_vectors(
        self,
        dense_vecs: list[np.ndarray],
        sparse_vec_dicts: list[dict[str, float]] | None = None,
        bt_filters: list[str | None] | None = None,
        domain: str = config.DOMAIN_MARKET,
        top_k: int = config.TOP_K_RETRIEVAL,
        bm25_vec_dicts: list[dict[str, float]] | None = None,
    ) -> list[list[dict[str, Any]]]:
        n = len(dense_vecs)
        bt_filters = bt_filters or [None] * n
        results: list[list[dict[str, Any]] | None] = [None] * n
        miss_indices: list[int] = []
        keys: list[str] = []
        cat_cache = self._cache.setdefault("catalog", {})
        for i in range(n):
            k = f"{domain}|{top_k}|{bt_filters[i] or ''}|{self._hash_vec(dense_vecs[i])}"
            keys.append(k)
            if k in cat_cache:
                results[i] = cat_cache[k]
            else:
                miss_indices.append(i)
        if miss_indices:
            fetched = super().search_batch_from_vectors(
                [dense_vecs[i] for i in miss_indices],
                [sparse_vec_dicts[i] for i in miss_indices] if sparse_vec_dicts else None,
                [bt_filters[i] for i in miss_indices],
                domain=domain,
                top_k=top_k,
                bm25_vec_dicts=[bm25_vec_dicts[i] for i in miss_indices] if bm25_vec_dicts else None,
            )
            for pos, orig_i in enumerate(miss_indices):
                cat_cache[keys[orig_i]] = fetched[pos]
                results[orig_i] = fetched[pos]
            self._dirty = True
        return [r if r is not None else [] for r in results]

    def search_batch_hybrid_tags(
        self,
        dense_queries: list[np.ndarray],
        sparse_queries: list[dict[str, float]] | None = None,
        filter_dict_type: str | None = None,
        domain: str = config.DOMAIN_MARKET,
        allowed_tags_list: list[list[str] | None] | None = None,
        limit: int = config.TAG_SEARCH_LIMIT,
        bm25_queries: list[dict[str, float]] | None = None,
    ):
        n = len(dense_queries)
        allowed_tags_list = allowed_tags_list or [None] * n
        results: list[Any] = [None] * n
        miss_indices: list[int] = []
        keys: list[str] = []
        tag_cache = self._cache.setdefault("tags", {})
        for i in range(n):
            allow_sig = hashlib.md5(",".join(sorted(allowed_tags_list[i] or [])).encode()).hexdigest()[:12]
            k = f"{domain}|{filter_dict_type}|{limit}|{allow_sig}|{self._hash_vec(dense_queries[i])}"
            keys.append(k)
            if k in tag_cache:
                results[i] = tag_cache[k]
            else:
                miss_indices.append(i)
        if miss_indices:
            fetched = super().search_batch_hybrid_tags(
                [dense_queries[i] for i in miss_indices],
                [sparse_queries[i] for i in miss_indices] if sparse_queries else None,
                filter_dict_type=filter_dict_type,
                domain=domain,
                allowed_tags_list=[allowed_tags_list[i] for i in miss_indices],
                limit=limit,
                bm25_queries=[bm25_queries[i] for i in miss_indices] if bm25_queries else None,
            )
            for pos, orig_i in enumerate(miss_indices):
                tag_cache[keys[orig_i]] = fetched[pos]
                results[orig_i] = fetched[pos]
            self._dirty = True
        return results

    def search_batch_catalog_neighbors(
        self,
        dense_vecs: list[np.ndarray],
        domain: str = config.DOMAIN_MARKET,
        top_k: int = 15,
    ) -> list[list[dict[str, Any]]]:
        n = len(dense_vecs)
        results: list[list[dict[str, Any]] | None] = [None] * n
        miss_indices: list[int] = []
        keys: list[str] = []
        nb_cache = self._cache.setdefault("neighbors", {})
        for i in range(n):
            k = f"{domain}|{top_k}|{self._hash_vec(dense_vecs[i])}"
            keys.append(k)
            if k in nb_cache:
                results[i] = nb_cache[k]
            else:
                miss_indices.append(i)
        if miss_indices:
            fetched = super().search_batch_catalog_neighbors(
                [dense_vecs[i] for i in miss_indices], domain=domain, top_k=top_k
            )
            for pos, orig_i in enumerate(miss_indices):
                nb_cache[keys[orig_i]] = fetched[pos]
                results[orig_i] = fetched[pos]
            self._dirty = True
        return [r if r is not None else [] for r in results]


class CachedRerankAdapter:
    """Wraps EvalEmbeddingEngine so identical query-doc rerank pairs are scored once & cached."""

    def __init__(
        self,
        emb_engine: EvalEmbeddingEngine | None,
        rerank_engine: EvalEmbeddingEngine | None,
        pair_score_cache: dict[tuple[str, str], float],
        query_emb_cache: dict[str, dict[str, Any]] | None = None,
        pair_ckpt_path: Path | None = None,
        flush_every: int = 5000,
    ):
        self._emb = emb_engine
        self._rerank = rerank_engine
        self._pair_cache = pair_score_cache
        self._query_emb_cache = query_emb_cache or {}
        self._pair_ckpt_path = pair_ckpt_path
        self._flush_every = int(flush_every)
        self.cross_session = getattr(rerank_engine, "cross_session", None)
        self.has_learned_sparse = getattr(emb_engine, "has_learned_sparse", False)
        self.embedding_dim = getattr(emb_engine, "embedding_dim", config.EMBEDDING_DIM)
        self.rerank_time_s = 0.0
        self.rerank_pairs_scored = 0

    def __getattr__(self, item: str) -> Any:
        if self._emb is not None and hasattr(self._emb, item):
            return getattr(self._emb, item)
        if self._rerank is not None and hasattr(self._rerank, item):
            return getattr(self._rerank, item)
        raise AttributeError(item)

    def embed_weighted_sku(
        self, names: list[str], descriptions: list[str], categories: list[str], weights=config.CLASSIFIER_WEIGHTS
    ) -> dict[str, Any]:
        # Fast lookup when exact full batch matches cached Stage B test embeddings
        cache_sig = hashlib.md5(
            f"{weights}||{'|'.join(names)}||{'|'.join(descriptions)}||{'|'.join(categories)}".encode()
        ).hexdigest()
        if cache_sig in self._query_emb_cache:
            return self._query_emb_cache[cache_sig]
        assert self._emb is not None
        res = self._emb.embed_weighted_sku(names, descriptions, categories, weights=weights)
        self._query_emb_cache[cache_sig] = res
        return res

    def score_cross_encoder(self, pairs: list[list[str]], progress_callback=None) -> np.ndarray:
        if not pairs:
            return np.array([], dtype=np.float32)
        out = np.empty(len(pairs), dtype=np.float32)
        missing_pos: list[int] = []
        missing_pairs: list[list[str]] = []
        seen_missing: dict[tuple[str, str], list[int]] = {}

        for idx, p in enumerate(pairs):
            key = (str(p[0]), str(p[1]))
            if key in self._pair_cache:
                out[idx] = self._pair_cache[key]
            else:
                if key not in seen_missing:
                    seen_missing[key] = [idx]
                    missing_pos.append(len(missing_pairs))
                    missing_pairs.append([key[0], key[1]])
                else:
                    seen_missing[key].append(idx)

        if missing_pairs:
            assert self._rerank is not None
            step = self._flush_every if (self._pair_ckpt_path and self._flush_every > 0) else len(missing_pairs)
            t0 = time.perf_counter()
            total_miss = len(missing_pairs)
            for start in range(0, total_miss, step):
                chunk = missing_pairs[start : start + step]
                cb = (
                    (lambda done, _tot, off=start, tot=total_miss: progress_callback(off + done, tot))
                    if progress_callback
                    else None
                )
                scores = self._rerank.score_cross_encoder(chunk, progress_callback=cb)
                for m_i, pair in enumerate(chunk):
                    key = (pair[0], pair[1])
                    sc = float(scores[m_i])
                    self._pair_cache[key] = sc
                    for orig_idx in seen_missing[key]:
                        out[orig_idx] = sc
                if self._pair_ckpt_path is not None and (start + step < total_miss):
                    _atomic_write_pickle(self._pair_ckpt_path, self._pair_cache)
            self.rerank_time_s += time.perf_counter() - t0
            self.rerank_pairs_scored += len(missing_pairs)

        if progress_callback:
            progress_callback(len(pairs), len(pairs))
        return out


def _compute_reranker_diagnostics_for_combo(
    domain: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_skus: list[dict[str, Any]],
    clf: TaxonomyClassifier,
    matcher: SKUMatcher,
    rec_vs: RecordingVectorStore,
    rerank_adapter: CachedRerankAdapter,
    cached_ner: CachedNEREngine,
) -> dict[str, Any]:
    """Evaluates pre- vs post-reranker ranking lift on catalog SKU & GK tag candidate pools."""
    third_col = config.get_third_tag_col(domain)
    third_csv = "Region" if domain == config.DOMAIN_FOOD else "Category"
    gt_bt_norm = [M.norm_label(x) for x in test_df["BasicType"].tolist()]
    gt_third_norm = [M.norm_label(x) for x in test_df[third_csv].tolist()]
    gt_gks = [M.parse_gk_set(v) for v in test_df["GenericKeywords"].tolist()]
    train_bt_counts = Counter_dict(train_df["basictype"].astype(str).str.strip().str.lower().tolist())

    # 1. Catalog reranking pool diagnostics (non-exact/non-fuzzy SKUs)
    names = [s["name"] for s in test_skus]
    descs = [s["description"] for s in test_skus]
    cats = [s["category"] for s in test_skus]
    clean_names = [TextPipeline.normalize_final(TextPipeline.standardize_units(n)) for n in names]
    emb_m = rerank_adapter.embed_weighted_sku(clean_names, descs, cats, weights=config.MATCHER_WEIGHTS)
    bt_preds = clf.batch_predict_bt(
        emb_m["dense"], [s["price"] for s in test_skus], sku_names=names, sku_descriptions=descs
    )
    bt_filters = [
        p[0] if (p and p[1] >= config.get_bt_confidence_threshold(p[2])) else None for p in bt_preds
    ]

    non_bypass_idx = [
        i
        for i, cn in enumerate(clean_names)
        if cn not in matcher.exact_match_map
        and " ".join(sorted(TextPipeline.strip_weights(cn).split())) not in matcher.token_sorted_map
    ]

    cat_diag: dict[str, Any] = {}
    if non_bypass_idx:
        q_dense = [emb_m["dense"][i] for i in non_bypass_idx]
        q_bm25 = [emb_m["bm25"][i] for i in non_bypass_idx]
        q_sp = [emb_m["sparse"][i] for i in non_bypass_idx] if emb_m.get("sparse") else None
        q_bt = [bt_filters[i] for i in non_bypass_idx]

        # Match matcher.py pool assembly: filtered + unfiltered
        hits_filt = rec_vs.search_batch_from_vectors(
            [q_dense[k] for k in range(len(non_bypass_idx)) if q_bt[k]],
            [q_sp[k] for k in range(len(non_bypass_idx)) if q_bt[k]] if q_sp else None,
            [q_bt[k] for k in range(len(non_bypass_idx)) if q_bt[k]],
            domain=domain,
            bm25_vec_dicts=[q_bm25[k] for k in range(len(non_bypass_idx)) if q_bt[k]],
        )
        hits_unf = rec_vs.search_batch_from_vectors(
            q_dense, q_sp, [None] * len(non_bypass_idx), domain=domain, bm25_vec_dicts=q_bm25
        )
        filt_map = {}
        f_ptr = 0
        for k in range(len(non_bypass_idx)):
            if q_bt[k]:
                filt_map[k] = hits_filt[f_ptr]
                f_ptr += 1

        ner_texts = [
            TextPipeline.prep_for_ner(TextPipeline.build_ner_input(names[i], descs[i], cats[i]))
            for i in non_bypass_idx
        ]
        ents_list = cached_ner.batch_extract_entities(ner_texts)

        all_pairs = []
        pool_per_q: list[tuple[int, list[dict[str, Any]], int, int]] = []
        for k, orig_i in enumerate(non_bypass_idx):
            merged = list(filt_map.get(k, [])) + list(hits_unf[k])
            merged.sort(key=lambda c: -round(float(c.get("_qdrant_score_", 0.0) or 0.0), 6))
            dedup = []
            seen_fmt = set()
            for cand in merged:
                fmt = TextPipeline.format_catalog_row_for_reranker(cand, domain=domain)
                if fmt not in seen_fmt:
                    seen_fmt.add(fmt)
                    dedup.append(cand)
                if len(dedup) >= config.MATCHER_RERANK_CANDIDATES:
                    break
            in_bf_key = "flavor" if domain == config.DOMAIN_FOOD else "brand"
            bf = ", ".join(sorted(str(x) for x in ents_list[k].get(in_bf_key, []) if x))
            fq = TextPipeline.format_matcher_rerank_text(
                name=TextPipeline.strip_weights(clean_names[orig_i]),
                seller_category=cats[orig_i],
                brand_or_flavor=bf,
                description=descs[orig_i],
                domain=domain,
            )
            start_p = len(all_pairs)
            for cand in dedup:
                all_pairs.append([fq, TextPipeline.format_catalog_row_for_reranker(cand, domain=domain)])
            pool_per_q.append((orig_i, dedup, start_p, len(all_pairs)))

        scores_flat = rerank_adapter.score_cross_encoder(all_pairs)
        pre_b, post_b, pre_g, post_g, tot_r, ideals = [], [], [], [], [], []
        pair_y, pair_s, top1_s, top1_ok = [], [], [], []
        for orig_i, dedup, p0, p1 in pool_per_q:
            q_scores = scores_flat[p0:p1]
            tot_bt_i = train_bt_counts.get(gt_bt_norm[orig_i], 0)
            tot_r.append(tot_bt_i)
            ideals.append([3.0] * min(30, tot_bt_i) + [0.0] * max(0, 30 - tot_bt_i))

            pre_rows = []
            for idx_c, cand in enumerate(dedup):
                b, g = M.catalog_candidate_relevance(
                    gt_bt_norm[orig_i], gt_third_norm[orig_i], gt_gks[orig_i], cand, third_col
                )
                sc = float(q_scores[idx_c]) if idx_c < len(q_scores) else -10.0
                pre_rows.append((b, g, sc))
                pair_y.append(b)
                pair_s.append(sc)
            post_rows = sorted(pre_rows, key=lambda x: -x[2])

            pre_b.append([x[0] for x in pre_rows])
            pre_g.append([x[1] for x in pre_rows])
            post_b.append([x[0] for x in post_rows])
            post_g.append([x[1] for x in post_rows])
            if post_rows:
                top1_s.append(post_rows[0][2])
                top1_ok.append(post_rows[0][0])

        cat_diag = M.compute_reranker_diagnostics(
            pre_b, post_b, tot_r, pair_y, pair_s,
            pre_graded_rels=pre_g, post_graded_rels=post_g, ideal_graded_per_query=ideals,
            top1_scores=top1_s, top1_correct=top1_ok,
        )

    # 2. GK tag reranking diagnostics (over all non-null GK test rows)
    emb_c = rerank_adapter.embed_weighted_sku(names, descs, cats, weights=config.CLASSIFIER_WEIGHTS)
    bt_preds_c = clf.batch_predict_bt(emb_c["dense"], [s["price"] for s in test_skus], sku_names=names, sku_descriptions=descs)
    strategy = get_strategy(domain)
    bt_gk_map = getattr(clf, "bt_gk_map", {})

    tag_hits = rec_vs.search_batch_hybrid_tags(
        list(emb_c["dense"]),
        list(emb_c["sparse"]) if emb_c.get("sparse") else None,
        filter_dict_type="gk",
        domain=domain,
        allowed_tags_list=[bt_gk_map.get(p[0], []) or None for p in bt_preds_c],
        limit=config.TAG_SEARCH_LIMIT,
        bm25_queries=list(emb_c["bm25"]),
    )
    from engine.classification.tagger import _rrf_fusion, _weighted_fusion

    tag_pairs = []
    tag_pools = []
    for i in range(len(test_skus)):
        if not gt_gks[i]:
            continue
        d_h, b_h, s_h = tag_hits[i]
        fused = _rrf_fusion(d_h, b_h, s_h) if config.FUSION_METHOD == "rrf" else _weighted_fusion(d_h, b_h, s_h)
        top_c = fused[: config.TOP_K_RETRIEVAL]
        allowed_l = {kw.lower().strip() for kw in bt_gk_map.get(bt_preds_c[i][0], [])}
        top_c = strategy.apply_post_search_filters(top_c, allowed_l)
        if not top_c:
            continue
        qt = f"{names[i]} {descs[i]}".strip()
        p0 = len(tag_pairs)
        for c in top_c:
            tag_pairs.append([qt, c["tag"]])
        tag_pools.append((i, [c["tag"].lower().strip() for c in top_c], p0, len(tag_pairs)))

    gk_diag: dict[str, Any] = {}
    if tag_pools:
        t_scores = rerank_adapter.score_cross_encoder(tag_pairs)
        t_pre_b, t_post_b, t_tots, t_ideals, t_py, t_ps = [], [], [], [], [], []
        for orig_i, c_tags, p0, p1 in tag_pools:
            gs = gt_gks[orig_i]
            t_tots.append(len(gs))
            t_ideals.append([1.0] * min(30, len(gs)) + [0.0] * max(0, 30 - len(gs)))
            sc_slice = t_scores[p0:p1]
            pre_r = [(1 if t in gs else 0, float(sc_slice[idx_c])) for idx_c, t in enumerate(c_tags)]
            post_r = sorted(pre_r, key=lambda x: -x[1])
            t_pre_b.append([x[0] for x in pre_r])
            t_post_b.append([x[0] for x in post_r])
            for rel, sc in pre_r:
                t_py.append(rel)
                t_ps.append(sc)
        gk_diag = M.compute_reranker_diagnostics(
            t_pre_b, t_post_b, t_tots, t_py, t_ps, ideal_graded_per_query=t_ideals
        )

    return {"catalog": cat_diag, "gk_tags": gk_diag}


def _apply_ablation_postprocess(
    ml_results: list[dict[str, Any]],
    prepassed_skus: list[dict[str, Any]],
    domain: str,
    mode: str,
    use_templates: bool,
    use_rules: bool,
) -> list[dict[str, Any]]:
    """Applies Template Tag Enrichment & Rules Engine deterministically over cached ML predictions."""
    rows = copy.deepcopy(ml_results)
    if use_templates:
        orig_flag = config.ENABLE_TEMPLATE_TAG_ENRICHMENT
        config.ENABLE_TEMPLATE_TAG_ENRICHMENT = True
        try:
            _apply_template_tag_enrichment(prepassed_skus, rows, domain, mode=mode)
        finally:
            config.ENABLE_TEMPLATE_TAG_ENRICHMENT = orig_flag

    if use_rules and mode in ("classifier", "pipeline"):
        _load_rules_for_ablation(True)
        for j, row in enumerate(rows):
            sku = prepassed_skus[j]
            if mode == "classifier" or row.get("pipeline_source") == "Classifier":
                conf_val = compute_weighted_confidence(
                    row.get("bt_confidence"),
                    row.get("gk_confidence"),
                    row.get("region_confidence"),
                    has_bt=bool(row.get("suggested_bt")),
                    has_gk=bool(row.get("suggested_gk")),
                    has_region=bool(row.get("suggested_region")),
                )
                match_src = "Classifier"
            else:
                conf_val = float(row.get("score") or 0.0)
                match_src = "Matcher"

            record = {
                "sku_name": sku.get("name", ""),
                "domain": domain,
                "bt": row.get("suggested_bt", ""),
                "gk": [x.strip() for x in str(row.get("suggested_gk", "")).split(",") if x.strip()],
                "region": row.get("suggested_region") if domain == config.DOMAIN_FOOD else None,
                "category": row.get("suggested_region") if domain == config.DOMAIN_MARKET else None,
                "price": sku.get("price", 0.0),
                "confidence": conf_val,
                "match_source": match_src,
                "matched_sku": row.get("matched_catalog_name", ""),
                "reasoning": row.get("logic_notes", ""),
            }
            aug = run_rules_engine(record)
            row["suggested_bt"] = str(aug.get("bt") or "")
            row["suggested_gk"] = ", ".join(aug.get("gk", []))
            row["suggested_region"] = (
                str(aug.get("region") or "")
                if domain == config.DOMAIN_FOOD
                else str(aug.get("category") or "")
            )
            applied = aug.get("rules_applied", [])
            row["rules_applied"] = json.dumps(applied) if applied else ""
    else:
        _load_rules_for_ablation(False)

    return rows


def _score_prediction_set(
    domain: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_skus: list[dict[str, Any]],
    preds: list[dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    third_csv = "Region" if domain == config.DOMAIN_FOOD else "Category"
    gt_bt = test_df["BasicType"].fillna("").astype(str).str.strip().tolist()
    gt_third = test_df[third_csv].fillna("").astype(str).str.strip().tolist()
    gt_gk = test_df["GenericKeywords"].tolist()

    p_bt = [str(r.get("suggested_bt", "") or "") for r in preds]
    p_third = [str(r.get("suggested_region", "") or "") for r in preds]
    p_gk = [str(r.get("suggested_gk", "") or "") for r in preds]

    bt_confs = [
        float(r.get("score") if mode == "matcher" else (r.get("bt_confidence") or r.get("score") or 0.0))
        for r in preds
    ]
    bt_statuses = [str(r.get("status") or r.get("bt_status") or "") for r in preds]
    bt_sources = [str(r.get("pipeline_source") or r.get("bt_source") or r.get("logic_notes") or mode) for r in preds]
    third_confs = [float(r.get("region_confidence") or r.get("score") or 0.0) for r in preds]
    third_statuses = [str(r.get("region_status") or r.get("status") or "") for r in preds]
    gk_statuses = [str(r.get("gk_status") or r.get("status") or "") for r in preds]

    train_names_lower = set(train_df["Name"].astype(str).str.strip().str.lower())
    seen_name_mask = [s["name"].strip().lower() in train_names_lower for s in test_skus]
    train_bt_counts = train_df["basictype"].astype(str).str.strip().value_counts().to_dict()
    train_seen_gks = {
        k.strip().lower()
        for raw in train_df["Generic keywords"].dropna()
        for k in str(raw).split(",")
        if k.strip()
    }

    esc_count = sum(1 for r in preds if r.get("escalated"))
    clf_won_count = sum(1 for r in preds if r.get("escalated") and r.get("pipeline_source") == "Classifier")

    return {
        "bt": M.compute_multiclass_metrics(
            gt_bt, p_bt, confidences=bt_confs, statuses=bt_statuses, sources=bt_sources,
            train_class_counts=train_bt_counts, seen_sku_mask=seen_name_mask,
        ),
        "third_tag": M.compute_multiclass_metrics(
            gt_third, p_third, confidences=third_confs, statuses=third_statuses, seen_sku_mask=seen_name_mask
        ),
        "gk": M.compute_gk_metrics(gt_gk, p_gk, train_seen_gks=train_seen_gks, statuses=gk_statuses),
        "joint": M.compute_joint_metrics(gt_bt, p_bt, gt_third, p_third, gt_gk, p_gk),
        "escalation_pct": round(100.0 * esc_count / max(len(preds), 1), 2),
        "classifier_won_pct": round(100.0 * clf_won_count / max(esc_count, 1), 2) if esc_count else 0.0,
    }


def run_evaluation(args: argparse.Namespace) -> None:
    global EVAL_DIR, STAGED_FIXTURES_DIR, RULES_SNAPSHOT_PATH

    os.environ.setdefault("STR_CACHE_MAX_SIZE", "100000")
    if getattr(args, "eval_dir", ""):
        EVAL_DIR = Path(args.eval_dir).resolve()
        STAGED_FIXTURES_DIR = EVAL_DIR / "fixtures" / "staged_sheets"
        RULES_SNAPSHOT_PATH = EVAL_DIR / "fixtures" / "rules_snapshot.json"

    work_dir = Path(args.work_dir).resolve()
    models_root = Path(args.models_root).resolve() if args.models_root else (work_dir / "models")
    work_dir.mkdir(parents=True, exist_ok=True)
    if not Path(config.GLINER_ONNX).exists():
        gliner_dir = models_root / "gliner"
        config.GLINER_MODEL = gliner_dir
        config.GLINER_ONNX_FP32 = gliner_dir / "model.onnx"
        config.GLINER_ONNX = config.GLINER_ONNX_FP32
    registry = _load_registry(work_dir)

    domains = ["market", "food"] if args.domains == "all" else [d.strip() for d in args.domains.split(",")]
    embeddings = list(EMBEDDING_MODELS.keys()) if args.embeddings == "all" else [e.strip() for e in args.embeddings.split(",")]
    rerankers = list(RERANKER_MODELS.keys()) if args.rerankers == "all" else [r.strip() for r in args.rerankers.split(",")]
    precisions = list(PRECISION_MODES) if args.precisions == "all" else [p.strip() for p in args.precisions.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]
    heads = [h.strip() for h in args.heads.split(",")]

    for domain in domains:
        logger.info(f"\n{'=' * 70}\nEVALUATING DOMAIN: {domain.upper()}\n{'=' * 70}")
        raw_train_df = pd.read_csv(EVAL_DIR / f"train_{domain}.csv")
        raw_test_df = pd.read_csv(EVAL_DIR / f"test_{domain}.csv")
        if args.max_test_rows > 0:
            raw_test_df = (
                _stratified_sample_df(raw_test_df, args.max_test_rows, seed=args.seed)
                if args.stratified_slice
                else raw_test_df.head(args.max_test_rows).reset_index(drop=True)
            )
            logger.info(f"[{domain.upper()}] Using test slice: {len(raw_test_df)} rows (stratified={args.stratified_slice})")

        train_df = _prepare_train_catalog_df(raw_train_df, domain=domain)
        brands_df, dicts = _load_staged_domain_knowledge(domain)
        test_skus = _build_test_skus(raw_test_df)

        # Stage A: NER cache (GLiNER + dict) once per domain
        train_df, cached_ner = ensure_stage_a_ner(domain, train_df, raw_test_df, brands_df, work_dir)
        registry["stage_a"][domain] = {"done": True, "train_rows": len(train_df), "test_rows": len(raw_test_df)}
        _save_registry(work_dir, registry)

        # Pre-compute food NER-enriched SKU copy (matching processor.py lines 214-235)
        prepassed_skus = [s.copy() for s in test_skus]
        if domain == config.DOMAIN_FOOD:
            nt_list = [
                TextPipeline.prep_for_ner(TextPipeline.build_ner_input(s["name"], s["description"], s["category"]))
                for s in prepassed_skus
            ]
            ner_res = cached_ner.batch_extract_entities(nt_list)
            for s, ent in zip(prepassed_skus, ner_res):
                suffix = TextPipeline.resolve_food_mixed_suffix(s["name"], ent.get("flavor", set()), cached_ner)
                if suffix:
                    s["name"] = f"{s['name']} {suffix}"

        for emb_model in embeddings:
            emb_slug = _slug(emb_model)
            for prec in precisions:
                # Skip combos if ALL rerankers for this (domain, emb, prec) are already completed
                pending_rerankers = [
                    rr
                    for rr in rerankers
                    if not (
                        args.resume
                        and registry["stage_c"]
                        .get(
                            _combo_id(
                                domain, emb_slug, _slug(rr), prec, args.max_test_rows, len(raw_test_df)
                            ),
                            {},
                        )
                        .get("done")
                    )
                ]
                if not pending_rerankers:
                    logger.info(f"[SKIP] All rerankers completed for {domain}/{emb_model}/{prec}.")
                    continue

                art_base = work_dir / "artifacts" / domain / f"{emb_slug}_{prec}"
                cache_dir = art_base / "cache"
                arcface_dir = art_base / "arcface"
                qdrant_dir = art_base / "qdrant_local"

                config.QDRANT_COLLECTION_FOOD = "eval_food_catalog"
                config.QDRANT_COLLECTION_MARKET = "eval_market_catalog"
                config.QDRANT_TAGS_PREFIX = "eval_"
                _setup_qdrant_client(args.qdrant_url or None, qdrant_dir)

                t_b0 = time.perf_counter()
                emb_engine = EvalEmbeddingEngine(
                    embedding_model=emb_model,
                    reranker_model=None,
                    emb_precision=prec,
                    models_root=models_root,
                    device=args.device,
                    batch_size=args.embed_batch_size,
                )
                config.EMBEDDING_DIM = int(getattr(emb_engine, "embedding_dim", config.EMBEDDING_DIM))

                clf, _ = _fit_eval_classifiers_and_arcface(
                    domain, train_df, brands_df, dicts, emb_engine, cached_ner,
                    cache_dir, arcface_dir, epochs=args.epochs, arcface_batch_size=args.arcface_batch_size,
                )
                matcher, base_vs = _sync_eval_qdrant(
                    domain, train_df, brands_df, dicts, emb_engine, cached_ner,
                    cache_dir, emb_model, prec,
                )

                stage_b_id = f"{domain}__{emb_slug}__{prec}__{len(raw_test_df)}"
                stage_b_path = work_dir / "checkpoints" / domain / f"stage_b_{emb_slug}_{prec}_{len(raw_test_df)}.pkl"
                if args.resume and stage_b_path.exists():
                    stage_b_data = joblib.load(stage_b_path)
                    logger.info(f"[STAGE-B] Loaded cached standalone metrics & retrieval pool: {stage_b_path.name}")
                else:
                    emb_standalone, head_bundle = _evaluate_standalone_embeddings_and_heads(
                        domain, train_df, raw_test_df, test_skus, clf, emb_engine, base_vs,
                        cache_dir, arcface_dir, prec,
                    )
                    stage_b_data = {
                        "emb_standalone": emb_standalone,
                        "head_standalone": head_bundle["head_standalone"],
                        "qdrant_cache": {},
                        "query_emb_cache": {},
                        "emb_backend": emb_engine.emb_backend,
                        "stage_b_s": round(time.perf_counter() - t_b0, 2),
                    }
                    _atomic_write_pickle(stage_b_path, stage_b_data)
                    registry["stage_b"][stage_b_id] = {"done": True, "stage_b_s": stage_b_data["stage_b_s"]}
                    _save_registry(work_dir, registry)

                rec_vs = RecordingVectorStore(base_vs, stage_b_data["qdrant_cache"])

                for rerank_model in pending_rerankers:
                    rerank_slug = _slug(rerank_model)
                    combo_id = _combo_id(
                        domain, emb_slug, rerank_slug, prec, args.max_test_rows, len(raw_test_df)
                    )
                    logger.info(f"\n--- Stage C: {combo_id} ---")

                    t_c0 = time.perf_counter()
                    rerank_engine = EvalEmbeddingEngine(
                        embedding_model=None,
                        reranker_model=rerank_model,
                        rerank_precision=prec,
                        models_root=models_root,
                        device=args.device,
                        batch_size=args.rerank_batch_size,
                    )

                    pair_ckpt = work_dir / "checkpoints" / domain / f"rerank_pairs_{combo_id}.pkl"
                    pair_cache: dict[tuple[str, str], float] = joblib.load(pair_ckpt) if (args.resume and pair_ckpt.exists()) else {}

                    adapter = CachedRerankAdapter(
                        emb_engine=emb_engine,
                        rerank_engine=rerank_engine,
                        pair_score_cache=pair_cache,
                        query_emb_cache=stage_b_data["query_emb_cache"],
                        pair_ckpt_path=pair_ckpt,
                    )
                    matcher.embedder = adapter
                    matcher.rules = LogicGates(adapter, brands_df=brands_df)
                    matcher.vector_store = rec_vs
                    clf.model = adapter
                    clf._active_model = adapter
                    if clf.cold_start_router is not None:
                        clf.cold_start_router.embed_engine = adapter
                        clf.cold_start_router.vector_store = rec_vs
                        clf.cold_start_router.tier1_zero_shot.embed_engine = adapter
                        clf.cold_start_router.tier2_few_shot.vector_store = rec_vs

                    _bind_eval_singletons(
                        domain, train_df, brands_df, dicts, adapter, cached_ner, clf, matcher
                    )

                    run_summaries: dict[str, Any] = {}
                    out_dir = work_dir / "results" / combo_id
                    out_dir.mkdir(parents=True, exist_ok=True)

                    for head in heads:
                        _set_classifier_head(clf, domain, head, prec, arcface_dir)
                        matcher.classifier = clf

                        for mode in modes:
                            # Run core ML-only pass once with templates=False & rules=False
                            orig_tmpl = config.ENABLE_TEMPLATE_TAG_ENRICHMENT
                            config.ENABLE_TEMPLATE_TAG_ENRICHMENT = False
                            _load_rules_for_ablation(False)
                            try:
                                raw_resp = process_request(skus=test_skus, task=mode, domain=domain)
                            finally:
                                config.ENABLE_TEMPLATE_TAG_ENRICHMENT = orig_tmpl

                            ml_rows = raw_resp["results"]
                            abl_configs = (
                                [(False, False), (True, False)]
                                if mode == "matcher"
                                else [(False, False), (True, False), (False, True), (True, True)]
                            )
                            for use_t, use_r in abl_configs:
                                abl_name = f"templates_{'on' if use_t else 'off'}__rules_{'on' if use_r else 'off'}"
                                run_key = f"{mode}__{head}__{abl_name}"
                                abl_rows = _apply_ablation_postprocess(
                                    ml_rows, prepassed_skus, domain, mode, use_t, use_r
                                )
                                run_summaries[run_key] = _score_prediction_set(
                                    domain, train_df, raw_test_df, test_skus, abl_rows, mode
                                )
                                # Save per-SKU predictions for primary production & ML-only configs
                                if (mode == "matcher" and use_t and not use_r) or (
                                    mode in ("classifier", "pipeline")
                                    and ((use_t and use_r) or (not use_t and not use_r))
                                ):
                                    sku_df = M.build_per_sku_df(raw_test_df, "Region" if domain == "food" else "Category", abl_rows)
                                    sku_df.to_parquet(out_dir / f"per_sku_{run_key}.parquet", index=False)

                            # Persist intermediate caches after each mode so disconnects lose minimal work
                            _atomic_write_pickle(pair_ckpt, pair_cache)
                            if rec_vs._dirty:
                                _atomic_write_pickle(stage_b_path, stage_b_data)
                                rec_vs._dirty = False

                    # Reranker lift diagnostics (computed once per combo using active head=heads[0])
                    _set_classifier_head(clf, domain, heads[0], prec, arcface_dir)
                    rerank_diag = _compute_reranker_diagnostics_for_combo(
                        domain, train_df, raw_test_df, test_skus, clf, matcher, rec_vs, adapter, cached_ner
                    )

                    # Persist updated pair & Qdrant caches cleanly
                    _atomic_write_pickle(pair_ckpt, pair_cache)
                    if rec_vs._dirty:
                        _atomic_write_pickle(stage_b_path, stage_b_data)
                        rec_vs._dirty = False

                    summary_payload = {
                        "meta": {
                            "combo_id": combo_id,
                            "domain": domain,
                            "embedding_model": emb_model,
                            "reranker_model": rerank_model,
                            "precision": prec,
                            "emb_backend": stage_b_data["emb_backend"],
                            "rerank_backend": rerank_engine.rerank_backend,
                            "train_rows": len(train_df),
                            "test_rows": len(raw_test_df),
                        },
                        "timings_s": {
                            "stage_b_s": stage_b_data["stage_b_s"],
                            "rerank_scoring_s": round(adapter.rerank_time_s, 2),
                            "rerank_unique_pairs": adapter.rerank_pairs_scored,
                            "total_combo_s": round(time.perf_counter() - t_c0, 2),
                        },
                        "embedding_standalone": stage_b_data["emb_standalone"],
                        "head_standalone": stage_b_data["head_standalone"],
                        "reranker_diagnostics": rerank_diag,
                        "runs": run_summaries,
                    }
                    _atomic_write_json(out_dir / "summary.json", summary_payload)
                    registry["stage_c"][combo_id] = {
                        "done": True,
                        "summary": str(out_dir / "summary.json"),
                    }
                    _save_registry(work_dir, registry)
                    generate_reports(work_dir)

                    del rerank_engine, adapter
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                del emb_engine, clf, matcher, base_vs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run end-to-end resumable SKU MatchOps evaluation.")
    parser.add_argument("--domains", default="all", help="'all', 'market', 'food', or comma list.")
    parser.add_argument("--embeddings", default="all", help="'all' or comma-separated HF embedding IDs.")
    parser.add_argument("--rerankers", default="all", help="'all' or comma-separated HF reranker IDs.")
    parser.add_argument("--precisions", default="all", help="'all' or comma-separated fp32,fp16,int8.")
    parser.add_argument("--modes", default="matcher,classifier,pipeline")
    parser.add_argument("--heads", default="arcface,logreg")
    parser.add_argument("--work-dir", default=str(EVAL_DIR / "runs"))
    parser.add_argument("--eval-dir", default="", help="Optional path containing train_*.csv, test_*.csv, and fixtures/.")
    parser.add_argument("--models-root", default="")
    parser.add_argument("--qdrant-url", default="", help="Optional Qdrant server URL (recommended in Colab).")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--embed-batch-size", type=int, default=128)
    parser.add_argument("--rerank-batch-size", type=int, default=64)
    parser.add_argument("--arcface-batch-size", type=int, default=64)
    parser.add_argument("--max-test-rows", type=int, default=0, help="0 = full test set; >0 = subset.")
    parser.add_argument("--stratified-slice", action="store_true", help="Stratify subset by BasicType.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
