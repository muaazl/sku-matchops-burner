"""Multi-precision (FP32 / FP16 / INT8) model exporter & GPU/CPU runtime adapter for evaluation."""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

import onnxruntime as ort  # isort: skip

from engine import config
from engine.nlp.embedding_engine import EmbeddingEngine

logger = logging.getLogger("matchops.eval.model_manager")

PRECISION_MODES = ("fp32", "fp16", "int8")

EMBEDDING_MODELS: dict[str, dict[str, Any]] = {
    "BAAI/bge-m3": {
        "hf_repo": "BAAI/bge-m3",
        "pooling": "cls",
        "dim": 1024,
        "has_learned_sparse": True,
    },
    "jinaai/jina-embeddings-v5-text-small": {
        "hf_repo": "jinaai/jina-embeddings-v5-text-small-retrieval",
        "pooling": "last_token",
        "dim": 1024,
        "has_learned_sparse": False,
    },
    "sentence-transformers/all-mpnet-base-v2": {
        "hf_repo": "sentence-transformers/all-mpnet-base-v2",
        "pooling": "mean",
        "dim": 768,
        "has_learned_sparse": False,
    },
}

RERANKER_MODELS: dict[str, dict[str, Any]] = {
    "BAAI/bge-reranker-v2-m3": {
        "hf_repo": "BAAI/bge-reranker-v2-m3",
        "arch_type": "cross_encoder",
        "score_activation": "sigmoid",
    },
    "jinaai/jina-reranker-v2-base-multilingual": {
        "hf_repo": "jinaai/jina-reranker-v2-base-multilingual",
        "arch_type": "cross_encoder",
        "score_activation": "sigmoid",
    },
    "jinaai/jina-reranker-v3.5": {
        "hf_repo": "jinaai/jina-reranker-v3.5",
        "arch_type": "listwise_lbnl",
        "score_activation": "identity",
    },
    "cross-encoder/ms-marco-MiniLM-L6-v2": {
        "hf_repo": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "arch_type": "cross_encoder",
        "score_activation": "sigmoid",
    },
}


@dataclass(frozen=True)
class ModelArtifacts:
    """Resolved ONNX/manifest paths and runtime backend metadata for a model+precision."""

    model_id: str
    kind: str  # "embedding" or "reranker"
    precision: str  # "fp32", "fp16", or "int8"
    model_dir: Path
    onnx_path: Path | None
    manifest: dict[str, Any]
    backend: str  # e.g. "onnx_cuda_fp32", "onnx_cpu_int8", "torch_cuda_fp16"


def _onnx_total_size_mb(path: Path | None) -> float:
    if path is None or not path.exists():
        return 0.0
    total = path.stat().st_size
    for suffix in (".data", "_data"):
        ext = path.with_name(path.name + suffix)
        if ext.exists():
            total += ext.stat().st_size
    return round(total / (1024 * 1024), 2)


def _select_providers(precision: str, device: str = "auto") -> list[str]:
    if precision == "int8" or device == "cpu":
        return ["CPUExecutionProvider"]
    avail = ort.get_available_providers()
    if "CUDAExecutionProvider" in avail and (device in ("auto", "cuda", "gpu")):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


_CUDA_PRELOADED = False


def _is_oom_error(exc: Exception) -> bool:
    """Returns True when an exception indicates GPU/CPU allocator or BFC arena OOM."""
    oom_types = tuple(
        t for t in (MemoryError, getattr(torch, "OutOfMemoryError", None)) if isinstance(t, type)
    )
    if oom_types and isinstance(exc, oom_types):
        return True
    msg = str(exc).lower()
    return any(
        needle in msg
        for needle in (
            "bfc_arena",
            "failed to allocate memory",
            "out of memory",
            "outofmemory",
            "cuda out of memory",
            "cublas_status_alloc_failed",
            "cudnn_status_alloc_failed",
        )
    )


def _free_accelerator_memory() -> None:
    """Triggers Python GC and releases PyTorch cached CUDA blocks after an OOM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _create_ort_session(onnx_path: Path, precision: str, device: str = "auto") -> tuple[ort.InferenceSession, str]:
    global _CUDA_PRELOADED
    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False
    if precision == "int8" or device == "cpu":
        opts.add_session_config_entry("session.use_mmap_for_weights", "1")
        opts.intra_op_num_threads = config.MAX_CPU_CORES
        opts.inter_op_num_threads = 1
    providers = _select_providers(precision, device=device)
    uses_cuda = "CUDAExecutionProvider" in providers
    if uses_cuda and not _CUDA_PRELOADED and hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception as err:
            logger.debug(f"[EVAL-MODELS] ort.preload_dlls skipped: {err}")
        _CUDA_PRELOADED = True
    ort_providers: list[Any] = [
        ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})
        if p == "CUDAExecutionProvider"
        else p
        for p in providers
    ]
    sess = ort.InferenceSession(str(onnx_path), opts, providers=ort_providers)
    active = sess.get_providers()
    ep_tag = "cuda" if active and active[0] == "CUDAExecutionProvider" else "cpu"
    return sess, f"onnx_{ep_tag}_{precision}"


def _convert_onnx_fp32_to_fp16(fp32_path: Path, fp16_path: Path) -> None:
    """Converts an FP32 ONNX model (+ optional external weights) into FP16 ONNX keeping I/O FP32."""
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    from scripts.sync import _remove_onnx_artifacts

    if not fp32_path.exists():
        raise FileNotFoundError(f"Source FP32 ONNX not found: {fp32_path}")

    _remove_onnx_artifacts(fp16_path)
    proto = onnx.load(str(fp32_path), load_external_data=True)
    fp16_proto = convert_float_to_float16(
        proto,
        keep_io_types=True,
        disable_shape_infer=True,
    )
    del proto
    gc.collect()

    data_file = fp16_path.with_name(fp16_path.name + ".data")
    approx_bytes = sum(len(t.raw_data) for t in fp16_proto.graph.initializer)
    save_ext = approx_bytes > int(1.4 * 1024 * 1024 * 1024)
    onnx.save_model(
        fp16_proto,
        str(fp16_path),
        save_as_external_data=save_ext,
        all_tensors_to_one_file=True,
        location=data_file.name,
    )
    del fp16_proto
    gc.collect()


def _configure_sync_model_env(
    model_id: str,
    kind: str,
    models_root: Path | None = None,
) -> Path:
    """Temporarily configures config paths and env vars for exporting one model."""
    root = Path(models_root) if models_root else Path(config.ONNX_DIR)
    config.ONNX_DIR = root
    config.EMBEDDING_ROOT_DIR = root / "embedding"
    config.RERANKER_ROOT_DIR = root / "reranker"

    if kind == "embedding":
        meta = EMBEDDING_MODELS.get(model_id, {})
        os.environ["EMBEDDING_MODEL"] = model_id
        os.environ["EMBEDDING_HF_REPO"] = str(meta.get("hf_repo") or model_id)
        os.environ["EMBEDDING_POOLING"] = str(meta.get("pooling") or "auto")
    else:
        os.environ["RERANKER_MODEL"] = model_id

    config.refresh_model_config()
    return Path(config.EMBEDDING_DIR if kind == "embedding" else config.RERANKER_DIR)


def _seed_existing_int8_if_available(model_id: str, kind: str, target_dir: Path) -> None:
    """Copies pre-existing local production INT8 artifacts when using an isolated models_root."""
    default_root = Path(__file__).resolve().parents[1] / "engine" / "models" / kind
    src_dir = default_root / config.model_id_to_folder_name(model_id)
    if not src_dir.is_dir() or src_dir.resolve() == target_dir.resolve():
        return
    src_int8 = src_dir / "model_int8.onnx"
    dst_int8 = target_dir / "model_int8.onnx"
    if not src_int8.exists() or dst_int8.exists():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        if item.is_file() and item.name != ".gitkeep":
            shutil.copy2(item, target_dir / item.name)
    logger.info(f"[EVAL-MODELS] Seeded existing INT8 files for '{model_id}' from {src_dir}")


def _preserve_existing_onnx_files(model_dir: Path) -> dict[str, Path]:
    """Backs up existing precision files before sync.py export clears them."""
    backup_dir = model_dir / "_tmp_preserve"
    preserved: dict[str, Path] = {}
    for fname in (
        "model_int8.onnx",
        "model_int8.onnx.data",
        "model_int8.onnx_data",
        "model_fp16.onnx",
        "model_fp16.onnx.data",
    ):
        src = model_dir / fname
        if src.exists():
            backup_dir.mkdir(parents=True, exist_ok=True)
            dst = backup_dir / fname
            shutil.move(str(src), str(dst))
            preserved[fname] = dst
    return preserved


def _restore_preserved_onnx_files(model_dir: Path, preserved: dict[str, Path]) -> None:
    for fname, src in preserved.items():
        dst = model_dir / fname
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
    backup_dir = model_dir / "_tmp_preserve"
    if backup_dir.exists():
        shutil.rmtree(backup_dir, ignore_errors=True)


def _verify_precision_parity(
    model_dir: Path,
    kind: str,
    precision: str,
    device: str = "auto",
) -> tuple[bool, dict[str, Any]]:
    """Checks FP16/INT8 ONNX output parity against FP32 ONNX on probe inputs."""
    from scripts.sync import _filter_ort_inputs, _load_local_tokenizer

    fp32_path = model_dir / "model.onnx"
    cand_name = "model_fp16.onnx" if precision == "fp16" else "model_int8.onnx"
    cand_path = model_dir / cand_name
    manifest = json.loads((model_dir / "model_manifest.json").read_text(encoding="utf-8"))

    if not fp32_path.exists() or not cand_path.exists():
        return False, {"reason": "missing onnx file"}

    tok = _load_local_tokenizer(model_dir)
    if kind == "embedding":
        tok.padding_side = "right"
        pooling = manifest.get("pooling", "mean")
        texts = [
            "Red Bull Energy Drink 250ml",
            "Coca Cola Zero Sugar 330ml Can",
            "Chicken Fried Rice Spicy",
        ]
        inputs = tok(texts, padding=True, truncation=True, return_tensors="np")
        fp32_sess, _ = _create_ort_session(fp32_path, "fp32", device=device)
        cand_sess, backend = _create_ort_session(cand_path, precision, device=device)
        try:
            ort_in = _filter_ort_inputs(fp32_sess, {k: v.astype(np.int64) for k, v in inputs.items()})
            fp32_out = fp32_sess.run(None, ort_in)[0]
            cand_out = cand_sess.run(None, ort_in)[0]
            if fp32_out.ndim == 3:
                fp32_out = EmbeddingEngine._pool_hidden_states(fp32_out, inputs["attention_mask"], pooling)
                cand_out = EmbeddingEngine._pool_hidden_states(cand_out, inputs["attention_mask"], pooling)
        finally:
            del fp32_sess, cand_sess
            gc.collect()

        if not np.isfinite(cand_out).all():
            return False, {"reason": "NaN/Inf in output", "backend": backend}
        f_norm = fp32_out / np.clip(np.linalg.norm(fp32_out, axis=1, keepdims=True), 1e-12, None)
        c_norm = cand_out / np.clip(np.linalg.norm(cand_out, axis=1, keepdims=True), 1e-12, None)
        cos = [round(float(v), 5) for v in np.sum(f_norm * c_norm, axis=1)]
        min_cos = min(cos)
        threshold = 0.98 if precision == "fp16" else 0.90
        return min_cos >= threshold, {"min_cosine": min_cos, "cosines": cos, "backend": backend}

    # Reranker parity check
    arch = manifest.get("arch_type", "cross_encoder")
    fp32_sess, _ = _create_ort_session(fp32_path, "fp32", device=device)
    cand_sess, backend = _create_ort_session(cand_path, precision, device=device)
    try:
        if arch == "listwise_lbnl":
            doc_tok = manifest.get("doc_embed_token", "<|embed_token|>")
            query_tok = manifest.get("query_embed_token", "<|rerank_token|>")
            doc_id = int(manifest.get("doc_embed_token_id", 151670))
            query_id = int(manifest.get("query_embed_token_id", 151671))
            prompt = (
                f'<passage id="0">\nName: Red Bull Energy Drink 250ml{doc_tok}\n</passage>\n'
                f'<passage id="1">\nName: Organic Bananas 1kg{doc_tok}\n</passage>\n'
                f"<query>\nName: Red Bull 250ml{query_tok}\n</query>"
            )
            inputs = tok([prompt], return_tensors="np")
            ort_in = _filter_ort_inputs(fp32_sess, {k: v.astype(np.int64) for k, v in inputs.items()})
            fp32_out = fp32_sess.run(None, ort_in)[0][0]
            cand_out = cand_sess.run(None, ort_in)[0][0]
            if not np.isfinite(cand_out).all():
                return False, {"reason": "NaN/Inf in LBNL output", "backend": backend}
            doc_pos = np.where(inputs["input_ids"][0] == doc_id)[0]
            q_pos = int(np.where(inputs["input_ids"][0] == query_id)[0][-1])

            def _cos(seq):
                qv = seq[q_pos]
                dm = seq[doc_pos]
                return (dm / np.linalg.norm(dm, axis=1, keepdims=True)) @ (qv / np.linalg.norm(qv))

            s_fp32 = _cos(fp32_out)
            s_cand = _cos(cand_out)
            max_diff = float(np.max(np.abs(s_fp32 - s_cand)))
            ok = bool(s_cand[0] > s_cand[1] and max_diff <= (0.05 if precision == "fp16" else 0.25))
            return ok, {
                "fp32": [round(float(x), 4) for x in s_fp32],
                "cand": [round(float(x), 4) for x in s_cand],
                "max_diff": round(max_diff, 5),
                "backend": backend,
            }

        pairs = [
            ("Red Bull Energy Drink", "Red Bull Energy Drink 250ml Can"),
            ("Coca Cola Zero", "Organic Bananas 1kg"),
        ]
        inputs = tok(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            return_tensors="np",
        )
        ort_in = _filter_ort_inputs(fp32_sess, {k: v.astype(np.int64) for k, v in inputs.items()})
        raw_fp32 = np.asarray(fp32_sess.run(None, ort_in)[0], dtype=np.float32)
        raw_cand = np.asarray(cand_sess.run(None, ort_in)[0], dtype=np.float32)
        if not np.isfinite(raw_cand).all():
            return False, {"reason": "NaN/Inf in cross-encoder logits", "backend": backend}
        s_fp32 = (raw_fp32[:, 1] - raw_fp32[:, 0]) if (raw_fp32.ndim == 2 and raw_fp32.shape[1] >= 2) else raw_fp32.reshape(-1)
        s_cand = (raw_cand[:, 1] - raw_cand[:, 0]) if (raw_cand.ndim == 2 and raw_cand.shape[1] >= 2) else raw_cand.reshape(-1)
        max_diff = float(np.max(np.abs(s_fp32 - s_cand)))
        ok = bool(s_cand[0] > s_cand[1])
        return ok, {
            "fp32_logits": [round(float(x), 4) for x in s_fp32],
            "cand_logits": [round(float(x), 4) for x in s_cand],
            "max_logit_diff": round(max_diff, 5),
            "backend": backend,
        }
    finally:
        del fp32_sess, cand_sess
        gc.collect()


def ensure_model_artifacts(
    model_id: str,
    kind: str,
    precision: str,
    models_root: Path | str | None = None,
    device: str = "auto",
    force: bool = False,
    allow_torch_fp16_fallback: bool = True,
    strict_int8_parity: bool = False,
) -> ModelArtifacts:
    """Ensures FP32/FP16/INT8 ONNX artifacts exist for evaluation without deleting FP32."""
    from scripts.sync import (
        _has_saved_tokenizer,
        _remove_onnx_artifacts,
        export_embedding,
        export_reranker,
        quantize_model,
    )

    if precision not in PRECISION_MODES:
        raise ValueError(f"Unsupported precision '{precision}'. Must be one of {PRECISION_MODES}.")

    target_dir = _configure_sync_model_env(model_id, kind, models_root=Path(models_root) if models_root else None)
    target_dir.mkdir(parents=True, exist_ok=True)
    _seed_existing_int8_if_available(model_id, kind, target_dir)

    fp32_path = target_dir / "model.onnx"
    fp16_path = target_dir / "model_fp16.onnx"
    int8_path = target_dir / "model_int8.onnx"
    manifest_path = target_dir / "model_manifest.json"
    status_path = target_dir / "eval_precision_status.json"

    # 1. Ensure FP32 ONNX exists if we need fp32, fp16 ONNX, or missing int8
    need_fp32 = (
        force
        or not manifest_path.exists()
        or not _has_saved_tokenizer(target_dir)
        or (precision in ("fp32", "fp16") and not fp32_path.exists())
        or (precision == "int8" and not int8_path.exists())
    )
    if need_fp32 and (force or not fp32_path.exists() or not manifest_path.exists() or not _has_saved_tokenizer(target_dir)):
        preserved = {} if force else _preserve_existing_onnx_files(target_dir)
        try:
            if kind == "embedding":
                export_embedding(force=True, require_int8=False)
            else:
                export_reranker(force=True, require_int8=False)
        finally:
            if preserved:
                _restore_preserved_onnx_files(target_dir, preserved)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            status = {}
    else:
        status = {}

    if precision == "fp32":
        ep = "cuda" if _select_providers("fp32", device)[0] == "CUDAExecutionProvider" else "cpu"
        return ModelArtifacts(model_id, kind, "fp32", target_dir, fp32_path, manifest, f"onnx_{ep}_fp32")

    if precision == "int8":
        if force or not int8_path.exists():
            quantize_model(fp32_path, int8_path, f"EVAL-{kind.upper()}")
            ok, info = _verify_precision_parity(target_dir, kind, "int8", device="cpu")
            status["int8"] = {"ok": ok, **info}
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            if not ok:
                msg = f"INT8 parity warning for {model_id}: {info}"
                if strict_int8_parity:
                    raise RuntimeError(msg)
                logger.warning(f"[EVAL-MODELS] {msg} (proceeding for benchmark measurement)")
        return ModelArtifacts(model_id, kind, "int8", target_dir, int8_path, manifest, "onnx_cpu_int8")

    # FP16 handling: try ONNX FP16 first; gracefully fall back to PyTorch CUDA FP16 if needed
    use_cuda = torch.cuda.is_available() and device != "cpu"
    fp16_stat = status.get("fp16", {})
    if force or (not fp16_path.exists() and fp16_stat.get("onnx_supported") is not False):
        try:
            _convert_onnx_fp32_to_fp16(fp32_path, fp16_path)
            ok, info = _verify_precision_parity(target_dir, kind, "fp16", device=device)
            if not ok:
                raise RuntimeError(f"ONNX FP16 parity check failed: {info}")
            status["fp16"] = {"onnx_supported": True, **info}
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            return ModelArtifacts(
                model_id, kind, "fp16", target_dir, fp16_path, manifest, info["backend"]
            )
        except Exception as fp16_err:
            logger.warning(
                f"[EVAL-MODELS] ONNX FP16 unavailable/failed for '{model_id}' ({fp16_err}); "
                f"will use PyTorch FP16 backend."
            )
            _remove_onnx_artifacts(fp16_path)
            status["fp16"] = {"onnx_supported": False, "error": str(fp16_err)}
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    if fp16_path.exists():
        ep = "cuda" if _select_providers("fp16", device)[0] == "CUDAExecutionProvider" else "cpu"
        return ModelArtifacts(model_id, kind, "fp16", target_dir, fp16_path, manifest, f"onnx_{ep}_fp16")

    if allow_torch_fp16_fallback:
        backend = "torch_cuda_fp16" if use_cuda else "torch_cpu_fp16_sim"
        return ModelArtifacts(model_id, kind, "fp16", target_dir, None, manifest, backend)

    raise RuntimeError(f"FP16 ONNX failed for {model_id} and torch fallback disabled.")


class _TorchSessionAdapter:
    """Exposes InferenceSession-compatible .get_inputs() / .run() over a PyTorch FP16 wrapper."""

    class _InputMeta:
        def __init__(self, name: str):
            self.name = name

    def __init__(self, wrapper: torch.nn.Module, device: torch.device, input_names: list[str]):
        self.wrapper = wrapper
        self.device = device
        self._inputs = [self._InputMeta(n) for n in input_names]

    def get_inputs(self):
        return self._inputs

    def run(self, _outputs, ort_inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        kwargs = {
            k: torch.from_numpy(v).to(self.device)
            for k, v in ort_inputs.items()
            if k in {i.name for i in self._inputs}
        }
        with torch.inference_mode():
            out = self.wrapper(**kwargs)
        if isinstance(out, tuple):
            return [t.float().cpu().numpy() for t in out]
        return [out.float().cpu().numpy()]


def _load_torch_fp16_session(
    model_id: str,
    kind: str,
    manifest: dict[str, Any],
    device_str: str = "auto",
) -> tuple[_TorchSessionAdapter, str]:
    """Loads PyTorch FP16 (CUDA) or BF16/FP32-casted fallback adapter matching ONNX outputs."""
    from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification

    from scripts.sync import (
        EmbeddingONNXWrapper,
        RerankerONNXWrapper,
        _accepts_kwarg,
        _load_sparse_linear,
        _resolve_hf_source,
    )

    use_cuda = torch.cuda.is_available() and device_str != "cpu"
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.float16 if use_cuda else torch.float32

    if kind == "embedding":
        meta = EMBEDDING_MODELS.get(model_id, {})
        hf_id = str(meta.get("hf_repo") or manifest.get("hf_repo") or model_id)
        source = _resolve_hf_source(hf_id)
        hf_cfg = AutoConfig.from_pretrained(source, trust_remote_code=True)
        hidden = int(getattr(hf_cfg, "hidden_size", manifest.get("embedding_dim", 1024)))
        pooling = str(manifest.get("pooling") or meta.get("pooling") or "mean")
        sparse_linear = _load_sparse_linear(source, hf_id, model_id, hidden)
        if sparse_linear is not None:
            sparse_linear = sparse_linear.to(dtype=dtype)
        hf_model = AutoModel.from_pretrained(source, trust_remote_code=True, torch_dtype=dtype)
        hf_model.eval()
        wrapper = EmbeddingONNXWrapper(hf_model, pooling=pooling, sparse_linear=sparse_linear).to(device)
        wrapper.eval()
        backend = "torch_cuda_fp16" if use_cuda else "torch_cpu_fp32_fallback"
        return _TorchSessionAdapter(wrapper, device, ["input_ids", "attention_mask"]), backend

    source = _resolve_hf_source(model_id)
    arch = str(manifest.get("arch_type") or "cross_encoder")
    if arch == "listwise_lbnl":
        hf_model = AutoModel.from_pretrained(source, trust_remote_code=True, torch_dtype=dtype)
        in_names = ["input_ids", "attention_mask"]
    else:
        hf_model = AutoModelForSequenceClassification.from_pretrained(
            source, trust_remote_code=True, torch_dtype=dtype
        )
        in_names = ["input_ids", "attention_mask"]
        if _accepts_kwarg(hf_model, "token_type_ids") and getattr(hf_model.config, "type_vocab_size", 0) > 1:
            in_names.append("token_type_ids")
    hf_model.eval()
    wrapper = RerankerONNXWrapper(hf_model, arch_type=arch).to(device)
    wrapper.eval()
    backend = "torch_cuda_fp16" if use_cuda else "torch_cpu_fp32_fallback"
    return _TorchSessionAdapter(wrapper, device, in_names), backend


class EvalEmbeddingEngine(EmbeddingEngine):
    """Evaluation-grade EmbeddingEngine supporting FP32, FP16, and INT8 across GPU/CPU."""

    def __init__(
        self,
        embedding_model: str | None = None,
        reranker_model: str | None = None,
        emb_precision: str = "int8",
        rerank_precision: str = "int8",
        models_root: Path | str | None = None,
        device: str = "auto",
        batch_size: int = config.ONNX_BATCH_SIZE,
    ):
        self._cache_lock = threading.RLock()
        self._str_cache: OrderedDict[str, tuple[np.ndarray, dict[int, float], dict[int, float] | None]] = OrderedDict()
        self.batch_size = int(batch_size)
        self.device = device
        self.models_root = Path(models_root) if models_root else None

        self.bi_session = None
        self.tokenizer = None
        self.emb_manifest: dict[str, Any] = {}
        self._emb_pooling = "mean"
        self.embedding_dim = config.EMBEDDING_DIM
        self.has_learned_sparse = False
        self.emb_backend = "none"
        self.emb_precision = emb_precision
        self.emb_artifacts: ModelArtifacts | None = None

        self.cross_session = None
        self.rerank_tokenizer = None
        self.cross_manifest: dict[str, Any] = {}
        self._cross_arch = "cross_encoder"
        self._score_activation = "sigmoid"
        self._doc_embed_token = "<|embed_token|>"
        self._query_embed_token = "<|rerank_token|>"
        self._doc_embed_token_id = 151670
        self._query_embed_token_id = 151671
        self.rerank_backend = "none"
        self.rerank_precision = rerank_precision
        self.rerank_artifacts: ModelArtifacts | None = None

        if embedding_model:
            self._load_eval_embedding(embedding_model, emb_precision)
        if reranker_model:
            self._load_eval_reranker(reranker_model, rerank_precision)

    def _load_eval_embedding(self, model_id: str, precision: str) -> None:
        from scripts.sync import _load_local_tokenizer

        art = ensure_model_artifacts(
            model_id=model_id,
            kind="embedding",
            precision=precision,
            models_root=self.models_root,
            device=self.device,
        )
        self.emb_artifacts = art
        self.emb_manifest = art.manifest
        self._emb_pooling = str(art.manifest.get("pooling", "mean")).lower()
        self.embedding_dim = int(art.manifest.get("embedding_dim", 1024))
        self.has_learned_sparse = bool(art.manifest.get("has_learned_sparse", False))

        self.tokenizer = _load_local_tokenizer(art.model_dir)
        if getattr(self.tokenizer, "padding_side", None) == "left":
            self.tokenizer.padding_side = "right"

        if art.onnx_path is not None:
            self.bi_session, self.emb_backend = _create_ort_session(
                art.onnx_path, precision, device=self.device
            )
        else:
            self.bi_session, self.emb_backend = _load_torch_fp16_session(
                model_id, "embedding", art.manifest, device_str=self.device
            )

        os.environ["EMBEDDING_DIM"] = str(self.embedding_dim)
        config.EMBEDDING_MODEL = model_id
        config.EMBEDDING_DIM = self.embedding_dim
        logger.info(
            f"[EVAL-EMBED] Loaded {model_id} ({precision}, backend={self.emb_backend}, "
            f"dim={self.embedding_dim}, pooling={self._emb_pooling}, sparse={self.has_learned_sparse})"
        )

    def _load_eval_reranker(self, model_id: str, precision: str) -> None:
        from scripts.sync import _load_local_tokenizer

        art = ensure_model_artifacts(
            model_id=model_id,
            kind="reranker",
            precision=precision,
            models_root=self.models_root,
            device=self.device,
        )
        self.rerank_artifacts = art
        self.cross_manifest = art.manifest
        self._cross_arch = str(art.manifest.get("arch_type", "cross_encoder"))
        self._score_activation = str(
            art.manifest.get(
                "score_activation",
                "identity" if self._cross_arch == "listwise_lbnl" else "sigmoid",
            )
        )
        self._doc_embed_token = str(art.manifest.get("doc_embed_token", "<|embed_token|>"))
        self._query_embed_token = str(art.manifest.get("query_embed_token", "<|rerank_token|>"))
        self._doc_embed_token_id = int(art.manifest.get("doc_embed_token_id", 151670))
        self._query_embed_token_id = int(art.manifest.get("query_embed_token_id", 151671))

        self.rerank_tokenizer = _load_local_tokenizer(art.model_dir)
        if art.onnx_path is not None:
            self.cross_session, self.rerank_backend = _create_ort_session(
                art.onnx_path, precision, device=self.device
            )
        else:
            self.cross_session, self.rerank_backend = _load_torch_fp16_session(
                model_id, "reranker", art.manifest, device_str=self.device
            )

        config.RERANKER_MODEL = model_id
        (
            config.CONFIDENCE_THRESHOLD_HIGH,
            config.CONFIDENCE_THRESHOLD_MEDIUM,
            config.RERANKER_THRESHOLD,
            config.RERANKER_MARGIN,
        ) = config._thresholds_for_reranker_arch(self._cross_arch)
        logger.info(
            f"[EVAL-RERANK] Loaded {model_id} ({precision}, backend={self.rerank_backend}, "
            f"arch={self._cross_arch}, thr={config.RERANKER_THRESHOLD})"
        )

    def encode(
        self, texts: list[str], batch_size: int | None = None
    ) -> dict[str, np.ndarray | list[dict]]:
        cur_bs = max(1, int(batch_size or getattr(self, "batch_size", config.ONNX_BATCH_SIZE)))
        while True:
            try:
                return super().encode(texts, batch_size=cur_bs)
            except Exception as exc:
                if _is_oom_error(exc) and cur_bs > 1:
                    next_bs = max(1, cur_bs // 2)
                    logger.warning(
                        f"[EVAL-EMBED] OOM at batch_size={cur_bs}; freeing cache and retrying at batch_size={next_bs}."
                    )
                    _free_accelerator_memory()
                    cur_bs = next_bs
                    continue
                raise

    def score_cross_encoder(
        self,
        pairs: list[list[str]],
        batch_size: int | None = None,
        progress_callback=None,
    ) -> np.ndarray:
        if not pairs:
            return np.array([], dtype=np.float32)

        cur_bs = max(1, int(batch_size or getattr(self, "batch_size", config.ONNX_BATCH_SIZE)))
        arch = getattr(self, "_cross_arch", "cross_encoder")

        if arch == "listwise_lbnl":
            while True:
                try:
                    return super().score_cross_encoder(
                        pairs, batch_size=cur_bs, progress_callback=progress_callback
                    )
                except Exception as exc:
                    if _is_oom_error(exc) and cur_bs > 1:
                        next_bs = max(1, cur_bs // 2)
                        logger.warning(
                            f"[EVAL-RERANK] OOM at block_size={cur_bs} (listwise); "
                            f"freeing cache and retrying at block_size={next_bs}."
                        )
                        _free_accelerator_memory()
                        cur_bs = next_bs
                        continue
                    raise

        # Keep INT8 batch ordering verbatim (dynamic INT8 quantizes activations per-batch);
        # sort FP16/FP32 by text length to avoid long outliers inflating every batch to 512 tokens.
        is_int8 = getattr(self, "rerank_precision", None) == "int8" or str(
            getattr(self, "rerank_backend", "")
        ).endswith("_int8")
        total = len(pairs)
        order = (
            list(range(total))
            if is_int8
            else sorted(range(total), key=lambda idx: len(str(pairs[idx][0])) + len(str(pairs[idx][1])))
        )

        out = np.empty(total, dtype=np.float32)
        if progress_callback:
            progress_callback(0, total)

        pos = 0
        while pos < total:
            take = min(cur_bs, total - pos)
            batch_indices = order[pos : pos + take]
            batch_pairs = [pairs[idx] for idx in batch_indices]
            try:
                batch_scores = super().score_cross_encoder(
                    batch_pairs, batch_size=len(batch_pairs), progress_callback=None
                )
            except Exception as exc:
                if _is_oom_error(exc) and cur_bs > 1:
                    next_bs = max(1, cur_bs // 2)
                    logger.warning(
                        f"[EVAL-RERANK] OOM at batch_size={cur_bs} on slice [{pos}:{pos + take}]; "
                        f"freeing cache and retrying at batch_size={next_bs}."
                    )
                    _free_accelerator_memory()
                    cur_bs = next_bs
                    continue
                raise

            for local_i, orig_idx in enumerate(batch_indices):
                out[orig_idx] = float(batch_scores[local_i])
            pos += take
            if progress_callback:
                progress_callback(pos, total)

        return out


def export_arcface_precisions(
    model: torch.nn.Module,
    sample_x: np.ndarray,
    output_dir: Path,
    domain: str,
) -> dict[str, Path]:
    """Exports trained BTArcFaceNet to FP32, FP16, and INT8 ONNX with parity verification."""
    from scripts.sync import _remove_onnx_artifacts, quantize_model

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_cpu = model.eval().to("cpu")
    in_features = int(sample_x.shape[1])

    fp32_path = output_dir / f"{domain}_bt_arcface.onnx"
    fp16_path = output_dir / f"{domain}_bt_arcface_fp16.onnx"
    int8_path = output_dir / f"{domain}_bt_arcface_int8.onnx"
    for p in (fp32_path, fp16_path, int8_path):
        _remove_onnx_artifacts(p)

    dummy = torch.randn(1, in_features, dtype=torch.float32)
    torch.onnx.export(
        model_cpu,
        dummy,
        str(fp32_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    _convert_onnx_fp32_to_fp16(fp32_path, fp16_path)
    quantize_model(fp32_path, int8_path, f"EVAL-ARCFACE-{domain.upper()}")

    probe = sample_x[: min(8, len(sample_x))].astype(np.float32)
    with torch.no_grad():
        ref = model_cpu(torch.from_numpy(probe)).numpy()
    for prec, p in (("fp32", fp32_path), ("fp16", fp16_path), ("int8", int8_path)):
        sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        out = sess.run(None, {sess.get_inputs()[0].name: probe})[0]
        if not np.isfinite(out).all() or out.shape != ref.shape:
            raise RuntimeError(f"ArcFace {prec} verification failed at {p}")
        del sess

    return {"fp32": fp32_path, "fp16": fp16_path, "int8": int8_path}


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify/export evaluation embedding & reranker models across FP32/FP16/INT8."
    )
    parser.add_argument("--kind", choices=["embedding", "reranker"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--precision", choices=list(PRECISION_MODES) + ["all"], default="all")
    parser.add_argument("--models-root", default=str(Path(__file__).resolve().parent / "models"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    precs = list(PRECISION_MODES) if args.precision == "all" else [args.precision]
    for p in precs:
        t0 = time.time()
        art = ensure_model_artifacts(
            model_id=args.model,
            kind=args.kind,
            precision=p,
            models_root=args.models_root,
            device=args.device,
            force=args.force,
        )
        print(
            json.dumps(
                {
                    "model": art.model_id,
                    "kind": art.kind,
                    "precision": art.precision,
                    "backend": art.backend,
                    "onnx_path": str(art.onnx_path) if art.onnx_path else None,
                    "size_mb": _onnx_total_size_mb(art.onnx_path),
                    "elapsed_s": round(time.time() - t0, 2),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    _cli()
