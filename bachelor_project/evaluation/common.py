"""Shared helpers for the RAGAS evaluation workflows.

Imported by the collectors (HiRAG venv / ``src/.venv``) and by ``score.py``
(evaluation venv), so it only depends on the standard library, PyYAML and
(optionally) python-dotenv. It never imports ``hirag`` or ``ragas``.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

EVAL_DIR = Path(__file__).resolve().parent
BACHELOR_DIR = EVAL_DIR.parent
HIRAG_DIR = BACHELOR_DIR / "HiRAG"
NAIVE_RAG_DIR = BACHELOR_DIR / "naive_RAG"
RAGAS_DIR = BACHELOR_DIR / "ragas"
APP_DIR = BACHELOR_DIR.parent
SRC_DIR = APP_DIR.parent
ENV_DEV_FILE = APP_DIR / ".env-dev"

# Default prompt language of the RAGAS metric prompts.
DEFAULT_PROMPT_LANGUAGE = "english"

# Files of a HiRAG working dir that change during plain querying and are
# therefore excluded from the index fingerprint.
FINGERPRINT_EXCLUDED_FILES = frozenset({"kv_store_llm_response_cache.json"})


def ensure_import_paths() -> None:
    """Put the HiRAG and bachelor_project folders on ``sys.path``.

    The HiRAG scripts import ``hirag`` and ``llm_config`` as top-level
    modules, so both folders are added instead of importing via ``app.*``.
    """
    for path in (HIRAG_DIR, BACHELOR_DIR):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def load_env_dev(env_file: Path = ENV_DEV_FILE) -> bool:
    """Load ``src/app/.env-dev`` (overriding shell exports) if it exists.

    Mirrors ``hi_insert_run.load_env_dev``: values are sanitized because the
    file may carry CRLF line endings or quotes. A stale shell export of the
    gateway key would otherwise shadow the real key.

    Args:
        env_file: Path of the env file.

    Returns:
        True when the file was found and loaded.
    """
    if not env_file.is_file():
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:
        print(
            f"WARNING: python-dotenv is not installed; {env_file} was not loaded.",
            file=sys.stderr,
        )
        return False
    load_dotenv(env_file, override=True)
    for key, value in list(os.environ.items()):
        cleaned = value.strip().strip('"').strip("'").replace("\r", "")
        if cleaned != value:
            os.environ[key] = cleaned
    return True


def load_llm_config_module() -> Any:
    """Import the shared ``bachelor_project/llm_config.py`` module.

    Returns:
        The imported ``llm_config`` module (central model config loader).
    """
    ensure_import_paths()
    import llm_config  # noqa: PLC0415 - path is prepared at runtime

    return llm_config


def get_evaluation_config(
    config: dict[str, Any], overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Resolve the RAGAS evaluator settings from ``config.yaml``.

    The evaluator models are read from the ``evaluation`` section of
    ``bachelor_project/config.yaml`` and are deliberately independent of the
    answer-generation models. CLI overrides win over the file.

    Args:
        config: Parsed ``config.yaml`` content.
        overrides: Optional CLI overrides (``None`` values are ignored).

    Returns:
        Dict with ``llm_model``, ``ragas_model_name``, ``embedding_model``,
        ``max_tokens`` and ``prompt_language``.

    Raises:
        ValueError: If a required model setting is missing or invalid.
    """
    section = config.get("evaluation") or {}
    if not isinstance(section, dict):
        raise ValueError("config.yaml: 'evaluation' must be a mapping.")
    resolved: dict[str, Any] = {
        "llm_model": section.get("llm_model"),
        "ragas_model_name": section.get("ragas_model_name"),
        "embedding_model": section.get("embedding_model"),
        "max_tokens": section.get("max_tokens", 8192),
        "prompt_language": section.get("prompt_language", DEFAULT_PROMPT_LANGUAGE),
    }
    for key, value in (overrides or {}).items():
        if value is not None:
            resolved[key] = value

    missing = [
        key
        for key in ("llm_model", "ragas_model_name", "embedding_model")
        if not str(resolved.get(key) or "").strip()
    ]
    if missing:
        raise ValueError(
            "Evaluator model settings missing: "
            + ", ".join(missing)
            + ". Set them in the 'evaluation' section of "
            f"{BACHELOR_DIR / 'config.yaml'} or pass the matching CLI flags "
            "(--llm-model / --ragas-model-name / --embedding-model)."
        )
    for key in ("llm_model", "ragas_model_name", "embedding_model", "prompt_language"):
        resolved[key] = str(resolved[key]).strip()
    try:
        resolved["max_tokens"] = int(resolved["max_tokens"])
    except (TypeError, ValueError) as exc:
        raise ValueError("evaluation.max_tokens must be an integer.") from exc
    if resolved["max_tokens"] <= 0:
        raise ValueError("evaluation.max_tokens must be positive.")
    return resolved


def utc_timestamp() -> str:
    """Return the current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def compact_timestamp() -> str:
    """Return a local timestamp usable in directory names."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sha256_file(path: Path) -> str:
    """Return the sha256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_directory(
    directory: Path, excluded: Iterable[str] = FINGERPRINT_EXCLUDED_FILES
) -> dict[str, str]:
    """Hash every regular file of a directory (non-recursive).

    Args:
        directory: Directory to fingerprint (e.g. a HiRAG working dir).
        excluded: File names to skip.

    Returns:
        Mapping ``file name -> sha256``.
    """
    excluded_set = set(excluded)
    return {
        path.name: sha256_file(path)
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name not in excluded_set
    }


def code_fingerprint(paths: Iterable[Path]) -> dict[str, str | None]:
    """Hash source files whose exact version matters for reproducibility.

    ``bachelor_project`` is not tracked by git, so a git commit alone does not
    identify the HiRAG code that produced a run.

    Args:
        paths: Files to hash.

    Returns:
        Mapping ``path relative to bachelor_project -> sha256`` (``None`` if
        the file is missing).
    """
    result: dict[str, str | None] = {}
    for path in paths:
        try:
            key = str(path.resolve().relative_to(BACHELOR_DIR))
        except ValueError:
            key = str(path)
        result[key] = sha256_file(path) if path.is_file() else None
    return result


def git_info(path: Path) -> dict[str, Any]:
    """Return commit and dirty state of the git repository containing a path.

    Only ``rev-parse`` and ``status`` are used; remotes are never read (a
    remote URL may embed credentials).

    Args:
        path: Any path inside the repository.

    Returns:
        Dict with ``toplevel``, ``commit``, ``dirty_files`` and
        ``path_tracked`` (False when the path itself is untracked).
    """

    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(path), *args],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    toplevel = run("rev-parse", "--show-toplevel")
    if toplevel is None:
        return {
            "toplevel": None,
            "commit": None,
            "dirty_files": None,
            "path_tracked": None,
        }
    status = run("status", "--porcelain")
    tracked = run("ls-files", "--error-unmatch", str(path))
    return {
        "toplevel": toplevel,
        "commit": run("rev-parse", "HEAD"),
        "dirty_files": len(status.splitlines()) if status else 0,
        "path_tracked": tracked is not None,
    }


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    """Return installed distribution versions (``None`` if not installed)."""
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def json_safe(value: Any) -> Any:
    """Recursively replace NaN/inf floats by ``None`` for strict JSON."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, data: Any) -> None:
    """Atomically write pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(data), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp_path, path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON line and flush it to disk immediately."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(record), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_json(path: Path) -> Any:
    """Read a JSON file."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def normalize_id(value: Any) -> str:
    """Normalize an ADONIS ID for COMPARISON only.

    ADONIS model IDs appear braced (``{uuid}``) in some exports and bare in
    others. Stored records always keep the raw ID; this form is only used to
    match retrieved IDs against reference IDs.

    TODO: the chunk corpus (``chunks.jsonl``) uses yet another layout
    (``model_{uuid}``); unify the input data structure of all RAG systems
    instead of extending this normalization.
    """
    return str(value).strip().strip("{}").strip().lower()


def load_model_prices(prices_path: Path) -> dict[str, Any]:
    """Load per-model token prices (per 1M tokens) from a YAML file.

    Same format as ``HiRAG/model_prices.yaml``::

        currency: USD
        models:
          <deployment>: {input: <float>, output: <float>}

    Args:
        prices_path: Pricing file path.

    Returns:
        Dict with ``currency``, ``models`` and ``source``.

    Raises:
        FileNotFoundError: If the explicitly requested file does not exist.
        ValueError: If the file structure is invalid.
    """
    import yaml  # noqa: PLC0415 - optional at import time

    if not prices_path.is_file():
        raise FileNotFoundError(f"Pricing file not found: {prices_path}")
    with prices_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    models = data.get("models") or {}
    if not isinstance(models, dict):
        raise ValueError(f"{prices_path}: 'models' must be a mapping.")
    for name, entry in models.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"{prices_path}: price entry for {name!r} must be a mapping."
            )
    return {
        "currency": str(data.get("currency") or "USD"),
        "models": models,
        "source": str(prices_path),
    }


def compute_usage_costs(
    usage: dict[str, dict[str, dict[str, int]]], prices: dict[str, Any]
) -> dict[str, Any]:
    """Compute costs from provider-reported token usage.

    Args:
        usage: ``{"chat"|"embedding": {model: {"prompt_tokens",
            "completion_tokens", ...}}}``.
        prices: Output of :func:`load_model_prices`.

    Returns:
        Dict with ``currency``, ``per_model``, ``unpriced_models`` and
        ``total`` (sum of priced models only).
    """
    per_model: dict[str, dict[str, float | None]] = {}
    unpriced: set[str] = set()
    total = 0.0
    for kind_usage in usage.values():
        for model, counts in kind_usage.items():
            entry = prices["models"].get(model)
            if entry is None:
                unpriced.add(model)
                per_model.setdefault(
                    model, {"input": None, "output": None, "total": None}
                )
                continue
            input_cost = (
                counts.get("prompt_tokens", 0) * float(entry.get("input", 0.0)) / 1e6
            )
            output_cost = (
                counts.get("completion_tokens", 0)
                * float(entry.get("output", 0.0))
                / 1e6
            )
            current = per_model.setdefault(
                model, {"input": 0.0, "output": 0.0, "total": 0.0}
            )
            current["input"] = (current["input"] or 0.0) + input_cost
            current["output"] = (current["output"] or 0.0) + output_cost
            current["total"] = (current["total"] or 0.0) + input_cost + output_cost
            total += input_cost + output_cost
    return {
        "currency": prices["currency"],
        "pricing_file": prices.get("source"),
        "per_model": per_model,
        "unpriced_models": sorted(unpriced),
        "total": total,
    }
