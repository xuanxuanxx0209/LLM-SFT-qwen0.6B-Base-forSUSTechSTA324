#!/usr/bin/env python3
"""Shared path helpers for the local LLM workflows."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


DEFAULT_RUNTIME_ROOT = Path(os.environ.get("LLM_RUNTIME_ROOT", "/dev/shm/llm")).expanduser()
LEGACY_MODEL_ROOT = Path("/dev/shm/models")


def _unique_paths(candidates: list[Path]) -> tuple[Path, ...]:
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(candidate.expanduser())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(candidate.expanduser())
    return tuple(unique)


def runtime_root(_project_root: Path | None = None) -> Path:
    return Path(os.environ.get("LLM_RUNTIME_ROOT", str(DEFAULT_RUNTIME_ROOT))).expanduser()


def runtime_model_root(project_root: Path) -> Path:
    default_root = runtime_root(project_root) / "models"
    return Path(os.environ.get("LLM_MODEL_ROOT", str(default_root))).expanduser()


def runtime_data_root(project_root: Path) -> Path:
    default_root = runtime_root(project_root) / "data"
    return Path(os.environ.get("LLM_DATA_ROOT", str(default_root))).expanduser()


def runtime_result_root(project_root: Path) -> Path:
    default_root = runtime_root(project_root) / "result"
    return Path(os.environ.get("LLM_RESULT_ROOT", str(default_root))).expanduser()


def runtime_cache_root(project_root: Path) -> Path:
    default_root = runtime_root(project_root) / "cache"
    return Path(os.environ.get("LLM_CACHE_ROOT", str(default_root))).expanduser()


def model_search_roots(project_root: Path) -> tuple[Path, ...]:
    explicit_model_root = os.environ.get("LLM_MODEL_ROOT")
    candidates = [
        Path(explicit_model_root).expanduser() if explicit_model_root else runtime_model_root(project_root),
        LEGACY_MODEL_ROOT,
        Path.home() / "models",
        project_root / "models",
    ]
    return _unique_paths(candidates)


def has_model_files(path: Path) -> bool:
    return (path / "config.json").is_file() and (path / "tokenizer.json").is_file()


def find_model_dir(project_root: Path, candidate_names: tuple[str, ...]) -> Path | None:
    explicit_model_path = os.environ.get("MODEL_PATH")
    if explicit_model_path:
        explicit = Path(explicit_model_path).expanduser()
        if has_model_files(explicit):
            return explicit

    for root in model_search_roots(project_root):
        for name in candidate_names:
            candidate = root / name
            if has_model_files(candidate):
                return candidate
    return None


def infer_default_model_path(project_root: Path, candidate_names: tuple[str, ...]) -> Path:
    cwd = Path.cwd()
    if has_model_files(cwd):
        return cwd

    fallback = find_model_dir(project_root, candidate_names)
    if fallback is not None:
        return fallback

    return runtime_model_root(project_root) / candidate_names[0]


def resolve_llm_env_executable(name: str) -> str:
    env_key = f"LLM_ENV_{name.upper()}"
    override = os.environ.get(env_key)
    if override and os.access(override, os.X_OK):
        return override

    current_env_candidate = Path(sys.executable).resolve().with_name(name)
    candidates = [
        Path("/home/ubuntu/miniconda3/envs/llm/bin") / name,
        Path("/root/miniconda3/envs/llm/bin") / name,
        current_env_candidate,
    ]

    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)

    which_candidate = shutil.which(name)
    if which_candidate:
        return which_candidate

    return str(current_env_candidate)
