from __future__ import annotations

import os
from pathlib import Path
from typing import Any


_ENV_FILENAME = "mindpilot.env"
_ENV_LOADED = False


def _expand(value: str) -> str:
    return os.path.expanduser(os.path.expandvars(value))


def load_mindpilot_env(env_filename: str = _ENV_FILENAME) -> Path | None:
    """Load key/value pairs from the repo-local env file into os.environ."""
    global _ENV_LOADED

    env_path = Path(__file__).resolve().with_name(env_filename)
    if not env_path.exists():
        _ENV_LOADED = True
        return None

    raw_values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]

        raw_values[key] = value

    for key, value in raw_values.items():
        os.environ.setdefault(key, value)

    for _ in range(5):
        changed = False
        for key in raw_values:
            expanded = _expand(os.environ[key])
            if expanded != os.environ[key]:
                os.environ[key] = expanded
                changed = True
        if not changed:
            break

    _ENV_LOADED = True
    return env_path


def _ensure_env_loaded() -> None:
    if not _ENV_LOADED:
        load_mindpilot_env()


def get_env_value(name: str, default: str | None = None, required: bool = False) -> str | None:
    _ensure_env_loaded()
    value = os.environ.get(name, default)
    if value is None:
        if required:
            raise KeyError(f"Missing required environment variable: {name}")
        return None
    return _expand(value)


def get_project_root() -> Path:
    _ensure_env_loaded()
    raw_root = os.environ.get("PROJECT_ROOT")
    if raw_root:
        root = Path(_expand(raw_root))
        if not root.is_absolute():
            root = (Path(__file__).resolve().parent / root).resolve()
        return root
    return Path(__file__).resolve().parent


def resolve_config_values(value: Any) -> Any:
    _ensure_env_loaded()
    if isinstance(value, dict):
        return {key: resolve_config_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_config_values(item) for item in value]
    if isinstance(value, tuple):
        return tuple(resolve_config_values(item) for item in value)
    if isinstance(value, str):
        return _expand(value)
    return value


load_mindpilot_env()