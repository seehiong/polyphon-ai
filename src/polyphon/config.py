"""Environment configuration loading for Polyphon.

Loads `.env` once at import time so that every entry point -- the CLI, the
server, and library use via `import polyphon` -- sees the same configuration.
Real environment variables always win over `.env`, so CI and container
deployments can override without editing files.
"""

from __future__ import annotations

import os
from pathlib import Path

_LOADED = False


def find_env_file(start: Path | None = None) -> Path | None:
    """Locate the nearest .env, walking up from `start` to the filesystem root,
    falling back to user-global configuration locations (~/.polyphon/.env, ~/.config/polyphon/.env).

    Walking up means `polyphon` works from any subdirectory of the project,
    not only from the directory holding the .env file.
    """
    override = os.environ.get("POLYPHON_ENV_FILE")
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_file() else None

    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate

    # Fallback to user-global locations for PyPI / system-wide installations
    for global_path in (Path.home() / ".polyphon" / ".env", Path.home() / ".config" / "polyphon" / ".env"):
        if global_path.is_file():
            return global_path

    return None


def parse_env_file(text: str) -> dict[str, str]:
    """Parse dotenv-style `KEY=value` lines, ignoring blanks and comments."""
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if not key:
            continue
        value = value.strip()
        # Strip one layer of matching quotes; leave inner quotes intact.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
    return values


def load_env(force: bool = False) -> Path | None:
    """Load `.env` into os.environ without clobbering existing variables.

    Returns the file that was loaded, or None. Safe to call repeatedly; the
    work only happens once unless `force` is set.
    """
    global _LOADED
    if _LOADED and not force:
        return None

    env_file = find_env_file()
    if env_file is None:
        _LOADED = True
        return None

    try:
        parsed = parse_env_file(env_file.read_text(encoding="utf-8"))
    except OSError:
        _LOADED = True
        return None

    for key, value in parsed.items():
        # Real environment variables take precedence over the file.
        os.environ.setdefault(key, value)

    _LOADED = True
    return env_file
