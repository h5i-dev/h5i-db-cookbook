"""Helpers for where recipes keep their databases.

Each recipe gets its own database directory under ``data/dbs`` so notebooks
are independently re-runnable. ``fresh_db`` wipes any previous run first.

Note (WSL2 users): keep databases on ext4 (e.g. under your home directory),
not under ``/mnt/c`` — h5i-db's crash-safety guarantees rely on POSIX rename
and fsync semantics that drvfs does not provide.
"""

from __future__ import annotations

import shutil
from pathlib import Path

DB_ROOT = Path(__file__).resolve().parent.parent / "data" / "dbs"


def db_path(name: str) -> str:
    """Path for a named recipe database (created lazily by h5i_db)."""
    DB_ROOT.mkdir(parents=True, exist_ok=True)
    return str(DB_ROOT / f"{name}.db")


def fresh_db(name: str) -> str:
    """Like :func:`db_path`, but removes any existing database first."""
    path = Path(db_path(name))
    if path.exists():
        shutil.rmtree(path)
    return str(path)
