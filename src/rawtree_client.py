"""src/rawtree_client.py: Connexió compartida i thread-safe a RawTree."""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import requests


def _load_env() -> None:
    here = Path(__file__).resolve().parent.parent
    for folder in (Path("."), Path(".."), here):
        env_file = folder / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if not line.strip() or line.strip().startswith("#"):
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())
            return


_load_env()

API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
TEAM = os.environ.get("TEAM_ID", "")
_API_KEY = os.environ.get("RAWTREE_API_KEY", "")

_tls = threading.local()

Record = dict[str, Any]


def _session() -> requests.Session:
    if not hasattr(_tls, "http"):
        s = requests.Session()
        s.headers["Authorization"] = f"Bearer {_API_KEY}"
        _tls.http = s
    return _tls.http


def sql(q: str, timeout: int = 600) -> list[Record]:
    r = _session().post(f"{API}/v1/query", json={"sql": q}, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"{r.status_code} a la consulta:\n{q}\n\nResposta de RawTree:\n{r.text}")
    return r.json()["data"]


def sql_df(q: str, timeout: int = 600):
    import pandas as pd
    return pd.DataFrame(sql(q, timeout=timeout))


def insert(table: str, rows: list[Record]) -> None:
    if not rows:
        return
    _session().post(f"{API}/v1/tables/{table}", params={"database": TEAM}, json=rows, timeout=10).raise_for_status()