"""Connexió compartida a RawTree, reutilitzada per training/ i per src/.

Llegeix TEAM_ID i RAWTREE_API_KEY de .env (arrel del repo) o de l'entorn. Es copia el
patró exacte de starter/recommender.py, però en un sol lloc perquè no calgui repetir-lo
a cada script.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests


def _load_env() -> None:
    here = Path(__file__).resolve().parent.parent  # arrel del repo (un nivell amunt de src/)
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

_http = requests.Session()
_http.headers["Authorization"] = f"Bearer {os.environ.get('RAWTREE_API_KEY', '')}"

Record = dict[str, Any]
"""RawTree retorna files com a objectes JSON: {"columna": valor, ...}"""


def sql(q: str, timeout: int = 600) -> list[Record]:
    """Consulta de lectura. Retorna una llista de dicts (una fila cada un)."""
    r = _http.post(f"{API}/v1/query", json={"sql": q}, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"{r.status_code} a la consulta:\n{q}\n\nResposta de RawTree:\n{r.text}")
    return r.json()["data"]


def sql_df(q: str, timeout: int = 600):
    """Igual que sql(), però com a pandas DataFrame (import diferit perquè els scripts
    de training/ que no necessiten pandas no l'hagin de tenir instal·lat)."""
    import pandas as pd
    return pd.DataFrame(sql(q, timeout=timeout))


def insert(table: str, rows: list[Record]) -> None:
    """Insereix en lot a una taula team_NN (p.ex. 'submissions' o 'scratch')."""
    if not rows:
        return
    _http.post(f"{API}/v1/tables/{table}", params={"database": TEAM}, json=rows, timeout=10).raise_for_status()
