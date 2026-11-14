"""Calcula un embedding de cold-start per a cada producte del catàleg, a partir del
seu text (category + subcategory + brand + description). Els productes nous que
apareixen durant l'esdeveniment només tenen aquest text (cap historial), així que
això és el que permet a src/recommender.py situar-los a prop de productes similars
des del primer minut.

    python training/build_embeddings.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.rawtree_client import sql

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

N_COMPONENTS = 100  # mida de l'embedding; amb ~5.000 productes, 100 és de sobres


def fetch_catalog() -> list[dict]:
    """Catàleg actual: shop.catalog amb l'última shop.catalog_updates de cada
    producte per sobre (mateix patró que queries.sql #1)."""
    return sql("""
        SELECT product_id,
               argMax(category, v)    AS category,
               argMax(subcategory, v) AS subcategory,
               argMax(brand, v)       AS brand,
               argMax(description, v) AS description
        FROM (
            SELECT product_id::String AS product_id, category::String AS category,
                   subcategory::String AS subcategory, brand::String AS brand,
                   description::String AS description, '' AS v
            FROM shop.catalog
            UNION ALL
            SELECT product_id::String, category::String, subcategory::String, brand::String,
                   description::String, toString(ts)
            FROM shop.catalog_updates
        )
        GROUP BY product_id""")


def main() -> None:
    print("descarregant catàleg...")
    catalog = fetch_catalog()
    print(f"  {len(catalog)} productes")

    product_ids = [row["product_id"] for row in catalog]
    texts = [
        " ".join(filter(None, [row.get("category", ""), row.get("subcategory", ""),
                                row.get("brand", ""), row.get("description", "")]))
        for row in catalog
    ]

    print("vectoritzant (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=20_000, stop_words="english")
    tfidf = vectorizer.fit_transform(texts)

    print(f"reduint a {N_COMPONENTS} dimensions (SVD)...")
    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=0)
    embeddings = svd.fit_transform(tfidf)
    # normalitzem perquè el producte escalar sigui directament similitud del cosinus
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-9, None)

    np.save(MODELS_DIR / "embeddings.npy", embeddings)
    with open(MODELS_DIR / "product_ids.pkl", "wb") as f:
        pickle.dump(product_ids, f)
    with open(MODELS_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MODELS_DIR / "svd.pkl", "wb") as f:
        pickle.dump(svd, f)
    print(f"guardat embeddings {embeddings.shape} a {MODELS_DIR}")


def embed_new_product(category: str, subcategory: str, brand: str, description: str) -> np.ndarray:
    """Vectoritza UN producte nou (p.ex. arribat via catalog_updates) fent servir el
    vectoritzador + SVD ja guardats, sense reentrenar res."""
    with open(MODELS_DIR / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(MODELS_DIR / "svd.pkl", "rb") as f:
        svd = pickle.load(f)
    text = " ".join(filter(None, [category, subcategory, brand, description]))
    vec = svd.transform(vectorizer.transform([text]))[0]
    return vec / max(np.linalg.norm(vec), 1e-9)


if __name__ == "__main__":
    main()
