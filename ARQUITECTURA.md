# Com encaixen les peces

```
training/train_model.py        --until CUT   ->  models/*_CUT.pkl   (per a notebooks/evaluacio.ipynb)
training/train_model.py        (sense --until) -> models/*.pkl      (per a producció, src/recommender.py)
training/build_embeddings.py                  -> models/embeddings.npy, product_ids.pkl, vectorizer.pkl, svd.pkl

notebooks/exploracio.ipynb   -> només lectura, gràfiques i anàlisi (res que alimenti producció)
notebooks/evaluacio.ipynb    -> valida policies noves ABANS de tocar src/recommender.py

src/features.py       -> LiveState: estoc, tendència regional, productes nous. Es refresca
                          sol en un fil de fons (1 s estoc, 15 s tendències). Mai es consulta
                          dins del camí de resposta de 3 segons.
src/recommender.py    -> Models (carrega els .pkl un sol cop) + recommend(req, models, live)
                          + bucle pull/predict/push, igual que starter/recommender.py.
```

## Primer cop de tot
```bash
pip install -r starter/requirements.txt -r requirements-ml.txt

# 1. Exploreu l'històric
jupyter notebook notebooks/exploracio.ipynb

# 2. Entreneu amb un tall de validació i valideu una idea
python training/train_model.py --until 2026-11-10
jupyter notebook notebooks/evaluacio.ipynb   # heu d'escriure CUT = '2026-11-10' si no coincideix

# 3. Si la idea guanya: entreneu per a producció (sense --until) i calculeu embeddings
python training/train_model.py
python training/build_embeddings.py

# 4. Desplegueu
python src/recommender.py
```

## Regla d'or
Cap canvi a `recommend()` passa directament a `src/recommender.py`. Primer es prova
com a nova policy a `notebooks/evaluacio.ipynb`, es mira `SNIPS_ctr_slot1` (fiant-se'n
només si `slot1_matches` és prou gran), i només si guanya es porten els mateixos pesos
a `src/recommender.py`.
