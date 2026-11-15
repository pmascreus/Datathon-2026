# Recomanador en temps real — Datathon FME 2026 - AlphaTree Team

Cada segon entren 3 compradors a la botiga i tenim fins a tres segons per triar
els deu productes que veurà cadascú. Aquest document explica el repte proposat per Tinybird, com ho resolem,
què fa cada fitxer, i les versions que vam anar provant fins a la que vam
desplegar.

## El repte

La botiga ven uns 5.000 productes en deu categories. A la portada i a cada fitxa
de producte hi ha una fila de deu recomanacions, i avui aquesta fila mostra els
més venuts de tota la història de la botiga. L'objectiu és vendre'n més, en viu,
amb compradors reals arribant en temps real.

| restricció | valor |
|---|---|
| productes per petició | exactament **10**, sense repetir |
| condició | actius i **disponibles** en aquell moment |
| termini | **3 segons** (ens n'imposem 2,5 i degradem abans) |
| com es puntua | per **compres**, i la posició importa |
| cold start | usuaris nous i productes nous han de rebre llista igualment |
| prohibit | el camp `in_stock` |

Dissabte la classificació és pública; diumenge de 10:00 a 12:00 el flux passa a
dades que ningú ha vist, la classificació es tanca i a les 12:00 surt el rànquing
final.

## La solució

```
petició ──> Sessió ──> 11 canals ──> 350 candidats ──> 40 features
                                                           │
                                                           v
                                    LightGBM P(compra) ──> barreja 85/15
                                                           │
                                                           v
                        regles (recompra, ingrés) ──> MMR ──> valida 10
                                                           │
                                                           v
                                                     submissions
```

No podem estudiar 5.000 productes per a cada persona en tres segons. Ho fem en
dues etapes: primer generem 350 candidats amb onze estratègies, després els
ordenem amb un model que aprèn de compres reals del passat.

### 1. La sessió

La petició porta l'identificador d'usuari, el dispositiu, la pàgina, la regió i
les últimes visites amb la seva hora. Amb això construïm una `Session`: qui és,
en quin context arriba, i quines visites pesen més (les recents pesen més que les
antigues).

### 2. De 5.000 a 350, amb onze canals

Cada canal proposa els seus 80 millors i la unió es limita a 350:

| canal | què captura |
|---|---|
| **item-item ×3** | co-visita, co-cistell, co-compra: "qui ha mirat això també ha mirat allò". Una compra pesa més que un cistell, que una visita |
| **ALS (48 factors)** | factorització de la matriu usuari × producte. Troba productes que ningú ha vist, perquè compara **gust**, no co-ocurrència |
| **contingut (TF-IDF → SVD)** | el text del producte. TF-IDF dona pes a les paraules que distingeixen un producte; SVD comprimeix a 80 dimensions; el cosinus diu si dos productes s'assemblen. És la base del cold start |
| **cold** | reforç explícit per als productes creats després del tall |
| **popularitat** | els més venuts de sempre, el baseline a batre |
| **tendència regional** | què puja a la regió del comprador |
| **tendència de producte** | els últims 15 minuts contra les 3 hores anteriors |
| **perfil d'usuari** | gust a llarg termini, semivida de 21 dies, compres incloses |
| **prior de categoria** | la categoria on aquest comprador sol comprar |

Cada canal es normalitza a **rang recíproc** (1/(rang+1)) abans de ponderar-se,
de manera que un canal amb comptes molt grans no ofega els altres. La unió es
completa fins a 350 per puntuació heurística i s'hi afegeix la llista de reserva.

### 3. 40 característiques per candidat

Per a cada un dels 350 candidats calculem 40 valors: la puntuació de cada canal,
popularitat i tendència, preu contra la mediana, si és nou, les posteriors del
bandit, quan el va veure o comprar l'usuari, i el context de la petició (usuari
nou, dispositiu, pàgina, regió, hora).

### 4. El re-ranker

LightGBM modela `P(compra | impressió)` per a cada candidat. La seva sortida es
barreja **85/15** amb la puntuació heurística, perquè un candidat que el model no
ha vist mai no caigui al fons per accident. Els clics s'utilitzen com a senyal
secundari, però **mai** com a objectiu: la competició es puntua per compres.

### 5. Regles de negoci i diversificació

- **Penalització de recompra**: un producte comprat fa un moment és poc probable
  que es torni a comprar. Fins a un **55%** menys de puntuació just després de la
  compra, amb una **semivida de 25 hores**, i desapareix al cap de 7 dies.
- **Tilt d'ingrés**: una preferència petita pel preu. Val **0** a la versió
  desplegada.
- **Exploració Thompson**: apagada a la versió desplegada.
- **MMR**: diversificació per similitud de contingut, perquè la fila no siguin
  deu productes gairebé iguals. Val **0** a la versió desplegada i `0,06` al
  camí degradat.

### 6. Validació: mai una resposta invàlida

La llista es torna a comprovar i es repara: exactament 10 productes distints,
actius i disponibles ara mateix. Si en falta cap, s'omple des de la llista de
reserva (estoc en viu → més venuts → catàleg). Un error del model no pot produir
una fila invàlida; com a molt, produeix una fila menys personalitzada.

## Cold start

- **Comprador nou, sense historial**: la sessió actual fa de perfil provisional.
  Fem la mitjana dels seus factors ALS i dels embeddings de text, i busquem
  veïns a l'item-item. Si no ha vist res, el model ordena per popularitat i
  tendència regional ajustades pel context.
- **Producte nou**: s'incrusta amb el vectoritzador i l'SVD ja entrenats, així
  que és recomanable en menys d'un minut sense reentrenar. Un canal específic el
  proposa per similitud amb la sessió, i el bandit li dona com a punt de partida
  la taxa de compra de la seva subcategoria, que s'afina amb el feedback.

## Quan no hi ha temps

Si el pressupost restant no pot pagar el model complet, la petició es respon amb
`p_fast`: el mateix nucli personalitzat (ALS, contingut, sessió, categoria, MMR
al cap) **sense el ranker**, unes deu vegades més barat. Si ni així s'hi arriba,
respon la llista de reserva, que és vàlida per construcció. A la ronda en viu
aquest camí va atendre el **0,3%** de les peticions.

## El codi

```
serve.py          el bucle en viu: pull → recommend → push, un sol fil
train.py          construeix models/<tag>/ per a un tall temporal
evaluate.py       porta offline: hit@10, NDCG, IPS/SNIPS, bootstrap aparellat
compare.py        comparació aparellada de dos informes

src/config.py     tots els pesos, terminis i interruptors; una política és un diff
src/data.py       client HTTP + tot el SQL + Source + lectors de dades
src/store.py      el bundle: ALS, item-item, TF-IDF/SVD, usuaris, bandit, ranker
src/retrieval.py  Session, els 11 canals, unió i límit
src/ranking.py    les 40 features i el re-ranker LightGBM
src/state.py      LiveSnapshot i el fil de refresc en segon pla
src/policy.py     recommend(): ordena → regles → diversifica → valida → reserva
src/common.py     parsing, percentils, caixa atòmica, mesura del rellotge
```

Les dependències van en una sola direcció, sense cicles:

```
data -> store -> retrieval -> ranking
                    \            /
                     policy -> serve
state -------------^
```

**Una sola funció de decisió.** `recommend()` és la que crida tant el bucle en viu
com l'avaluador offline. El que mesurem és exactament el que corre.

## Les versions

Cada pas de la escala afegeix una sola peça, i tots es mesuren sobre les mateixes
sessions, així que una diferència no pot venir de tenir sessions fàcils. Sobre
6.000 sessions, compra hit@10:

| versió | què afegeix | compra hit@10 | NDCG@10 |
|---|---|---:|---:|
| `p0` | els més venuts de sempre | 0,5446 | 0,304 |
| `p1` | item-item | 0,4643 | 0,283 |
| `p2` | ALS + historial d'usuari | **0,6429** | 0,377 |
| `p3` | contingut | 0,6190 | 0,397 |
| `p4` | re-ranker LightGBM | **0,6786** | **0,507** |
| `p_fast` | sense ranker, per al camí degradat | 0,6436 | 0,400 |

Hi ha dos sots i són reals: l'item-item **sol** perd contra els més venuts
(els seus veïns són massa estrets), i el contingut no apuja la compra pel seu
compte — compra NDCG i cobertura de productes nous. Cap canal es treu pel seu
resultat individual; es jutgen a la barreja.

Més enllà de `p4`, a la mostra de 5.000: tendències `0,6786`, cold start
`0,6786`, i MMR a 0,16 **`0,6071`** — set punts per sota. Per això al desplegament
l'MMR va a zero.

### Què vam provar i vam descartar

- **MMR a 0,16**: negatiu en compra (0,6071 contra 0,6786). A 0,06 es queda només
  al camí degradat.
- **Tilt d'ingrés**: no movia la compra i costava 1,3 punts de click. A zero.
- **Ranker amb etiqueta de click**: té 2,5 vegades més positius i va perdre
  **12,06 punts** de compra. L'objectiu és la compra.
- **Compres recents com a llavors de candidats**: no va superar la prova.
- **Feed de compres en viu**: apagat; `feedback` no porta `user_id`, així que el
  senyal és massa sorollós per al que aporta.

### El desplegat

`versio4`, amb el bundle `2026-11-13r` (tall del 13 de novembre). Els bundles
`2026-10-23`, `2026-11-07` i `2026-11-07r` són les iteracions anteriors.

Una cosa que val més dir-la que amagar-la: el ranker desplegat té **un sol
arbre**. Es va entrenar amb 416 compres sobre 1,2 milions de files i l'early
stopping es va aturar a la primera ronda. L'AUC de validació (0,94) enganya; el
número honest és el PR-AUC de 0,0155, que contra una taxa base del 0,035% és un
**lift de 45 vegades**. Aporta guany mesurat (+6,0 punts de `p3` a `p4`, +1,0
sobre `p_fast`), però és un aprenent dèbil i fràgil, i canviar-lo exigeix una
prova aparellada, no una intuïció.

## Resultats

Sobre 33.000 sessions amb el tall temporal: **0,6892** de compra hit@10 amb **489
compradors**, i el producte comprat és dins del pool de 350 candidats el
**98,9%** de les vegades.

En viu: càlcul p50 **20,6 ms** i p95 **206 ms**; enviament de punta a punta p50
**780 ms** dins d'un pressupost de 2,5 s. El coll d'ampolla és l'escriptura, no
el model.

## Requisits

| | valor |
|---|---|
| Python | **3.10+** (mesurat a 3.12.10) |
| RAM | **~0,5 GB** (409 MB amb el bundle i l'snapshot carregats) |
| Disc | **48 MB** el bundle, que es construeix en local i **no** és al repo |
| Xarxa | HTTPS cap a l'API de RawTree |

```powershell
pip install -r requirements.txt
```

`numpy`, `pandas`, `scipy`, `scikit-learn`, `lightgbm`, `implicit`, `requests`,
`pytest`.

Cal un `.env` a l'arrel, que **no es committeja mai**:

```
TEAM_ID=team_NN
RAWTREE_API_KEY=rt_...
```

Els artefactes van a `models/<tag>/`. No es publica el bundle perquè `users.pkl`
conté **59.149 identificadors reals** de compradors; al repo només hi ha
`meta.json` com a testimoni del tall. Es reconstrueix amb `train.py`.

## Executar-ho

```powershell
# Servei (necessita .env; no el posis en marxa amb un altre procés de l'equip)
python serve.py --models 2026-11-13r --version versio4

# Prova en sec: construeix respostes i no envia res
python serve.py --models 2026-11-13r --duration 120 --dry-run

# Reconstruir els artefactes per a un tall
python train.py --until 2026-11-13 --tag 2026-11-13r

# Porta offline, la mateixa mostra per a totes les polítiques
python evaluate.py --models 2026-11-13r --cut 2026-11-13 --limit 6000 `
    --salt v1 --policies deployed p_fast p3_content --keep-rows

# Comparació aparellada (refusa mostres o talls diferents)
python compare.py reports/a.json reports/b.json

# Tests
python -m pytest tests -q
```
