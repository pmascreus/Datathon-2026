# RawTree Challenge — Real-Time Recommendation

**Datathon FME 2026 · UPC Barcelona · 14–15 November 2026**

> Shoppers arrive at an online store every few seconds. For each one you have
> **3 seconds** to choose the **10 products** they will see. A hidden model
> simulates each shopper's decision to click and buy.

This repository is our submission, reorganised: **8 modules instead of 17**, the
same model, byte-identical config, and identical recommendations. It is a
structural refactor, not a new model — the evidence is in
[Verified, not assumed](#verified-not-assumed).

---

## The challenge

An online store sells **5,000 products** in ten categories. Its home page shows
a row of ten recommended products; today that row is just the **all-time best
sellers**. The task is to beat it, live, on shoppers arriving in real time.

Hard constraints, from the API contract:

| constraint | value |
|---|---|
| products per request | exactly **10**, distinct |
| must be | active **and** available at that moment |
| deadline | **3 s** (we self-impose **2.5 s**, and degrade earlier) |
| scored on | **purchases**, and position matters |
| cold start | new shoppers and brand-new products must still get a list |
| history | sessions, views, carts and purchases, before a temporal cut |
| forbidden | the `in_stock` field |

The leaderboard runs live on Saturday; on Sunday 10:00–12:00 CET the stream
switches to unseen data, the leaderboard goes dark, and the final ranking
appears at 12:00.

## The solution at a glance

```
request ──> Session ──> 11 channels ──> 350 candidates ──> 40 features
                                                               │
                                                               v
                                        LightGBM P(purchase) ──> blend 85/15
                                                               │
                                                               v
                             rules (re-purchase, tilt) ──> MMR ──> validate 10
                                                               │
                                                               v
                                                         submissions
```

One decision function, `src/policy.py:recommend()`, is called by both the live
loop and the offline evaluator. **What we measure offline is literally what
runs.**

### 1. From request to session

A request carries a user id, a device, a page, a region and a short trail of
recent views with timestamps. It becomes a `Session`: identity, context, and
time-decayed view weights. There is no heavyweight query per request — the
session is built from the row that was already pulled.

### 2. Candidate generation: 5,000 → 350, eleven ways

Studying all 5,000 products per shopper is impossible in the budget, so eleven
channels each propose their own head (top 80) and the union is capped at 350:

| channel | what it captures |
|---|---|
| **item-item ×3** | co-view, co-cart, co-purchase: "who looked at this also looked at that". A purchase weighs more than a cart, more than a view, and recent seeds weigh more than old ones |
| **ALS (48 factors)** | matrix factorisation over the sparse user × item matrix. Finds products nobody has viewed yet, because it matches **taste**, not co-occurrence |
| **content (TF-IDF → SVD)** | product text. TF-IDF up-weights words that distinguish a product; SVD compresses to 80 dimensions; cosine similarity says two products are alike. This is the cold-start backbone |
| **cold** | explicit boost for products created after the cut |
| **popularity** | all-time best sellers, the baseline to beat |
| **region trend** | what is rising in the shopper's region |
| **product trend** | last 15 min against the previous 3 h |
| **user profile** | long-term taste, 21-day half-life, purchases included |
| **category prior** | the category a shopper usually buys in |

Each channel is normalised to a **reciprocal rank** (1/(rank+1)) before being
weighted, so a channel with huge raw counts cannot drown the others. The union
keeps every channel's head plus the cold head, then fills to 350 by heuristic
score, and appends the always-valid fallback pool.

### 3. 40 features per candidate

For each of the 350 candidates we build 40 features: the raw and reciprocal-rank
score of every channel, popularity and trend counts, price against the median,
cold/new flags, bandit posteriors, how recently the shopper saw or bought it,
and request context (new user, device, page, region, hour).

### 4. The re-ranker

LightGBM models `P(purchase | impression)` for each candidate. Its output is
blended **85/15** with the normalised heuristic, so a candidate the ranker has
never seen cannot fall to the bottom by accident. Clicks are used as a secondary
signal but **never** as the target, because the competition is scored on
purchases.

### 5. Business rules and diversification

* **Re-purchase penalty**: a product bought moments ago is unlikely to be bought
  again. Up to **55%** off the score right after the purchase, decaying with a
  **25-hour half-life** (time constant 36 h), gone after the 7-day window.
* **Revenue tilt**: a small price preference, **0 in the deployed config**.
* **Thompson exploration**: off in the deployed config (`eps = 0`).
* **MMR diversification**: greedy maximal marginal relevance over the content
  embeddings, so the row is not ten near-identical products. **0 in the
  deployed config**, `0.06` on the degraded path.

### 6. Validation: never an invalid answer

The list is re-checked and repaired: exactly 10 distinct products, active now,
available now. Anything short is filled from the always-valid fallback pool
(live stock list → all-time best sellers → catalogue). A model bug cannot
produce an invalid row; it degrades to a less personal one.

### Cold start

* **New shopper, no history**: the current session becomes a provisional
  profile. We average its ALS factors and its text embeddings, and look up
  item-item neighbours. With no views at all, the ranker orders by popularity
  and regional trend adjusted by context.
* **New product**: it is embedded with the *already-fitted* vectoriser and SVD
  (`add_items`), so it is recommendable within one catalogue refresh (~30 s)
  without retraining. A dedicated channel proposes it by similarity to the
  session, and the bandit gives it its subcategory purchase rate as a prior,
  refined by feedback.

### Degradation

If the remaining budget cannot pay for the full model, the request is answered
with `p_fast`: the same personalised core (ALS, content, session, category,
MMR on the head) **without the ranker**, roughly 10x cheaper. If even that is
too slow, the always-valid fallback list answers instead. In the live round
this path handled **0.3%** of requests.

## Results

Offline, on a temporal cut (train strictly before the cut, evaluate after) so
there is no leakage. Every policy runs on the **same sessions**, which is what
makes the comparison meaningful.

**The phase ladder, 6,000 sessions, purchase hit@10:**

| policy | purchase hit@10 | click hit@10 | NDCG@10 |
|---|---:|---:|---:|
| `p0` all-time best sellers | 0.5446 | 0.4292 | 0.3038 |
| `p1` + item-item only | 0.4643 | 0.4203 | 0.2831 |
| `p2` + ALS + user history | **0.6429** | 0.5072 | 0.3774 |
| `p3` + content | 0.6190 | 0.5072 | 0.3971 |
| `p4` + LightGBM re-ranker | **0.6786** | **0.5556** | **0.5074** |
| `p_fast` (no ranker, deployed on lateness) | 0.6436 | 0.5021 | 0.3995 |

Two dips are real and worth stating: item-item **alone** loses to best sellers
(its neighbours are too narrow), and content does not lift purchases by itself —
it buys NDCG and cold coverage. Neither channel is removed on its own score; we
judge them in the blend.

**The gate that matters, 33,000 sessions:** `deployed` reaches **0.6892**
purchase hit@10 over **489 buyers**, and the candidate pool contains the
purchased product **98.9%** of the time.

**Negative results we keep on the record:** a click-labelled twin of the ranker
(2.5x more positives) lost **12.06 pp** of purchase hit; MMR at 0.16 was
negative on purchases; the revenue tilt moved nothing.

**The ranker, honestly:** the shipped booster has **one tree** — `416`
positives, `best_iteration: 1`. `valid_auc 0.94` flatters it; the honest number
is `valid_pr_auc 0.0155`, which against a `0.035%` base rate is a **45x lift**.
It adds measurable purchase hit (+6.0 pp from `p3` to `p4`, +1.0 pp over
`p_fast`) but it is a weak, fragile learner. Documented here, not hidden.

**Live timing:** compute p50 **20.6 ms**, p95 **206 ms**; end-to-end submission
p50 **780 ms** under a 2.5 s budget. The bottleneck is the write, not the model.

## Requirements

### Runtime

| item | value |
|---|---|
| Python | **3.10+** (developed and measured on 3.12.10) |
| RAM | **~0.5 GB** (measured: 409 MB with the bundle and the live snapshot loaded) |
| Disk | **48 MB** for the bundle, which is built locally and **not** in this repo + `state/` |
| Network | HTTPS to the RawTree API |

### Dependencies

```powershell
pip install -r requirements.txt
```

`numpy`, `pandas`, `scipy`, `scikit-learn` (TF-IDF/SVD), `lightgbm` (the
re-ranker), `implicit` (ALS), `requests`, `pytest`.

### Configuration

`.env` at the repo root — **never committed**:

```
TEAM_ID=team_NN
RAWTREE_API_KEY=rt_...
```

Model artefacts go in `models/<tag>/`. **The bundle is not committed**, because
`users.pkl` alone holds **59,149 real user ids** from the shop and publishing
third-party identities is not acceptable. Only `models/2026-11-13r/meta.json`
(cut, sizes, timings, metrics) travels with the code as evidence. Rebuild the
rest with `train.py` (needs `.env` and API access, a few minutes); `serve.py`
will not start without it.

### Non-negotiable operational rules

These are not style preferences. Each one has a failure behind it:

1. **One process per team.** `TEAM_ID` and the `submissions` table are shared.
   Two processes answering the same `request_id` is a **duplicate submission**.
   Never run this copy while the original is serving.
2. **Never run `evaluate.py`, `train.py` or `compare.py` while serving.** They
   compete for the same API quota; we measured coverage dropping to **~52%**
   when we did.
3. **Do not commit `.env`.** `.gitignore` covers it, but check before pushing.
4. **Keep the machine clock in sync.** Request timestamps come from the
   platform, so deadlines are in platform time. The code measures the offset,
   but a badly drifted clock still breaks the age estimates.
5. **Never use `in_stock`.** It was forbidden by the challenge.

## Run

```powershell
# Serve (needs .env; do not run alongside another team process)
python serve.py --models 2026-11-13r --version versio4

# Dry run: build answers, submit nothing
python serve.py --models 2026-11-13r --duration 120 --dry-run

# Rebuild artefacts for a temporal cut
python train.py --until 2026-11-13 --tag 2026-11-13r

# Offline gate, same sample for every policy
python evaluate.py --models 2026-11-13r --cut 2026-11-13 --limit 6000 `
    --salt v1 --policies deployed p_fast p3_content --keep-rows

# Paired comparison (refuses mismatched cut/sample)
python compare.py reports/a.json reports/b.json

# Tests
python -m pytest tests -q
```

## Layout

```
serve.py        the live loop: pull -> recommend -> push, one thread
train.py        builds models/<tag>/ for a temporal cut
evaluate.py     offline gate: hit@10, NDCG, IPS/SNIPS, paired bootstrap
compare.py      paired comparison of two report files

src/config.py     every weight, deadline and switch; a policy is a config diff
src/data.py       HTTP client + all SQL + Source + DataFrame readers
src/store.py      the bundle: ALS, item-item, TF-IDF/SVD, users, bandit, ranker
src/retrieval.py  Session, 11 candidate channels, union + cap
src/ranking.py    40 features + the LightGBM re-ranker
src/state.py      LiveSnapshot + the background refresher
src/policy.py     recommend(): rank -> rules -> diversify -> validate -> fallback
src/common.py     parsing, percentiles, atomic box, clock measurement
```

Dependencies flow one way, no cycles:

```
data -> store -> retrieval -> ranking
                    \            /
                     policy -> serve
state -------------^
```

## Verified, not assumed

Against the original tree, same bundle (`2026-11-13r`), same 10 requests
(8 users with history, one cold user with a view, one with no history), each run
through **both** policy paths:

| check | result |
|---|---|
| `Config` defaults | **52 / 52 fields identical** |
| `Config.load()` with the deployed JSON | identical |
| `PHASE_POLICIES` | identical (`p0`…`p8`, `p_fast`) |
| candidate pool | identical in all 20 cases |
| recommended list | **identical in all 20 cases**, cold and empty included |
| ranker used | same on every case (yes on `deployed`, no on `p_fast`) |
| tests | **27 passed** |
| `src/common.py` vs `src/util.py` | **byte-identical** |

## What changed in this copy

Only where the code lives, plus three functions that had no caller:

* `rawtree_client` + `queries` + `sources` + `datasets` + `add_cold_flag`
  → `src/data.py` (a query, its parameters and its parsing change together).
* `als_model` + `itemitem` + `content` + `models` → `src/store.py` (one
  lifecycle: built for a cut, saved, loaded, read-only while serving).
* `features` → `state.py`, `candidates` → `retrieval.py`, `ranker` →
  `ranking.py`, `util` → `common.py`, `recommender` → `serve.py`,
  `eval_offline` + `evaluation` → `evaluate.py`.
* Removed because nothing called them: `ContentModel.similarity` and
  `.similarity_matrix` (MMR builds its own matrix inline, which is why it is
  fast), `ItemItemModel.coverage`, and the orphan `stale_state_s`.
* One thread instead of poller + 4 workers + writer. Lower burst ceiling,
  identical answers; average load is ~3 req/s against a ~33 req/s ceiling.

Already-saved bundles pickle their class paths (`src.als_model`, …); those
module names are registered as aliases in `store.py` so every validated artefact
loads byte for byte instead of being rebuilt.

## Not included

`reports/`, `docs/`, `scripts/*.ps1`, `starter/`, `notebooks/` and the four
analysis tools (`build_embeddings`, `diagnose_recall`, `retrain_ranker`,
`report_md`). The reasoning behind each rejected experiment lives in the
original tree.

**If this copy becomes the project**, transplant `scripts/promote.ps1`,
`scripts/watchdog.ps1` and `scripts/status.ps1` unchanged: without them there is
no one-line rollback, no restart on crash, and no health view. None of them
touches the model.
