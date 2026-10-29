# CLAUDE.md — ReefWatch

You are helping Disha build **ReefWatch**, a personal portfolio project (solo, B.Tech).
Read this file fully before every task. If a request conflicts with a rule here, stop and say so.

## 1. Project context

ReefWatch compares two coral-reef photo-surveys and returns one of three decision-support outputs:

- `LIKELY_CHANGE` — likely visible habitat change
- `NO_CLEAR_CHANGE` — no clear visible change
- `REVIEW` — evidence insufficient (image quality, registration, uncertainty)

Pipeline: point-centred patch classifier → cover estimate per image (share of points per class)
→ comparability features per pair → pair-level model → calibrated three-way decision with abstention.

Project question: do image-comparability signals and calibrated abstention reduce false change
alerts compared with naively subtracting two cover estimates?

Data:
- **ReefNet** (HF `ReefNet/ReefNet-1.0`, CC BY-NC-SA 4.0): sparse point labels, 39 hard-coral classes
  (arXiv 2510.16822 v3). Trains and benchmarks the patch classifier. Primary metric: macro recall.
  ReefNet has NO temporal task and NO capture dates. Never pair ReefNet images.
- **Moorea Labeled Corals (MLC)** (MCR LTER, CC BY 4.0): 2008–2010, 200 expert points per image,
  9 classes. Real same-quadrat pairs (Path A), matched only by explicit quadrat ID.
- **Red Sea Global / Al-Wajh test set** (HF zip): cross-source test only.
- **Synthetic stress test**: physics-based image degradations. Always labelled synthetic.

Active path: **Path A** unless `docs/DECISIONS.md` records a switch to Path B.

## 2. Scope boundaries

In scope: visible, image-derived cover of functional groups (L1) and genera (L2); comparability;
uncertainty; abstention; a Streamlit demo; a PDF report.

Out of scope (refuse and explain if asked): causal ecology claims (why a reef changed), reef health
scores, bleaching or disease diagnosis, species-level ID, real-time video, mobile apps, accounts or
payments, claims of conservation impact.

## 3. NEVER rules

1. **Never read, evaluate on, or tune with the test split** (`split == "test"`, the RSG set, or the
   held-out MLC fold) unless the task text literally says `FINAL TEST RUN` and
   `docs/TEST_PLAN.md` is committed. Hyperparameters come from `val`; temperatures and
   conformal thresholds come from `cal`.
2. **Never change `data/splits/TEST_HASH.txt`** or regenerate splits without an entry in
   `docs/DECISIONS.md` and an updated test plan.
3. **Never fabricate pairs.** A real pair needs the same `quadrat_id` in two `survey_year`s.
   Never pair by file order, similar names, or source names containing years.
4. **Never mix synthetic and real results** in one table without a `synthetic` column. Every
   synthetic pair has `is_synthetic = True` and a `perturb_params` record.
5. **Never define change labels from model predictions.** Ground-truth change comes from expert
   points only (see `src/reefwatch/change/pairs.py`).
6. **Never commit data**: no images, patches, Parquet shards, checkpoints, or W&B/MLflow run
   folders. Allowed in git: code, configs, `data/README.md`, split ID lists, `TEST_HASH.txt`,
   small CSV review sheets with IDs only.
7. **Never commit secrets** (CoralNet credentials, W&B keys). They live in `.env`, which is gitignored.
8. **Never silently drop rows.** Corrupt or rejected items stay in the manifest with a `status`.
9. **Never invent numbers, citations, or URLs** in docs, README, or reports. Unknown → write
   `[UNVERIFIED — check: <where>]`.
10. **Never weaken, skip, or delete a test to make CI pass.** Fix the code or ask.

## 4. Directory rules

```
reefwatch/
├── CLAUDE.md
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml              # uv-managed; Python 3.11
├── uv.lock
├── .pre-commit-config.yaml
├── .env.example                # names of secrets only
├── .github/
│   └── workflows/
│       └── ci.yml
├── configs/
│   ├── taxonomy.yaml           # single source of truth for labels (L1 + L2)
│   ├── data.yaml
│   ├── train_convnext.yaml
│   ├── probe_dinov2.yaml
│   ├── perturb.yaml
│   └── decision.yaml
├── data/                       # gitignored except README.md and splits/
│   ├── README.md
│   ├── raw/                    # downloads, untouched
│   ├── interim/                # ingest outputs, fetch logs
│   ├── patches/                # Parquet shards of JPEG bytes
│   ├── manifests/              # images.parquet, points.parquet
│   └── splits/                 # split ID lists + TEST_HASH.txt (committed)
├── docs/
│   ├── DECISIONS.md            # dated log of every design decision
│   ├── TEST_PLAN.md            # what to measure and expect; committed before the test run
│   ├── DATA_LICENSES.md        # per-source attribution and licenses
│   └── data_dictionary.md
├── notebooks/                  # thin Colab/Kaggle wrappers that call src/ — no logic here
├── src/
│   └── reefwatch/
│       ├── data/               # ingest_reefnet, select_subset, fetch_patches, ingest_mlc,
│       │                       # build_manifest, dedup, split, datasets, transforms
│       ├── models/             # backbones, train, embed, probe, calibrate
│       ├── comparability/      # quality, registration
│       ├── change/             # cover, pairs, pair_model
│       ├── perturb/            # formation_model
│       ├── decision/           # conformal, policy
│       └── evaluation/         # metrics, stats, run_eval, plots, report_pdf
├── app/
│   └── streamlit_app.py
├── tests/
├── reports/
│   ├── figures/
│   └── tables/
└── results/                    # gitignored run outputs
```

- Logic lives in `src/reefwatch/`. Notebooks only import and call it.
- One module per responsibility. New module → add tests in `tests/` in the same change.
- Paths come from configs, never hard-coded. Colab Drive and Kaggle paths are config overrides.
- Labels always go through `configs/taxonomy.yaml`; no label lists inside code.

## 5. Coding conventions

- Python 3.11, type hints on public functions, docstrings stating inputs, outputs, and units
  (cover is in **percentage points** 0–100 unless the name ends in `_frac`).
- Formatting and linting: `ruff check` and `ruff format` (black-compatible) via pre-commit.
- Tables: pandas + pyarrow; Parquet for data, CSV only for small human-review files.
- Read CSVs with `dtype=str`, then cast known columns explicitly.
- Randomness: every function that samples takes a `seed` argument; no global RNG use.
- Models: `timm` backbones, PyTorch AMP fp16 on T4, atomic checkpoint writes, resumable training.
- Metrics: macro recall = `sklearn.metrics.balanced_accuracy_score`; always report class count.
- Logging: `logging` module, not `print`, in `src/`.
- Keep functions small and pure where possible; I/O at the edges.

## 6. How to run things

```bash
uv sync                                  # install
uv run pre-commit install                # once
uv run pytest -q                         # all tests (must pass before every commit)
uv run pytest -q -m "not slow"           # fast subset used in CI
uv run ruff check . && uv run ruff format --check .
uv run python -m reefwatch.data.build_manifest --config configs/data.yaml
uv run python -m reefwatch.models.train --config configs/train_convnext.yaml
uv run streamlit run app/streamlit_app.py
```

Tests use tiny synthetic fixtures in `tests/fixtures/` — never real data, never network.

## 7. How to work with Disha

- Before coding, restate the task's goal, files, inputs, outputs, and acceptance tests in 3–6 lines.
- Make the smallest change that passes the acceptance tests. Do not refactor unrelated code.
- After coding: run `uv run pytest -q` and ruff, and report the exact output.
- If a requirement is ambiguous, pick the default from `docs/DECISIONS.md` or the guide, say which
  default you used, and continue.
- When you add a design decision, append a dated line to `docs/DECISIONS.md`.
- Explain non-obvious code briefly (why, not what). Disha is learning; teaching comments are welcome
  in tests and docs, short in code.
- This is a personal project. Word commits, docs, README and the app as a personal project, never
  as research (no "hypotheses", "experiments", "paper" framing). Push finished work to
  `DishaSingghh/reefwatch`.
