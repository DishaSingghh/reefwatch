# ReefWatch

Comparability-aware, abstaining change alerts for repeat coral-reef photo-surveys.

Given two photo-surveys of the same quadrat, ReefWatch returns one of:

| Output | Meaning |
|---|---|
| `LIKELY_CHANGE` | likely visible habitat change |
| `NO_CLEAR_CHANGE` | no clear visible change |
| `REVIEW` | evidence insufficient (image quality, registration, uncertainty) |

It is decision support for image review, not an ecological diagnosis. See `CLAUDE.md` for scope.

**Project question.** Do image-comparability signals and calibrated abstention reduce false change
alerts compared with naively subtracting two cover estimates?

## Status

Early stage: repository scaffold, data ingestion next.

## Setup

```bash
uv sync --extra dev          # core + test/lint tools
uv run pre-commit install    # ruff + large-file guard on every commit
uv run pytest -q
```

Training (Colab/Kaggle): `uv sync --extra train`. Demo app: `uv sync --extra app`.

## Data

No data is committed. See [`data/README.md`](data/README.md) for download steps and
[`docs/DATA_LICENSES.md`](docs/DATA_LICENSES.md) for licences.

## Layout

```
configs/        taxonomy.yaml (single label source), data.yaml
src/reefwatch/  data, models, comparability, change, perturb, decision, evaluation
tests/          pytest suite (fast tests run in CI)
docs/           DECISIONS, TEST_PLAN, DATA_LICENSES
reports/        figures and tables generated from results/
app/            Streamlit demo
```

## Licence

Code: MIT. Data keeps its own licence (ReefNet annotations are CC BY-NC-SA 4.0, so no commercial use
of models trained on them).
