# Decisions log

One dated entry per decision. Never edit an old entry; add a new one that supersedes it.

## 2026-10-29 — Repository lives in project files until GitHub is connected
- **Decision:** scaffold built as a local git repo at `/mnt/project-files/reefwatch`.
- **Why:** no GitHub repository is connected to the project yet.
- **Next:** create `reefwatch` on GitHub, `git remote add origin ...`, push `main`.

## 2026-10-29 — Repository moved to GitHub
- **Decision:** remote `origin` = https://github.com/DishaSingghh/reefwatch; supersedes the entry above.

## 2026-10-29 — Python and tooling
- **Decision:** `requires-python >=3.11`, uv, ruff (lint + format), pytest, pre-commit, GitHub Actions.
- **Why:** Colab runs 3.11+; one tool each for env, lint and test keeps setup small.

## 2026-10-29 — Active path
- **Decision:** Path A (real MLC same-quadrat pairs), pending the quadrat-link check.
- **Quadrat-link check:** PASS if at least 100 quadrats appear in 2+ years with explicit IDs. Otherwise switch
  to Path B and record it here.
