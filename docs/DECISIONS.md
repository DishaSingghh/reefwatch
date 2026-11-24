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

## 2026-11-12 — Moorea photos are for testing only
- **Decision:** the 2008–2010 Moorea photos are used only as real before/after test pairs. The point
  classifier trains on ReefNet alone; nothing is trained, tuned or calibrated on Moorea.
- **Why:** the photos are old, so they are not used to teach the model; they remain a valid record of
  what each quadrat looked like in its survey year, which is what a change test needs.

## 2026-11-12 — NOAA permanent sites become the main before/after test set
- **Decision:** test pairs come mainly from NOAA NCRMP permanent-site photos (2012–2025). Moorea stays
  as an optional extra check. Both are test only.
- **Why:** newer data; per-point labels with x/y positions; fixed sites revisited across years.
- **Caveat:** divers re-lay the tape from a permanent stake each visit, so photo N in two years shows
  roughly the same patch. Site-level cover change is the reliable comparison; photo pairs are approximate.

## 2026-11-24 — ReefNet labels mapped to the six functional groups
- **Data:** the full CSV has 5,935,899 points, 0 rejected at ingest. Broad categories: Algae 3,177,343;
  Hard corals 1,220,741; Abiotic background 1,132,663; Other biota 302,024; Unsorted 103,128.
  Official split: train 4,762,555, test 1,173,344. (The paper's 924,626 points are the coral
  benchmark subset.)
- **Decision:** the 50 most common labels (99.4% of points) are mapped one by one in
  `configs/taxonomy.yaml`; rarer labels use their broad category. Hard substrate joins sand and
  rubble as bare substrate; Millepora counts as hard coral; Octocorallia is soft coral; Zoantharia,
  Corallimorpharia, sponges, seagrass and other animals are `other`; Unsorted, Unknown, Shadow and
  Human objects are excluded.
- **Check on the full list (128 labels):** 0 unmapped. Water (1,825 points, under Abiotic background)
  was falling back to bare substrate; it is now excluded. Resulting shares: other_algae 47.3%,
  hard_coral 22.5%, sand_rubble 15.0%, cca 10.1%, other 3.3%, soft_coral 1.8%.
