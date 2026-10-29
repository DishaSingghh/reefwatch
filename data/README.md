# data/

Nothing in this folder is committed except this README and the split ID lists in `splits/`
(enforced by `.gitignore` and the pre-commit large-file hook).

| Folder | Contents | How it is made |
|---|---|---|
| `raw/` | downloads, untouched | by hand (below) |
| `interim/` | cleaned Parquet tables | ReefNet / MLC ingest |
| `patches/` | 256 px point-centred crops in shards | patch cutting |
| `manifests/` | image and patch manifests with hashes | ingest and patch cutting |
| `splits/` | train/val/test ID lists + `TEST_HASH.txt` (tracked) | split step |

## Downloads

1. **ReefNet annotations**: <https://huggingface.co/datasets/ReefNet/ReefNet-1.0>
   → `raw/All_ReefNet_annotations.csv`, plus `ReefNet_labelmapping.xlsx`.
   Images are not on HF; they are crawled from CoralNet with the authors' crawler (needs a CoralNet
   account, credentials in `.env`).
2. **Red Sea Global test set**: `ReefNet_RSG-TestDataset_images.zip` from the same HF page → `raw/`.
3. **Moorea Labeled Corals**: EDI package `knb-lter-mcr.5006`
   (<https://portal.edirepository.org/nis/mapbrowse?packageid=knb-lter-mcr.5006.3>) → `raw/mlc/`.
   [UNVERIFIED — check: package revision number and file names on the EDI page.]

Record the download date and file sizes in `docs/DECISIONS.md`.
