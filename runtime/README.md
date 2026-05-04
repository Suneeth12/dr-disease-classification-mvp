# Runtime Directory

This directory is a local workspace for files needed or generated while running the app. Most contents are ignored by Git.

Download the model ZIP from:

https://drive.google.com/drive/folders/1OsLOgkkSU_drRmFESKNo6d0V2WPbBGlZ?usp=sharing

Recommended Drive layout:

```text
Google Drive folder/
  retina-models.zip
```

Best ZIP contents:

```text
  models/
```

If possible, keep this as a single ZIP in Drive so users can download it and unzip `models/` directly
into this `runtime/` folder. The E-REG recipe and thresholds are tracked in GitHub and should not be
part of the Drive ZIP.

Expected local subdirectories:

- `models/`: trained `.keras` model weights.
- `notebook-artifacts/ensemble/`: tracked current E-REG `final_ensemble_recipe.json`.
- `notebook-artifacts/thresholds/`: tracked current `*_thresholds.json` files.
- `data/uploads/`: uploaded or sample images used during local runs.
- `data/artifacts/`: generated review and explanation outputs.

The local app database is expected at `runtime/data/app.db` when created. Keep this README tracked so the placeholder directory is visible in GitHub.

Minimum runtime setup:

```text
runtime/
  models/
    attention.keras
    lesion.keras
    multiscale.keras
    patch_mil.keras
  notebook-artifacts/
    ensemble/final_ensemble_recipe.json
    thresholds/*_thresholds.json
```

The recipe and threshold JSON files are already included with the repository. Download/copy only the
heavy `.keras` model files from the shared artifact bundle before starting the backend.
