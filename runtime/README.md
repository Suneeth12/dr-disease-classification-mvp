# Runtime Directory

This folder holds local runtime data, trained model weights, uploaded images, and generated Grad-CAM visualizations.

---

## Model Weights Download

Model weights (`.keras` files) are stored externally in `models.zip` (~1.2 GB) to keep the repository size manageable:

**[Google Drive Model Weights](https://drive.google.com/drive/folders/1zDQ9y3y0kpvL0yF6yjFyp2ehWYZ1HL4e)**

### How to set up:
1. Download `models.zip` from the Google Drive link.
2. Unzip it directly into this `runtime/` folder.
3. Keep the file structure intact so the `.keras` files reside inside `runtime/models/`.

---

## Directory Structure

```text
runtime/
├── models/                               # Downloaded .keras weights (gitignored)
│   ├── attention.keras
│   ├── lesion.keras
│   ├── multiscale.keras
│   └── patch_mil.keras
│
├── notebook-artifacts/                   # Tracked in Git (do not delete)
│   ├── ensemble/
│   │   └── final_ensemble_recipe.json    # Active E-REG ensemble configuration
│   └── thresholds/
│       ├── attention_thresholds.json
│       ├── lesion_thresholds.json
│       ├── multiscale_thresholds.json
│       └── patch_mil_thresholds.json
│
└── data/                                 # Generated at runtime (gitignored)
    ├── uploads/                          # Temporary storage for uploaded scans
    ├── artifacts/                        # Generated Grad-CAM overlay images
    └── app.db                            # SQLite application database
```

---

## Quick Notes

- The recipe and threshold JSON files in `notebook-artifacts/` are tracked in version control. You do not need to download or recreate them.
- If you need to re-initialize your local database or clear generated artifacts, you can safely delete `data/app.db`, `data/uploads/*`, and `data/artifacts/*`. The backend will recreate them on the next run.
