# Dynamic DR Grading

A full-stack clinical assist application for Diabetic Retinopathy (DR) grading. The system takes retinal fundus images, runs them through an ensemble of four specialized deep learning models, applies dynamic E-REG decision thresholds, and returns grade predictions (0 to 4) alongside Grad-CAM attention heatmaps.

The project consists of:
- **Backend**: FastAPI service managing model loading, ensemble inference, SQLite persistence, and Grad-CAM generation.
- **Frontend**: React + Vite interface for uploading fundus scans, inputting patient details, and inspecting predictions and heatmaps.
- **Runtime**: Local directory holding model weights, ensemble recipes, decision thresholds, uploads, and generated heatmaps.

---

## 1. Prerequisites

Make sure the following tools are installed:

- **Python 3.12.x** (verified on Python 3.12.10)
  > **Note**: Do not use Python 3.14. The backend pins `tensorflow==2.20.0`, which currently lacks pre-built wheels for Python 3.14 on Windows. Python 3.12 is the recommended version.
- **Node.js 20+** (Node 22 or 24 recommended)
- **Git**
- ~2 GB of free disk space for TensorFlow dependencies and model weights

Check your installed Python versions on Windows:
```powershell
py -0p
```

---

## 2. Download Model Weights

Trained model weights (`.keras` files) are stored externally to keep the git repository lightweight. Download `models.zip` (~1.2 GB) from Google Drive:

**[Google Drive Model Weights](https://drive.google.com/drive/folders/1zDQ9y3y0kpvL0yF6yjFyp2ehWYZ1HL4e)**

### Extraction Steps:
1. Download `models.zip` from the Drive link above.
2. Unzip `models.zip` directly into the `runtime/` directory.
3. Keep the file structure intact so the `.keras` files are placed inside `runtime/models/`:

```text
runtime/
└── models/
    ├── attention.keras
    ├── lesion.keras
    ├── multiscale.keras
    └── patch_mil.keras
```

> **PowerShell Unzip Example:**
> ```powershell
> # If models.zip is in your Downloads folder:
> Expand-Archive -Path "$HOME\Downloads\models.zip" -DestinationPath runtime\ -Force
> ```

> If your downloaded bundle also contains `multibranch.keras`, leave it inside `runtime/models/`. The active E-REG recipe uses the four core models listed above.

Create the remaining local runtime folders if they do not already exist:
```powershell
New-Item -ItemType Directory -Force runtime\models
New-Item -ItemType Directory -Force runtime\data\uploads
New-Item -ItemType Directory -Force runtime\data\artifacts
```

> **Note**: Do not modify or replace `runtime/notebook-artifacts/`. The ensemble recipe (`final_ensemble_recipe.json`) and per-model threshold files are tracked in Git and already in place.

---

## 3. Backend Setup

Open a PowerShell terminal in the project root:

```powershell
cd backend

# 1. Create a Python 3.12 virtual environment
py -3.12 -m venv .venv

# 2. Upgrade pip and install dependencies
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. Install the retina_api package in editable mode
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

Verify your virtual environment is using Python 3.12:
```powershell
.\.venv\Scripts\python.exe --version
# Output should show: Python 3.12.x
```

Start the API server:
```powershell
.\.venv\Scripts\python.exe -m uvicorn retina_api.app:app --host 0.0.0.0 --port 8000 --reload
```

Once running:
- **API Server**: `http://localhost:8000`
- **Interactive Swagger Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/api/v1/health`

> Initial startup may take 10-20 seconds while TensorFlow initializes and loads weights into memory. Check `/api/v1/health` to confirm `ready: true`.

---

## 4. Frontend Setup

Open a second terminal in the project root:

```powershell
cd frontend

# Install Node dependencies
npm install

# Start the Vite development server
npm run dev
```

The frontend will run at `http://localhost:5173`.

By default, the UI talks to `http://localhost:8000`. To point to a different API address, create a `.env` file in `frontend/`:
```env
VITE_API_BASE_URL=http://localhost:8000
```

---

## 5. How It Works

```
Retinal Fundus Image (.png/.jpg)
              │
              ▼
   ┌──────────────────────┐
   │ Preprocessing (224x) │
   └──────────┬───────────┘
              │
      ┌───────┴────────────────────────┐
      ▼               ▼                ▼               ▼
┌───────────┐   ┌───────────┐   ┌─────────────┐   ┌───────────┐
│ Attention │   │  Lesion   │   │ Multiscale  │   │ Patch MIL │
│   Model   │   │   Model   │   │    Model    │   │   Model   │
└─────┬─────┘   └─────┬─────┘   └──────┬──────┘   └─────┬─────┘
      │               │                │                │
      └───────┬───────┴────────────────┴────────────────┘
              ▼
   ┌────────────────────────────────────────┐
   │ Weighted Logit Fusion & E-REG Decision │
   └──────────────────┬─────────────────────┘
                      │
              ┌───────┴───────┐
              ▼               ▼
      Predicted Grade    Grad-CAM Heatmaps
         (0 to 4)        (Visual Explanations)
```

1. **Preprocessing**: Images are decoded, validated, and normalized to 224x224 RGB tensors.
2. **Model Predictions**: Four independent vision models evaluate the scan:
   - `attention`: Spatial self-attention network highlighting diffuse retinal changes.
   - `lesion`: Focuses on localized microaneurysms, hemorrhages, and exudates.
   - `multiscale`: Captures features across varying receptive field scales.
   - `patch_mil`: Multiple instance learning network evaluating high-resolution image patches.
3. **E-REG Thresholding**: Logits are weighted and combined against pre-computed decision thresholds to assign the final International Clinical Diabetic Retinopathy (ICDR) grade:
   - `0`: No DR
   - `1`: Mild DR
   - `2`: Moderate DR
   - `3`: Severe DR
   - `4`: Proliferative DR
4. **Grad-CAM Explanations**: Saliency heatmaps are generated for each active model and saved to `runtime/data/artifacts/` for clinical review.

---

## 6. Experimental Results

Evaluated on the APTOS 2019 Blindness Detection dataset under a split-safe protocol (70% train, 15% validation, 15% test; seed 42, 558 frozen test samples). All decision thresholds and ensemble recipes were tuned strictly on validation data before reporting frozen test metrics.

### Model-Backed Test Set Performance

| Model | Architecture Highlights | Accuracy | Precision | Recall | F1-Score | QWK |
| --- | --- | --- | --- | --- | --- | --- |
| **DB-AttnNet** | Dual-stream attention refinement | 91.58% | 0.8704 | 0.8482 | 0.9149 | 0.9472 |
| **DB-MScaleNet** | Multi-scale dilated context fusion | 86.92% | 0.7937 | 0.7907 | 0.8658 | 0.9376 |
| **DB-LesionNet** | Spatial lesion-focused gating | 90.14% | 0.8315 | 0.8618 | 0.9022 | 0.9545 |
| **DB-PatchMIL** | 9-patch bag + gated attention pooling | 89.78% | 0.8130 | 0.8457 | 0.9000 | 0.9482 |
| **E-REG Fusion** | **Dynamic 4-model reliability fusion** | **93.37%** | **0.9342** | **0.9337** | **0.9336** | **0.9602** |

### Per-Class Performance (E-REG Fusion)

| Grade | Clinical Category | Precision | Recall | F1-Score | Test Support |
| --- | --- | --- | --- | --- | --- |
| **0** | No DR | 0.9890 | 0.9926 | 0.9908 | 271 |
| **1** | Mild DR | 0.8333 | 0.8475 | 0.8403 | 59 |
| **2** | Moderate DR | 0.8896 | 0.9013 | 0.8954 | 152 |
| **3** | Severe DR | 0.8611 | 0.8857 | 0.8732 | 35 |
| **4** | Proliferative DR | 0.9444 | 0.8293 | 0.8831 | 41 |

### Comparison with Published DR Baselines

| Method / Study | Backbone / Approach | Accuracy | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- | --- |
| ViT-Large/32 *(Nazih et al., 2023)* | Vision Transformer | 80.40% | 0.8040 | 0.8040 | 0.8040 |
| MSAmix-Net *(Gao et al., 2024)* | Multiscale Self-Attention | 82.30% | 0.7300 | 0.9270* | 0.8164 |
| Hybrid Model *(Prakash & Vinoth, 2024)* | Feature-Level Fusion | 84.02% | 0.8312 | 0.8402 | 0.8291 |
| **E-REG Fusion (Ours)** | **Dual-Backbone + Patch-MIL + E-REG** | **93.37%** | **0.9342** | **0.9337** | **0.9336** |

*\*MSAmix-Net reported sensitivity on a merged dataset; table reflects published values mapped to comparable metric conventions.*

### Test Split Confusion Matrix (558 Cases)

| True \ Predicted | Grade 0 (No DR) | Grade 1 (Mild) | Grade 2 (Moderate) | Grade 3 (Severe) | Grade 4 (PDR) |
| --- | --- | --- | --- | --- | --- |
| **Grade 0 (No DR)** | **269** | 2 | 0 | 0 | 0 |
| **Grade 1 (Mild)** | 1 | **50** | 8 | 0 | 0 |
| **Grade 2 (Moderate)** | 2 | 7 | **137** | 4 | 2 |
| **Grade 3 (Severe)** | 0 | 0 | 4 | **31** | 0 |
| **Grade 4 (PDR)** | 0 | 1 | 5 | 1 | **34** |

---

## 7. Project Layout

```text
dr-disease-classification-mvp/
├── backend/
│   ├── pyproject.toml
│   ├── requirements.txt
│   ├── src/retina_api/
│   │   ├── app.py                     # FastAPI entry point & lifespan handler
│   │   ├── api/                       # HTTP routes, schemas, and serializers
│   │   ├── core/                      # App settings and path resolution
│   │   ├── db/                        # SQLite models (Case, Prediction, Artifact)
│   │   └── ml/                        # Preprocessing, ensemble, Grad-CAM, loader
│   └── tests/                         # Pytest test suite
├── frontend/
│   ├── src/
│   │   ├── api/                       # Backend client and TypeScript types
│   │   ├── app/                       # Main React shell and error boundaries
│   │   ├── features/cases/            # Upload, results, and case history views
│   │   └── styles/                    # Base CSS and layout styling
│   ├── package.json
│   └── vite.config.ts
├── runtime/                           # Local runtime artifacts (gitignored)
│   ├── models/                        # .keras weight files
│   ├── notebook-artifacts/            # Tracked E-REG recipe and threshold JSONs
│   └── data/                          # Uploaded images, heatmaps, and app.db
└── sample_data/                       # Sample fundus images for local testing
```

---

## 8. Testing & Verification

Run backend tests:
```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest
```

Run frontend linting and type checks:
```powershell
cd frontend
npm run lint
npm run build
```

---

## 9. API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api/v1/health` | Returns runtime readiness, device info, and loaded models |
| `POST` | `/api/v1/cases/predict` | Uploads an image, runs ensemble inference, and records a case |
| `GET` | `/api/v1/cases` | Lists all recorded cases ordered by visit date |
| `GET` | `/api/v1/cases/{id}` | Returns full case details, predictions, and heatmap URLs |
| `POST` | `/api/v1/cases/{id}/explainability` | Generates full Grad-CAM heatmaps across all models |

---

## 10. Environment Variables

Default paths resolve automatically relative to the repository root. You can customize them if needed:

| Variable | Default | Purpose |
| --- | --- | --- |
| `RETINA_RUNTIME_DIR` | `../runtime` | Root directory for runtime files |
| `RETINA_DATA_DIR` | `../runtime/data` | Directory for uploads, outputs, and database |
| `RETINA_MODELS_DIR` | `../runtime/models` | Directory where `.keras` files live |
| `RETINA_NOTEBOOK_ARTIFACTS_DIR` | `../runtime/notebook-artifacts` | Directory for recipe and threshold JSON files |
| `RETINA_DATABASE_URL` | `sqlite:///../runtime/data/app.db` | SQLAlchemy database connection string |
| `VITE_API_BASE_URL` | `http://localhost:8000` | Frontend backend API URL |

---

## 11. Troubleshooting

- **`No matching distribution found for tensorflow==2.20.0`**: You are likely using Python 3.14. Recreate your virtual environment with Python 3.12 (`py -3.12 -m venv .venv`).
- **`ModuleNotFoundError: No module named 'retina_api'`**: Make sure you ran `pip install --no-deps -e .` inside `backend/` so Python registers the local source package.
- **Backend reports missing models**: Verify that all four files (`attention.keras`, `lesion.keras`, `multiscale.keras`, `patch_mil.keras`) exist directly inside `runtime/models/`.
- **Health check stays in `loading`**: TensorFlow takes a few seconds to warm up and verify model weights on startup. Check the backend terminal logs for errors.
- **Frontend cannot connect**: Ensure the backend is running on `http://localhost:8000` and not blocked by local firewall settings.
