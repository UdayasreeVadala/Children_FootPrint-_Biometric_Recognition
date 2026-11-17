# FootprintID

A FastAPI + TensorFlow application for biometric recognition of children from
plantar-footprint images. It implements the full pipeline requested for the
project:

1. **ROI detection** — robust footprint segmentation with multiple threshold
   candidates, skin-colour masking, and a GrabCut fallback.
2. **Gabor filtering** — CLAHE contrast enhancement followed by a six-orientation
   Gabor filter bank fused by maximum response.
3. **Confidence and rejection** — every prediction returns a confidence score,
   top-three candidates, and an **Unknown Child** result below the configured
   threshold (70% by default).
4. **Enrollment flow** — drag-and-drop enrollment from the browser, preprocessing
   previews, metadata storage, and enrollment-based retraining.
5. **Explainability** — Grad-CAM overlays for both softmax identity models and
   embedding-similarity verification.

The interface also includes a **Model Lab** tab that trains and compares three
architectures on the bundled dataset: MobileNetV2, EfficientNet-B0, and a custom
CNN.

---

## Quick start

Python 3.10 or 3.11 is required (TensorFlow 2.15 does not support Python 3.12+).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Optional but recommended: train the three models on the bundled dataset
python train.py

# Optional: calibrate embedding-similarity confidence from the trained models
python calibrate.py

# Start the web application
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`. The lightweight health check is
`http://127.0.0.1:8000/api/health` (also exposed as `/health` for uptime
probes).

The application starts even before training: it falls back to a clearly labelled
demo mode. Once `models/registry.json` exists, the best trained model is loaded
automatically.

---

## Dataset

The bundled `Training_Data` directory contains one child (`B003`) captured
across ten visits with four acquisition modalities:

```text
Training_Data/
└── B003/
    ├── Visit 1/
    │   ├── Camera/Left/*.JPG      # 17 / 15 images
    │   ├── Camera/Right/*.JPG
    │   ├── Paper/Left/*.tif       # 4 / 4 images
    │   ├── Paper/Right/*.tif
    │   ├── Scanner/Left/*.bmp     # 7 / 7 images
    │   └── Scanner/Right/*.bmp
    ├── Visit 2/ … Visit 4/        # same sensor layout
    └── Visit 5/ … Visit 10/
        ├── Left/*.jpeg            # phone photographs
        └── Right/*.jpeg
```

- **149 images** from `Camera`, `Paper`, and `Scanner` are used by default.
- **Phone photographs are excluded by default** because their cluttered
  background makes the plantar ROI unreliable. They can be included with
  `--include-phone` (training) or the “Include phone photos” checkbox in the UI.
- The dataset contains **only one child**. A true multi-child identity classifier
  therefore cannot be trained from it. See
  [Single-child verification mode](#single-child-verification-mode).

### Training command

```powershell
python train.py
# or, for a quick smoke test:
python train.py --architectures custom_cnn --epochs 3 --fine-tune-epochs 0
```

Useful options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--architectures` | `mobilenetv2,efficientnetb0,custom_cnn` | Models to train. |
| `--task` | `auto` | `auto`, `identity`, `side`, `visit`, or `sensor`. |
| `--epochs` | `18` | Head-training epochs. |
| `--fine-tune-epochs` | `6` | Backbone fine-tuning epochs for transfer models. |
| `--batch-size` | `16` | Training batch size. |
| `--include-phone` | off | Include the noisy phone photographs. |
| `--max-images` | all | Cap the number of images for quick experiments. |
| `--force-preprocess` | off | Rebuild the preprocessing cache. |

Training writes:

```text
models/
├── registry.json                       # architectures, metrics, active model, calibration
├── mobilenetv2_side.keras              # one model per architecture
├── efficientnetb0_side.keras
├── custom_cnn_side.keras
├── mobilenetv2_side_metrics.json
├── mobilenetv2_side_history.json
├── mobilenetv2_side_training.png
├── mobilenetv2_side_confusion_matrix.png
├── calibration_report.json             # written by calibrate.py
├── footprint_model.keras               # alias of the best model
├── footprint_model_classes.json
└── footprint_model_meta.json
```

The preprocessing cache lives in `data/cache/dataset_224/` and is reused on
later runs unless `--force-preprocess` is supplied.

---

## Single-child verification mode

A softmax identity classifier needs at least two children. Because the bundled
dataset has one, the application chooses a **Left vs Right foot-side** task
automatically (`resolve_task`) and trains all three architectures on that label.
The model's 256-D penultimate layer is then reused for **child verification**:

- **Softmax mode** — used when two or more child identities are enrolled/trained.
  Confidence is the softmax probability of the winning child.
- **Embedding verification mode** — used for the single-child dataset. The query
  embedding is compared with the mean embedding of every enrolled child using
  cosine similarity. Similarity is mapped to a calibrated confidence with a
  logistic function, and the same 70% rejection threshold is applied.

The UI always shows which mode produced a result, the raw cosine similarity, the
calibration parameters, and the enrolled-child count, so a similarity score is
never mistaken for a softmax probability.

### Calibration

```powershell
python calibrate.py
```

With one child, the closest available impostor proxy is the **opposite foot
side**. The script measures:

- genuine pairs — same side, different capture groups;
- impostor pairs — opposite side.

It fits a logistic `center` and `scale` between the two distributions, writes
per-architecture calibrations into `models/registry.json`, and records
false-accept/false-reject rates at the 70% threshold in
`models/calibration_report.json`. The classifier applies the calibration that
belongs to the active architecture, so switching models keeps the confidence
scale meaningful.

For a real multi-child dataset, re-run training with at least two subjects and
the application will switch to identity/softmax mode automatically.

---

## Preprocessing pipeline

```text
Upload → ROI detection → 224×224 midfoot crop → CLAHE → six-orientation Gabor fusion
```

- **ROI detection** builds several candidate masks (inverse/normal Otsu, fixed
  threshold, adaptive threshold, skin-colour exclusion) and scores them by
  contour area, solidity, position, and texture. If no candidate is convincing,
  a GrabCut fallback refines the largest plausible region. Unusable crops fall
  back to the full frame instead of producing a black image.
- The largest contour is divided into forefoot (0–40%), midfoot/arch (40–75%),
  and heel (75–100%); the midfoot/arch is the biometric ROI.
- **CLAHE** improves local contrast, and six zero-mean Gabor kernels at 0°, 30°,
  60°, 90°, 120°, and 150° are fused with the maximum response.

The original, ROI, and enhanced images are returned by `/api/preprocess-preview`
and displayed in the enrollment and identification results.

---

## Model comparison

| Architecture | Initialisation | Notes |
| --- | --- | --- |
| MobileNetV2 | ImageNet (fallback: scratch) | Lightweight transfer model with rescaling to `[-1, 1]`. |
| EfficientNet-B0 | ImageNet (fallback: scratch) | Stronger transfer model; slower on CPU. |
| Custom CNN | Scratch | Compact four-block CNN with batch normalisation, ~1M parameters. |

Every model exposes:

- a named `embedding` layer (256-D) for verification;
- a final convolutional layer for Grad-CAM;
- a raw 0–255 RGB input contract, with architecture-specific scaling inside the
  model.

Training uses grouped splits by capture group (subject + visit + sensor + side)
so near-duplicate images cannot leak between train, validation, and test. It
also applies light rotation/zoom/translation/brightness/contrast augmentation and
class weights. The registry chooses the best model by test accuracy, then top-3
accuracy, then macro F1 — but any architecture can be activated from the Model
Lab.

---

## Web interface

- **Enroll Child** — name, ID, drag-and-drop multiple images, preprocessing
  previews, and a live enrolled-children list.
- **Identify** — single-image recognition with confidence bar, top three, mode
  badge, cosine similarity, calibrated confidence, preprocessing stages, and
  Grad-CAM (raw heatmap, colour heatmap, and overlay).
- **Models** — dataset summary, architecture checkboxes, epochs/batch controls,
  live training progress (including epoch-level accuracy/loss), model comparison
  cards with metrics, active-model switching, and one-click dataset-sample
  enrollment.

The sidebar shows the active architecture, inference mode, threshold, and
enrolled-child count, with a shortcut to the Model Lab.

---

## API reference

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | Web application interface. |
| `GET` | `/api/health` | Lightweight service status without loading TensorFlow. |
| `POST` | `/api/enroll` | Enroll one child with one or more images. |
| `POST` | `/api/identify` | Identify one image; returns prediction, top three, similarity, preprocessing, and Grad-CAM. |
| `GET` | `/api/enrolled` | List enrolled children and actual image counts. |
| `PATCH` | `/api/enrolled/{child_id}/name` | Rename an enrolled child (display name only). |
| `DELETE` | `/api/enrolled/{child_id}` | Remove an enrolled child and their images. |
| `POST` | `/api/preprocess-preview` | Return original/ROI/CLAHE/Gabor previews. |
| `POST` | `/api/retrain?epochs=12&architecture=custom_cnn` | Train an identity model from enrolled images. |
| `GET` | `/api/model-info` | Active model, mode, threshold, and calibration. |
| `GET` | `/api/dataset-info` | Dataset counts, task resolution, cache/registry status. |
| `GET` | `/api/models` | Registry, per-architecture metrics, active model. |
| `POST` | `/api/models/select?architecture=…` | Activate a registered architecture. |
| `POST` | `/api/train?architectures=…&epochs=…` | Start background dataset training. |
| `GET` | `/api/train/status` | Training stage, progress events, error/result. |
| `POST` | `/api/enroll-sample?subject=B003&limit=12` | Enroll a subject from the bundled dataset for a quick demo. |

### Renaming children

A child has a fixed `child_id` (`CHILD_001`, also the model class and image
folder name) and a display `child_name`. Only the name changes when you rename;
the images, the id and the trained model stay as they are, so **no retraining is
needed**.

Rename one child from the UI (pencil button in the enrolled list), or all of
them from a name sheet:

```powershell
python rename_children.py --list
python rename_children.py --file child_names.csv --dry-run
python rename_children.py --file child_names.csv
```

`child_names.csv` holds `child_id,name` rows; the second column can also be the
current name (`B008`) instead of the id. Use `--mapping "B008=Aarav,B035=Meera"`
for a few inline changes.

### Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `FOOTPRINT_CONFIDENCE_THRESHOLD` | `0.70` | Minimum confidence to accept a child instead of returning **Unknown Child**. |

```powershell
$env:FOOTPRINT_CONFIDENCE_THRESHOLD = "0.80"
uvicorn app.main:app --reload
```

---

## Demonstration workflow

1. Start the server and open `http://127.0.0.1:8000`.
2. Open **Models** and click **Train Selected Models** (or run `python train.py`
   beforehand). Watch preprocessing, epoch metrics, and per-architecture results.
3. Click **Enroll Dataset Sample** to enroll `B003` from the training images, or
   use **Enroll Child** with your own consented images.
4. Switch to **Identify**, upload a footprint, and inspect the mode badge,
   confidence/similarity, top three, preprocessing pipeline, and Grad-CAM.
5. Upload a poor-quality or unrelated image; a score below the threshold must
   return **Unknown Child**.
6. In the Model Lab, click **Use this model** to switch architectures and repeat
   identification to compare behaviour.

---

## Verification

```powershell
python -m unittest discover -s tests -v
```

The suite covers preprocessing contracts, safe enrollment IDs, confidence and
rejection logic, embedding calibration, dataset discovery/task resolution,
grouped splits, training metrics, explainability encoding, API route contracts,
and threshold configuration.

---

## Known limitations

- The bundled dataset has one child, so the demo uses side-trained embeddings for
  verification rather than a true multi-child identity classifier.
- Opposite-side similarity is a proxy for impostor similarity; a real deployment
  needs multiple children and a held-out impostor set.
- A model trained on 149 images is not suitable for production biometrics.
- Phone photographs are excluded by default because ROI quality is poor.
- ROI orientation assumes consistently captured footprints; add orientation
  normalisation if scans can be rotated or inverted.

## Data handling and privacy

Footprints are biometric data. For any real deployment, obtain informed consent,
encrypt stored images and embeddings, restrict access, define retention rules,
and comply with applicable privacy law. This educational project stores processed
images locally under `data/enrolled/` and implements none of those protections.
