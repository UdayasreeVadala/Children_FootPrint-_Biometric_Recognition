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
| `--dataset` | `Training_Data` | Dataset root to train on. |
| `--models-dir` | `models` | Where models, metrics and `registry.json` are written. |
| `--cache-dir` | `data/cache/<dataset>_224` | Preprocessing cache to reuse. |
| `--task` | `auto` | `auto`, `identity`, `side`, `visit`, or `sensor`. |
| `--epochs` | `18` | Epochs for the custom CNN (trained from scratch). |
| `--head-epochs` | `40` | Head-only epochs for the transfer models; their frozen backbone features are cached, so these are cheap. |
| `--fine-tune-epochs` | `6` | Backbone fine-tuning epochs for transfer models (slow on CPU). |
| `--resume` | off | Continue the custom CNN from its checkpoint instead of starting over. |
| `--learning-rate` | `1e-3` | From-scratch Adam learning rate (use a smaller value when resuming). |
| `--batch-size` | `16` | Training batch size. |
| `--seed` | `42` | Random seed for the split, shuffling and initialisation. |
| `--quiet` | off | Suppress per-epoch progress output. |
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

## Adult benchmark and children prototype

The public adult dataset is the primary experiment; the children's captures are
a small cross-domain prototype.  They are deliberately kept in separate folders
so a benchmark run can never overwrite the child models.

### Adult dataset

The benchmark uses the 220-subject MATLAB footprint dataset: six acquisition
visits per subject, 1,320 foot images, exported as

```text
data/external/adult_mat_footprints/
└── Adult001/Scanner/Visit 1/001_1.png   …  Adult220/Scanner/Visit 6/220_6.png
```

Convert the raw `.mat` files (named `001_1.mat` … `220_6.mat`) once:

```powershell
python import_mat_adult_dataset.py --source data/external/adult_mat_raw
```

### Training the benchmark

```powershell
python train.py `
  --dataset data/external/adult_mat_footprints `
  --models-dir models/adult_220_benchmark `
  --cache-dir data/cache/adult_mat_footprints_224 `
  --architectures mobilenetv2,efficientnetb0,custom_cnn `
  --epochs 25 --head-epochs 60 --fine-tune-epochs 0 --batch-size 16
```

Visits 1–4 train, visit 5 validates, visit 6 is the held-out test set (one image
per identity).  Models, curves, confusion matrices and `registry.json` are
written to `models/adult_220_benchmark`.

| Architecture | Initialisation | Test top-1 | Test top-3 |
| --- | --- | --- | --- |
| EfficientNet-B0 | ImageNet, frozen | 93.6% | 94.5% |
| MobileNetV2 | ImageNet, frozen | 93.2% | 94.6% |
| Custom CNN | from scratch (resumed) | 77.3% | 86.8% |

Run `python build_report.py` to regenerate `results/REPORT.md` and the figures.

### Resuming the custom CNN

The transfer models are cheap to redo — their head trains on cached backbone
features in about a minute. The custom CNN does not: a full run is hours on CPU.
`--resume` continues it from the saved checkpoint instead of restarting:

```powershell
python train.py `
  --dataset data/external/adult_mat_footprints `
  --models-dir models/adult_220_benchmark `
  --cache-dir data/cache/adult_mat_footprints_224 `
  --architectures custom_cnn --epochs 40 --resume --learning-rate 5e-4 `
  --batch-size 16 --fine-tune-epochs 0
```

What `--resume` does:

- reloads `custom_cnn_identity.keras` and first copies it to
  `custom_cnn_identity.keras.prev`, because checkpoint selection restarts in the
  new process and the first epoch would otherwise overwrite the previous best;
- appends the new epochs to the existing `custom_cnn_identity_history.json`, so
  the training curve covers both stages;
- merges the registry instead of replacing it, so the MobileNetV2 and
  EfficientNet-B0 rows survive a custom-CNN-only resume.

Only the custom CNN can be resumed. A transfer model is trained through a frozen
backbone head stage that cannot be continued halfway.

On the 220-identity benchmark a second stage at `--learning-rate 5e-4` raised
visit-5 validation accuracy from **0.5545** after 25 epochs to **0.7545** after
65 cumulative epochs, and held-out visit-6 test top-1 from **59.1% to 77.3%**
(top-3 **76.8% → 86.8%**, macro F1 **0.519 → 0.725**). The resume trained all 40
additional epochs; the final numbers are in the registry, the metrics file and
`results/REPORT.md`.

### Two fixes that made this work

- **Spatial head instead of global average pooling.**  A footprint is identified
  by *where* its creases and arch sit.  GAP averaged that away and kept every
  model at chance (~0.5% for 220 classes) even though a nearest-neighbour check
  on the same images reached 90%.  The heads now keep a 7×7 grid: a 1×1
  projection to 64 channels and flatten for the transfer models, a 7×7 average
  pool for the custom CNN.
- **Freeze the pretrained backbone while the head learns.**  The dataset trainer
  used to fine-tune the whole ImageNet backbone at the head learning rate
  (`Adam(1e-3)`), which destroyed the pretrained filters in a few steps.  The
  transfer models now cache the frozen backbone's feature maps, train the head
  for dozens of epochs in seconds, and only then optionally unfreeze the last 30
  layers at `1e-5`.

### Children's data prototype

Do not score children against the adult softmax class labels: an adult identity
classifier can only emit adult IDs, so that accuracy is 0% by construction.  The
prototype instead reuses the adult network's 256-D embedding as a fixed feature
extractor and matches each held-out child image against a gallery built from
that child's other images:

```powershell
python evaluate_case_study.py --models-dir models/adult_220_benchmark
```

With EfficientNet-B0 as the adult feature extractor and 34 children (248 images,
random chance 2.9%), a leave-one-identity-out open-set probe on PCA-whitened
features (128 dimensions) measures **61.1% top-1**, genuine/impostor ROC-AUC
0.60, and an equal error rate of 0.30 on the margin score.  On the deployed
pipeline the operating point requires both an absolute whitened cosine
similarity of at least **0.50** and a lead over the runner-up of at least
**0.12**; on held-out probes this rejects ~11% of adult (out-of-domain) feet
while still accepting the enrolled gallery.  These are feasibility numbers on a
very small, heterogeneous set — not a production accuracy claim.

> Earlier revisions of this document reported 40.5% top-1 / 73.5% rejected.
> That protocol included the query image in its own gallery centroid and used a
> non-discriminative ReLU embedding, so those figures were optimistic and are
> superseded by the leave-one-identity-out numbers above.

### Non-footprint rejection

Before any embedding comparison, `process_footprint` computes a deterministic
quality report (foreground area, contrast, aspect ratio, solidity, outline and
border contact).  Blank pages, solid colours, noise, gradients and solid
geometric blobs fail this gate and the API returns **Unknown Child** without
calling the classifier.  Calibration on 40 raw adult footprints gives 100%
acceptance, while synthetic non-foot images are all rejected.

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
| Custom CNN | Scratch | Four-block CNN with batch normalisation, ~3M parameters. |

Every model exposes:

- a named `embedding` layer (256-D) for verification;
- a final convolutional layer for Grad-CAM (`spatial_head_projection` for the
  transfer models, `custom_last_conv` for the custom CNN);
- a raw 0–255 RGB input contract, with architecture-specific scaling inside the
  model;
- a spatial classification head (a 7×7 grid flattened into the dense layers)
  rather than global average pooling, because crease *location* carries the
  identity.

Training uses grouped splits by capture group (subject + visit + sensor + side)
so near-duplicate images cannot leak between train, validation, and test. It
also applies light rotation/zoom/translation/brightness/contrast augmentation and
class weights.  Transfer models cache their frozen backbone features for the
head stage, so dozens of head epochs cost seconds; the custom CNN trains from
scratch.  The registry chooses the best model by test accuracy, then top-3
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
