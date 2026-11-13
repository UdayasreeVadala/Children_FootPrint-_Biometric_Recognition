"""
FootprintID — FastAPI Application Entry Point

Serves the web application for children's footprint biometric recognition.
Provides API endpoints for enrollment, identification, model training and
management.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "enrolled"
MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "Training_Data"
CACHE_DIR = BASE_DIR / "data" / "cache" / "dataset_224"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
CHILD_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
DEFAULT_ARCHITECTURES = "mobilenetv2,efficientnetb0,custom_cnn"
DEFAULT_SENSORS = ("Paper", "Scanner", "Camera")

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="FootprintID",
    description="Deep Learning for Biometric Recognition of Children using Footprints",
    version="1.1.0",
)

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# ---------------------------------------------------------------------------
# Service status
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health_check():
    """Return a lightweight status response without loading TensorFlow."""
    return {
        "status": "ok",
        "service": "FootprintID",
        "version": app.version,
        "model_loaded": _classifier is not None,
        "training": _TRAINING_STATE["running"],
    }


# ---------------------------------------------------------------------------
# Lazy-loaded classifier state
# ---------------------------------------------------------------------------
_classifier = None
_CLASSIFIER_LOCK = threading.Lock()
_TRAINING_LOCK = threading.Lock()
_TRAINING_STATE: dict[str, Any] = {
    "running": False,
    "stage": "idle",
    "started_at": None,
    "finished_at": None,
    "current": 0,
    "total": 0,
    "detail": "",
    "events": [],
    "error": None,
    "result": None,
}


def _configured_threshold() -> float:
    """Read the confidence rejection threshold from the environment."""
    raw_value = os.getenv("FOOTPRINT_CONFIDENCE_THRESHOLD", "0.70")
    try:
        threshold = float(raw_value)
    except ValueError as error:
        raise RuntimeError("FOOTPRINT_CONFIDENCE_THRESHOLD must be a number between 0 and 1") from error
    if not 0.0 < threshold <= 1.0:
        raise RuntimeError("FOOTPRINT_CONFIDENCE_THRESHOLD must be greater than 0 and at most 1")
    return threshold


def _get_classifier():
    """Lazy-load the FootprintClassifier so TF only imports when needed."""
    global _classifier
    if _classifier is not None:
        return _classifier
    with _CLASSIFIER_LOCK:
        if _classifier is None:
            from app.inference.classifier import FootprintClassifier

            classifier = FootprintClassifier(
                models_dir=MODELS_DIR,
                enrollment_dir=DATA_DIR,
                confidence_threshold=_configured_threshold(),
            )
            _classifier = classifier
            if classifier.is_demo or not classifier.class_records:
                _sync_demo_class_names()
            try:
                classifier.refresh_enrollment_index()
            except Exception:
                # Embedding computation is best-effort; identification will retry.
                pass
    return _classifier


def _sync_demo_class_names() -> None:
    """Use enrolled children as the demo class list so the UI has labels."""
    global _classifier
    if _classifier is None or not _classifier.is_demo:
        return
    class_records = {}
    meta_path = DATA_DIR / "metadata.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        for index, child in enumerate(metadata.get("children", [])):
            class_records[index] = {
                "child_id": child.get("child_id", f"Child_{index}"),
                "child_name": child.get("child_name", child.get("child_id", f"Child {index}")),
            }
    if class_records:
        _classifier.update_class_records(class_records)


def _save_metadata(child_id: str, child_name: str, image_count: int) -> None:
    """Add or update a child in the metadata file atomically."""
    meta_path = DATA_DIR / "metadata.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    else:
        metadata = {"children": []}

    found = False
    for child in metadata["children"]:
        if child["child_id"] == child_id:
            child["child_name"] = child_name
            child["image_count"] = child.get("image_count", 0) + image_count
            found = True
            break
    if not found:
        metadata["children"].append(
            {
                "child_id": child_id,
                "child_name": child_name,
                "image_count": image_count,
            }
        )

    temporary_path = meta_path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    temporary_path.replace(meta_path)

    _sync_demo_class_names()
    if _classifier is not None and not _classifier.is_demo:
        _classifier.refresh_enrollment_index(force=True)


def _numpy_to_base64(image: np.ndarray | None, *, rgb: bool = False) -> str:
    """Convert a NumPy image (BGR or grayscale) to a base64-encoded PNG."""
    if image is None:
        return ""
    if rgb and image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".png", image)
    if not success:
        return ""
    return base64.b64encode(buffer).decode("utf-8")


def _read_upload(upload_file_bytes: bytes) -> np.ndarray:
    """Convert raw upload bytes to an OpenCV BGR image."""
    nparr = np.frombuffer(upload_file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")
    return image


def _validate_child_details(child_name: str, child_id: str) -> tuple[str, str]:
    """Normalize enrollment details and keep child folders inside DATA_DIR."""
    normalized_name = child_name.strip()
    normalized_id = child_id.strip()
    if not normalized_name or not normalized_id:
        raise HTTPException(status_code=400, detail="child_name and child_id are required")
    if len(normalized_name) > 100:
        raise HTTPException(status_code=400, detail="child_name must be 100 characters or fewer")
    if not CHILD_ID_PATTERN.fullmatch(normalized_id):
        raise HTTPException(
            status_code=400,
            detail="child_id must use 1-64 letters, numbers, underscores, or hyphens",
        )
    return normalized_name, normalized_id


def _child_directory(child_id: str) -> Path:
    """Return a validated enrolled-child directory without path traversal."""
    candidate = (DATA_DIR / child_id).resolve()
    if DATA_DIR.resolve() not in candidate.parents:
        raise HTTPException(status_code=400, detail="Invalid child_id")
    return candidate


def _preprocess_upload(image: np.ndarray) -> dict:
    """Run the ROI → enhancement pipeline and return displayable previews."""
    from app.preprocessing.enhancement import enhance_footprint
    from app.preprocessing.roi_detection import process_footprint

    roi_result = process_footprint(image, return_visualization=True)
    roi_image = roi_result["roi"]
    enhancement = enhance_footprint(roi_image)
    return {
        "roi": roi_image,
        "enhanced": enhancement["enhanced"],
        "clahe": enhancement.get("clahe"),
        "gabor_fused": enhancement.get("gabor_fused"),
        "roi_viz": roi_result.get("visualization"),
    }


# ---------------------------------------------------------------------------
# Page route
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main web application page."""
    return templates.TemplateResponse("index.html", {"request": request})


# ---------------------------------------------------------------------------
# API: Enroll
# ---------------------------------------------------------------------------
@app.post("/api/enroll")
async def enroll_child(
    child_name: str = Form(...),
    child_id: str = Form(...),
    files: list[UploadFile] = File(...),
):
    """Enroll a child by uploading footprint images.

    Images are preprocessed (ROI → CLAHE → Gabor) and the enhanced 224×224
    PNGs are stored under ``data/enrolled/{child_id}/``.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one image file is required")

    child_name, child_id = _validate_child_details(child_name, child_id)
    child_dir = _child_directory(child_id)
    child_existed = child_dir.exists()
    child_dir.mkdir(parents=True, exist_ok=True)

    preprocessing_previews = []
    images_enrolled = 0

    for file in files:
        image = None
        try:
            contents = await file.read()
            if not contents:
                raise ValueError("The uploaded file is empty")
            if len(contents) > MAX_UPLOAD_BYTES:
                raise ValueError("Each image must be 10 MB or smaller")
            image = _read_upload(contents)

            processed = _preprocess_upload(image)
            roi_image = processed["roi"]
            enhanced_image = processed["enhanced"]

            filename = f"{child_id}_{uuid.uuid4().hex[:8]}.png"
            save_path = child_dir / filename
            if not cv2.imwrite(str(save_path), enhanced_image):
                raise RuntimeError("Could not save the processed image")
            images_enrolled += 1

            preview = {
                "original": _numpy_to_base64(cv2.resize(image, (224, 224))),
                "roi": _numpy_to_base64(roi_image),
                "enhanced": _numpy_to_base64(enhanced_image),
            }
            if processed.get("roi_viz") is not None:
                preview["roi_viz"] = _numpy_to_base64(processed["roi_viz"])
            preprocessing_previews.append(preview)

        except Exception as error:
            preprocessing_previews.append(
                {
                    "error": str(error),
                    "original": _numpy_to_base64(
                        cv2.resize(image, (224, 224))
                        if image is not None
                        else np.zeros((224, 224, 3), dtype=np.uint8)
                    ),
                }
            )

    if images_enrolled == 0:
        if not child_existed:
            shutil.rmtree(child_dir, ignore_errors=True)
        raise HTTPException(
            status_code=422,
            detail="No usable footprint images were found. Use a clear dark footprint on a light background.",
        )

    _save_metadata(child_id, child_name, images_enrolled)
    return JSONResponse(
        {
            "success": True,
            "child_id": child_id,
            "child_name": child_name,
            "images_enrolled": images_enrolled,
            "preprocessing_previews": preprocessing_previews,
        }
    )


# ---------------------------------------------------------------------------
# API: Identify
# ---------------------------------------------------------------------------
@app.post("/api/identify")
async def identify_footprint(file: UploadFile = File(...)):
    """Identify a child from a footprint image."""
    from app.preprocessing.enhancement import enhance_footprint
    from app.preprocessing.roi_detection import process_footprint

    try:
        contents = await file.read()
        if len(contents) > MAX_UPLOAD_BYTES:
            raise ValueError("Image must be 10 MB or smaller")
        image = _read_upload(contents)
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {error}") from error

    try:
        roi_result = process_footprint(image, return_visualization=True)
        roi_image = roi_result["roi"]
        roi_viz = roi_result.get("visualization")
    except Exception:
        roi_image = cv2.resize(image, (224, 224))
        roi_viz = None

    enhanced_image = enhance_footprint(roi_image)["enhanced"]

    classifier = _get_classifier()
    prediction = classifier.predict(enhanced_image)

    # --- Grad-CAM ------------------------------------------------------
    gradcam_data: dict[str, Any] = {}
    try:
        from app.inference.gradcam import (
            generate_gradcam,
            generate_similarity_gradcam,
            gradcam_to_base64,
        )

        target_embedding = None
        if prediction.get("match_mode") == "embedding":
            target_child = prediction.get("predicted_id")
            if not target_child and prediction.get("top_3"):
                # Even a rejected probe should explain its closest candidate.
                target_child = prediction["top_3"][0].get("child_id")
            if target_child:
                target_embedding = classifier.get_child_embedding(target_child)

        if target_embedding is not None:
            gradcam_result = generate_similarity_gradcam(
                classifier.model,
                enhanced_image,
                classifier.last_conv_layer_name,
                target_embedding,
            )
        else:
            class_index = prediction.get("predicted_class")
            if class_index is None and prediction.get("top_3"):
                class_index = prediction["top_3"][0].get("class", 0)
            gradcam_result = generate_gradcam(
                classifier.model,
                enhanced_image,
                classifier.last_conv_layer_name,
                pred_index=int(class_index or 0),
            )

        raw_heatmap = cv2.resize(gradcam_result["heatmap_raw"], (224, 224))
        raw_heatmap = np.clip(raw_heatmap * 255.0, 0, 255).astype(np.uint8)
        gradcam_data = {
            "original": _numpy_to_base64(enhanced_image),
            "overlay": gradcam_to_base64(gradcam_result["overlay"]),
            "heatmap": gradcam_to_base64(gradcam_result["heatmap_colored"]),
            "raw_heatmap": _numpy_to_base64(raw_heatmap),
            "method": "embedding_similarity" if target_embedding is not None else "class_activation",
        }
    except Exception as error:
        gradcam_data = {"error": str(error)}

    preprocessing_data = {
        "original": _numpy_to_base64(cv2.resize(image, (224, 224))),
        "roi": _numpy_to_base64(roi_image),
        "enhanced": _numpy_to_base64(enhanced_image),
    }
    if roi_viz is not None:
        preprocessing_data["roi_viz"] = _numpy_to_base64(roi_viz)

    return JSONResponse(
        {
            "success": True,
            "prediction": {
                "predicted_name": prediction["predicted_name"],
                "predicted_id": prediction["predicted_id"],
                "predicted_class": prediction.get("predicted_class"),
                "confidence_percent": round(prediction["confidence_percent"], 2),
                "is_recognized": prediction["is_recognized"],
                "top_3": [
                    {
                        "name": item["name"],
                        "child_id": item["child_id"],
                        "confidence": round(item["confidence"] * 100, 2),
                        "similarity": (
                            round(item["similarity"] * 100, 2)
                            if item.get("similarity") is not None
                            else None
                        ),
                    }
                    for item in prediction["top_3"]
                ],
                "threshold": prediction["threshold"],
                "is_demo": prediction["is_demo"],
                "match_mode": prediction.get("match_mode", "softmax"),
                "similarity": (
                    round(prediction["similarity"] * 100, 2)
                    if prediction.get("similarity") is not None
                    else None
                ),
                "calibration": prediction.get("calibration"),
                "architecture": prediction.get("architecture"),
                "architecture_label": prediction.get("architecture_label"),
                "task": prediction.get("task"),
                "enrolled_children": prediction.get("enrolled_children", 0),
                "message": prediction.get("message"),
            },
            "gradcam": gradcam_data,
            "preprocessing": preprocessing_data,
        }
    )


# ---------------------------------------------------------------------------
# API: List / delete enrolled children
# ---------------------------------------------------------------------------
@app.get("/api/enrolled")
async def list_enrolled():
    """List all enrolled children and their actual image counts."""
    meta_path = DATA_DIR / "metadata.json"
    if not meta_path.exists():
        return JSONResponse({"children": []})

    with meta_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    children = []
    for child in metadata.get("children", []):
        child_dir = DATA_DIR / child["child_id"]
        actual_count = len(list(child_dir.glob("*.png"))) if child_dir.exists() else 0
        children.append(
            {
                "child_id": child["child_id"],
                "child_name": child["child_name"],
                "image_count": actual_count,
            }
        )
    return JSONResponse({"children": children})


@app.delete("/api/enrolled/{child_id}")
async def delete_enrolled(child_id: str):
    """Remove an enrolled child and their images."""
    _, child_id = _validate_child_details("delete", child_id)
    child_dir = _child_directory(child_id)
    if child_dir.exists():
        shutil.rmtree(child_dir)

    meta_path = DATA_DIR / "metadata.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        metadata["children"] = [c for c in metadata["children"] if c["child_id"] != child_id]
        with meta_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

    _sync_demo_class_names()
    if _classifier is not None and not _classifier.is_demo:
        _classifier.refresh_enrollment_index(force=True)
    return JSONResponse({"success": True, "message": f"Child {child_id} removed"})


# ---------------------------------------------------------------------------
# API: Preprocessing preview
# ---------------------------------------------------------------------------
@app.post("/api/preprocess-preview")
async def preprocess_preview(file: UploadFile = File(...)):
    """Preview the preprocessing pipeline on an uploaded image."""
    try:
        contents = await file.read()
        if len(contents) > MAX_UPLOAD_BYTES:
            raise ValueError("Image must be 10 MB or smaller")
        image = _read_upload(contents)
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Invalid image: {error}") from error

    try:
        processed = _preprocess_upload(image)
    except Exception:
        processed = {
            "roi": cv2.resize(image, (224, 224)),
            "enhanced": cv2.cvtColor(
                cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (224, 224)),
                cv2.COLOR_GRAY2BGR,
            ),
            "clahe": None,
            "gabor_fused": None,
            "roi_viz": None,
        }

    return JSONResponse(
        {
            "original": _numpy_to_base64(cv2.resize(image, (224, 224))),
            "roi": _numpy_to_base64(processed.get("roi")),
            "roi_visualization": (
                _numpy_to_base64(processed["roi_viz"])
                if processed.get("roi_viz") is not None
                else None
            ),
            "clahe": _numpy_to_base64(processed.get("clahe")),
            "gabor_fused": _numpy_to_base64(processed.get("gabor_fused")),
            "enhanced": _numpy_to_base64(processed.get("enhanced")),
        }
    )


# ---------------------------------------------------------------------------
# API: Dataset information
# ---------------------------------------------------------------------------
@app.get("/api/dataset-info")
async def dataset_info():
    """Summarize the on-disk training dataset without running preprocessing."""
    from app.inference.dataset import discover_dataset, resolve_task, summarize_samples

    sensors = list(DEFAULT_SENSORS)
    try:
        samples = discover_dataset(DATASET_DIR, include_sensors=sensors)
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    summary = summarize_samples(samples)
    summary["task"] = resolve_task(samples, "auto")
    summary["sensors_used"] = sensors
    summary["cache_exists"] = (CACHE_DIR / "dataset_manifest.json").exists()
    summary["registry_exists"] = (MODELS_DIR / "registry.json").exists()
    return JSONResponse(summary)


# ---------------------------------------------------------------------------
# API: Model registry and selection
# ---------------------------------------------------------------------------
@app.get("/api/models")
async def list_models():
    """Return the trained-model registry and the comparison metrics."""
    registry_path = MODELS_DIR / "registry.json"
    if not registry_path.exists():
        return JSONResponse(
            {
                "has_registry": False,
                "architectures": {},
                "best_model": None,
                "active_model": None,
                "task": None,
                "message": "No dataset-trained models yet. Start training to compare the three architectures.",
            }
        )

    with registry_path.open("r", encoding="utf-8") as file:
        registry = json.load(file)
    return JSONResponse(
        {
            "has_registry": True,
            **registry,
            "model_info": _get_classifier().get_model_info(),
        }
    )


@app.get("/api/models/artifact")
async def model_artifact(
    architecture: str = Query(..., min_length=1),
    kind: str = Query(default="training"),
):
    """Return a training curve or confusion matrix as a base64 PNG."""
    registry_path = MODELS_DIR / "registry.json"
    if not registry_path.exists():
        raise HTTPException(status_code=404, detail="No trained model registry exists")
    with registry_path.open("r", encoding="utf-8") as file:
        registry = json.load(file)

    metrics = registry.get("architectures", {}).get(architecture)
    if not isinstance(metrics, dict):
        raise HTTPException(status_code=404, detail=f"Architecture '{architecture}' is not registered")
    task = registry.get("task", "model")

    if kind == "training":
        candidate = metrics.get("training_plot_path") or str(
            MODELS_DIR / f"{architecture}_{task}_training.png"
        )
    elif kind == "confusion":
        candidate = metrics.get("confusion_matrix_path") or str(
            MODELS_DIR / f"{architecture}_{task}_confusion_matrix.png"
        )
    else:
        raise HTTPException(status_code=400, detail="kind must be 'training' or 'confusion'")

    path = Path(candidate).resolve()
    if MODELS_DIR.resolve() not in path.parents or not path.exists():
        raise HTTPException(status_code=404, detail="Artifact is not available")
    payload = base64.b64encode(path.read_bytes()).decode("utf-8")
    return JSONResponse(
        {
            "architecture": architecture,
            "kind": kind,
            "filename": path.name,
            "image": payload,
        }
    )


@app.post("/api/models/select")
async def select_model(architecture: str = Query(..., min_length=1)):
    """Make one of the registered architectures the active model."""
    global _classifier
    registry_path = MODELS_DIR / "registry.json"
    if not registry_path.exists():
        raise HTTPException(status_code=404, detail="No trained model registry exists")

    with registry_path.open("r", encoding="utf-8") as file:
        registry = json.load(file)
    if architecture not in registry.get("architectures", {}):
        raise HTTPException(status_code=404, detail=f"Architecture '{architecture}' is not registered")

    model_path = registry.get("model_paths", {}).get(architecture)
    if not model_path or not Path(model_path).exists():
        raise HTTPException(status_code=404, detail="The model file for this architecture is missing")

    registry["active_model"] = architecture
    active_calibration = (registry.get("calibrations", {}) or {}).get(architecture)
    if isinstance(active_calibration, dict):
        registry["calibration"] = active_calibration
    temporary = registry_path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(registry, file, indent=2)
    temporary.replace(registry_path)

    _classifier = None
    return JSONResponse({"success": True, "active_model": architecture})


# ---------------------------------------------------------------------------
# API: Model information
# ---------------------------------------------------------------------------
@app.get("/api/model-info")
async def model_info():
    """Return information about the loaded model."""
    classifier = _get_classifier()
    return JSONResponse(classifier.get_model_info())


# ---------------------------------------------------------------------------
# API: Enrollment-based retraining
# ---------------------------------------------------------------------------
@app.post("/api/retrain")
async def retrain_model(
    epochs: int = Query(default=12, ge=1, le=100),
    architecture: str = Query(default="custom_cnn"),
):
    """Train a child-identity model from the preprocessed enrolled-image folders."""
    global _classifier
    from app.inference.architectures import normalize_architecture_name
    from app.inference.training import retrain_from_enrollments

    try:
        architecture = normalize_architecture_name(architecture)
        result = retrain_from_enrollments(
            DATA_DIR, MODELS_DIR, epochs=epochs, architecture=architecture
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    # Make the freshly trained identity model active, even if a dataset
    # registry from the three-architecture comparison already exists.
    _activate_enrollment_model(result, architecture)
    _classifier = None
    return JSONResponse({"success": True, **result})


def _activate_enrollment_model(result: dict, architecture: str) -> None:
    """Write a minimal registry so the classifier picks the new identity model."""
    classes_path = Path(result["classes_path"])
    meta_path = MODELS_DIR / "footprint_model_meta.json"
    with classes_path.open("r", encoding="utf-8") as file:
        classes = json.load(file)
    class_names = [
        str(value.get("child_id", value)) if isinstance(value, dict) else str(value)
        for _, value in sorted(classes.items(), key=lambda item: int(item[0]))
    ]
    meta = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as file:
            meta = json.load(file)

    model_path = str(Path(result["model_path"]).resolve())
    registry = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": "identity",
        "class_names": class_names,
        "dataset": {"source": "enrolled children", "images": result.get("images_used", 0)},
        "split": {},
        "architectures": {
            architecture: {
                "architecture": architecture,
                "architecture_label": "Custom CNN (enrollment)"
                if architecture == "custom_cnn"
                else architecture,
                "model_path": model_path,
                "test_accuracy": None,
                "test_top3_accuracy": None,
                "test_macro_f1": None,
                "training_seconds": None,
                "parameters": None,
                "last_conv_layer": meta.get("last_conv_layer", ""),
                "source": "enrollments",
            }
        },
        "best_model": architecture,
        "active_model": architecture,
        "model_paths": {architecture: model_path},
        "best_model_path": model_path,
        "calibration": {"center": 0.72, "scale": 0.08},
        "calibrations": {architecture: {"center": 0.72, "scale": 0.08}},
        "preprocessing": {
            "pipeline": "ROI (midfoot/arch 40-75%) -> CLAHE -> 6-orientation Gabor bank",
            "input_size": 224,
        },
    }
    registry_path = MODELS_DIR / "registry.json"
    temporary = registry_path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(registry, file, indent=2)
    temporary.replace(registry_path)


# ---------------------------------------------------------------------------
# API: Dataset training (background)
# ---------------------------------------------------------------------------
def _training_progress(event: dict) -> None:
    """Record a training progress event for the polling endpoint."""
    with _TRAINING_LOCK:
        _TRAINING_STATE["stage"] = event.get("stage", _TRAINING_STATE["stage"])
        if "current" in event:
            _TRAINING_STATE["current"] = int(event["current"])
        if "total" in event:
            _TRAINING_STATE["total"] = int(event["total"])
        if "detail" in event:
            _TRAINING_STATE["detail"] = str(event["detail"])
        _TRAINING_STATE["events"].append(event)
        _TRAINING_STATE["events"] = _TRAINING_STATE["events"][-80:]


def _run_training_job(config: dict) -> None:
    """Background worker that trains all selected architectures."""
    global _classifier
    try:
        from app.inference.training import train_dataset_models

        result = train_dataset_models(
            dataset_root=DATASET_DIR,
            models_dir=MODELS_DIR,
            cache_dir=CACHE_DIR,
            task=config["task"],
            architectures=config["architectures"],
            epochs=config["epochs"],
            fine_tune_epochs=config["fine_tune_epochs"],
            batch_size=config["batch_size"],
            force_preprocess=config["force_preprocess"],
            max_images=config["max_images"],
            include_sensors=config["include_sensors"],
            verbose=1,
            progress=_training_progress,
        )
        with _TRAINING_LOCK:
            _TRAINING_STATE["result"] = result
            _TRAINING_STATE["stage"] = "complete"
        _classifier = None
    except Exception as error:  # pragma: no cover - surfaced through status
        with _TRAINING_LOCK:
            _TRAINING_STATE["stage"] = "failed"
            _TRAINING_STATE["error"] = f"{type(error).__name__}: {error}"
    finally:
        with _TRAINING_LOCK:
            _TRAINING_STATE["running"] = False
            _TRAINING_STATE["finished_at"] = datetime.now(timezone.utc).isoformat()


@app.post("/api/train")
async def train_models(
    architectures: str = Query(default=DEFAULT_ARCHITECTURES),
    epochs: int = Query(default=18, ge=1, le=100),
    fine_tune_epochs: int = Query(default=6, ge=0, le=50),
    batch_size: int = Query(default=16, ge=1, le=64),
    task: str = Query(default="auto"),
    include_phone: bool = Query(default=False),
    force_preprocess: bool = Query(default=False),
    max_images: int | None = Query(default=None, ge=4),
):
    """Start training MobileNetV2, EfficientNet-B0 and/or the custom CNN.

    Training runs in a background thread; poll ``GET /api/train/status``.
    """
    global _classifier
    with _TRAINING_LOCK:
        if _TRAINING_STATE["running"]:
            raise HTTPException(status_code=409, detail="A training job is already running")
        _TRAINING_STATE.update(
            {
                "running": True,
                "stage": "starting",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": None,
                "current": 0,
                "total": 0,
                "detail": "",
                "events": [],
                "error": None,
                "result": None,
            }
        )

    architecture_list = [
        name.strip() for name in architectures.split(",") if name.strip()
    ] or ["custom_cnn"]
    sensors = list(DEFAULT_SENSORS)
    if include_phone:
        sensors.append("Phone")
    config = {
        "architectures": architecture_list,
        "epochs": epochs,
        "fine_tune_epochs": fine_tune_epochs,
        "batch_size": batch_size,
        "task": task,
        "include_sensors": sensors,
        "force_preprocess": force_preprocess,
        "max_images": max_images,
    }
    _classifier = None
    thread = threading.Thread(target=_run_training_job, args=(config,), daemon=True)
    thread.start()
    return JSONResponse({"success": True, "message": "Training started", "config": config})


@app.get("/api/train/status")
async def training_status():
    """Return the background training state and recent events."""
    with _TRAINING_LOCK:
        state = dict(_TRAINING_STATE)
        state["events"] = list(_TRAINING_STATE["events"])[-40:]
    return JSONResponse(state)


# ---------------------------------------------------------------------------
# API: Enroll a sample from the training dataset (demo convenience)
# ---------------------------------------------------------------------------
@app.post("/api/enroll-sample")
async def enroll_sample(
    subject: str = Query(default=""),
    limit: int = Query(default=10, ge=2, le=60),
    sensors: str = Query(default="Paper,Scanner,Camera"),
):
    """Enroll a dataset subject for a quick end-to-end demonstration."""
    from app.inference.dataset import discover_dataset

    sensor_list = [name.strip() for name in sensors.split(",") if name.strip()]
    try:
        samples = discover_dataset(DATASET_DIR, include_sensors=sensor_list)
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    if not samples:
        raise HTTPException(status_code=404, detail="No dataset images were found")

    available_subjects = sorted({sample.subject for sample in samples})
    selected_subject = subject.strip() or available_subjects[0]
    subject_samples = [sample for sample in samples if sample.subject == selected_subject]
    if not subject_samples:
        raise HTTPException(
            status_code=404,
            detail=f"Subject '{selected_subject}' was not found. Available: {', '.join(available_subjects)}",
        )

    # Spread the selected images across sensors, visits and sides.
    chosen = []
    seen_groups = set()
    for sample in subject_samples:
        if len(chosen) >= limit:
            break
        key = (sample.sensor, sample.visit, sample.side)
        if key in seen_groups:
            continue
        seen_groups.add(key)
        chosen.append(sample)
    for sample in subject_samples:
        if len(chosen) >= limit:
            break
        if sample not in chosen:
            chosen.append(sample)

    child_id = re.sub(r"[^A-Za-z0-9_-]", "_", selected_subject)[:64] or "SAMPLE"
    child_name = f"{selected_subject} (dataset sample)"
    child_dir = _child_directory(child_id)
    child_dir.mkdir(parents=True, exist_ok=True)

    enrolled = 0
    previews = []
    for sample in chosen:
        try:
            image = cv2.imread(sample.source_path)
            if image is None:
                continue
            processed = _preprocess_upload(image)
            destination = child_dir / f"{child_id}_{uuid.uuid4().hex[:8]}.png"
            if not cv2.imwrite(str(destination), processed["enhanced"]):
                continue
            enrolled += 1
            if len(previews) < 6:
                previews.append(
                    {
                        "original": _numpy_to_base64(cv2.resize(image, (224, 224))),
                        "roi": _numpy_to_base64(processed["roi"]),
                        "enhanced": _numpy_to_base64(processed["enhanced"]),
                    }
                )
        except Exception:
            continue

    if enrolled == 0:
        raise HTTPException(status_code=422, detail="No dataset images could be preprocessed")
    _save_metadata(child_id, child_name, enrolled)
    return JSONResponse(
        {
            "success": True,
            "child_id": child_id,
            "child_name": child_name,
            "images_enrolled": enrolled,
            "preprocessing_previews": previews,
        }
    )


# ---------------------------------------------------------------------------
# Run with Uvicorn
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
