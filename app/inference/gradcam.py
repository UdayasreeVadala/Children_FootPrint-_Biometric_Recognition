"""
Grad-CAM Visualization Module
==============================

Generates Gradient-weighted Class Activation Mapping (Grad-CAM)
visualizations to highlight which spatial regions of a footprint image
contribute most to the classifier's decision.

Reference
---------
Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017.
"""

from __future__ import annotations

import base64

import cv2
import numpy as np
try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None


# ------------------------------------------------------------------
# Core Grad-CAM computation
# ------------------------------------------------------------------

def _prepare_image_batch(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale/BGR image to a raw 0–255 RGB ``(1, 224, 224, 3)`` batch.

    The trained models contain their own architecture-specific rescaling, so
    the batch must stay in the same 0–255 range used during training.
    """
    img = np.asarray(image)
    if img.size == 0:
        raise ValueError("A non-empty image is required for Grad-CAM.")
    if img.shape[0] != 224 or img.shape[1] != 224:
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    if img.ndim == 2:
        img_3ch = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 1:
        img_3ch = np.concatenate([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 3:
        # The web application supplies BGR-enhanced images.  Grayscale images
        # are replicated above so the channel-order conversion is a no-op for
        # the biometric input, but remains correct for colour callers.
        img_3ch = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Image must be grayscale or have three channels.")

    img_float = img_3ch.astype(np.float32)
    if img_float.max() <= 1.0:
        img_float = img_float * 255.0
    return np.expand_dims(img_float, axis=0)


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str,
    pred_index: int = None,
) -> np.ndarray:
    """Compute a Grad-CAM heatmap for a given image and model.

    Parameters
    ----------
    img_array : np.ndarray
        Preprocessed image batch of shape ``(1, 224, 224, 3)``.
    model : tf.keras.Model
        Trained Keras model.
    last_conv_layer_name : str
        Name of the last convolutional layer whose activations will
        be used to produce the heatmap.
    pred_index : int, optional
        Class index to explain.  When *None*, the predicted (argmax)
        class is used.

    Returns
    -------
    np.ndarray
        Grayscale heatmap normalised to [0, 1], shape ``(H, W)``
        matching the spatial dimensions of the target conv layer.
    """
    if tf is None:
        raise RuntimeError("TensorFlow is required for Grad-CAM. Run pip install -r requirements.txt.")

    # Build a sub-model that outputs both the conv-layer activations
    # and the final predictions.
    conv_tensor, prediction_tensor = _grad_tensors(model, last_conv_layer_name)
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[conv_tensor, prediction_tensor],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradients of the predicted class w.r.t. the conv-layer output
    grads = tape.gradient(class_channel, conv_outputs)

    # Global-average-pool the gradients over the spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight each activation channel by its corresponding gradient
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU and normalise
    heatmap = tf.nn.relu(heatmap)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def make_similarity_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str,
    target_embedding: np.ndarray,
    embedding_layer_name: str = "embedding",
) -> np.ndarray:
    """Grad-CAM for an embedding-verification match.

    Instead of explaining a softmax class, this explains the cosine similarity
    between the query image and the matched enrolled child's mean embedding.
    The result highlights the crease regions that made the verification
    decision.
    """
    if tf is None:
        raise RuntimeError("TensorFlow is required for Grad-CAM. Run pip install -r requirements.txt.")

    try:
        model.get_layer(embedding_layer_name)
    except ValueError as error:
        raise ValueError(
            f"The model does not expose an '{embedding_layer_name}' layer for similarity Grad-CAM."
        ) from error

    conv_tensor, embedding_tensor = _grad_tensors(
        model,
        last_conv_layer_name,
        extra_layer_name=embedding_layer_name,
    )
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[conv_tensor, embedding_tensor],
    )
    target = np.asarray(target_embedding, dtype=np.float32).reshape(-1)
    target_tensor = tf.convert_to_tensor(target, dtype=tf.float32)
    target_tensor = tf.math.l2_normalize(target_tensor)

    with tf.GradientTape() as tape:
        conv_outputs, embeddings = grad_model(img_array)
        normalized = tf.math.l2_normalize(embeddings, axis=-1)
        similarity = tf.reduce_sum(normalized * target_tensor[tf.newaxis, :], axis=-1)
        class_channel = tf.reduce_mean(similarity)

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise RuntimeError("Could not compute gradients for the matched embedding.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.nn.relu(heatmap)
    maximum = tf.reduce_max(heatmap)
    if maximum > 0:
        heatmap = heatmap / maximum
    return heatmap.numpy()


# ------------------------------------------------------------------
# Colormap & overlay helpers
# ------------------------------------------------------------------

def apply_colormap(
    heatmap: np.ndarray,
    colormap_name: str = "jet",
) -> np.ndarray:
    """Apply a pseudocolor map to a grayscale heatmap.

    Parameters
    ----------
    heatmap : np.ndarray
        2-D heatmap with values in [0, 1].
    colormap_name : str
        OpenCV colormap name.  Supported shortcuts: ``'jet'``,
        ``'hot'``, ``'bone'``, ``'inferno'``.  Defaults to ``'jet'``.

    Returns
    -------
    np.ndarray
        Coloured heatmap in RGB format, dtype ``uint8``, shape
        ``(H, W, 3)``.
    """
    heatmap_uint8 = np.uint8(255 * heatmap)

    colormap_lut = {
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "bone": cv2.COLORMAP_BONE,
        "inferno": cv2.COLORMAP_INFERNO,
    }
    cmap = colormap_lut.get(colormap_name, cv2.COLORMAP_JET)
    colored = cv2.applyColorMap(heatmap_uint8, cmap)

    # OpenCV returns BGR → convert to RGB
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return colored_rgb


def overlay_heatmap(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    target_size: tuple = (224, 224),
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on the original image.

    Parameters
    ----------
    original_image : np.ndarray
        Original image (any size, 1- or 3-channel).
    heatmap : np.ndarray
        Raw grayscale heatmap in [0, 1].
    alpha : float
        Blending factor for the heatmap (0 = invisible, 1 = fully
        opaque).  Default ``0.4``.
    target_size : tuple of int
        ``(height, width)`` to which both images are resized before
        blending.

    Returns
    -------
    np.ndarray
        Blended RGB image, dtype ``uint8``, shape
        ``(target_size[0], target_size[1], 3)``.
    """
    # Resize heatmap to target size and colourise
    heatmap_resized = cv2.resize(heatmap, (target_size[1], target_size[0]))
    colored_heatmap = apply_colormap(heatmap_resized).astype(np.float32)

    # Prepare the original image
    img = original_image.copy()
    if img.shape[0] != target_size[0] or img.shape[1] != target_size[1]:
        img = cv2.resize(img, (target_size[1], target_size[0]))

    # Ensure 3-channel
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 1:
        img = np.concatenate([img] * 3, axis=-1)

    # Normalise original to 0-255 float for blending
    if img.max() <= 1.0:
        img = (img * 255.0).astype(np.float32)
    else:
        img = img.astype(np.float32)

    superimposed = colored_heatmap * alpha + img * (1 - alpha)
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    return superimposed


# ------------------------------------------------------------------
# High-level pipeline
# ------------------------------------------------------------------

def generate_gradcam(
    model: tf.keras.Model,
    image: np.ndarray,
    last_conv_layer_name: str,
    pred_index: int = None,
) -> dict:
    """Run the full Grad-CAM pipeline on a single image.

    Parameters
    ----------
    model : tf.keras.Model
        Trained classification model.
    image : np.ndarray
        Raw input image (any size, 1- or 3-channel).
    last_conv_layer_name : str
        Name of the target convolutional layer.
    pred_index : int, optional
        Class index to visualise.  Defaults to the predicted class.

    Returns
    -------
    dict
        Dictionary with the following keys:

        - **heatmap_raw** – Grayscale heatmap ``(H, W)`` in [0, 1].
        - **heatmap_colored** – Coloured heatmap ``(224, 224, 3)``
          uint8 RGB.
        - **overlay** – Heatmap superimposed on the original image
          ``(224, 224, 3)`` uint8 RGB.
        - **pred_index** – Class index used for the explanation.
    """
    if tf is None or model is None:
        raise RuntimeError("A TensorFlow model is required for Grad-CAM.")

    img_batch = _prepare_image_batch(image)

    # --- resolve class index if needed -------------------------------
    if pred_index is None:
        preds = model.predict(img_batch, verbose=0)
        pred_index = int(np.argmax(preds[0]))

    # --- generate heatmap & overlay ----------------------------------
    heatmap_raw = make_gradcam_heatmap(
        img_batch, model, last_conv_layer_name, pred_index=pred_index
    )
    heatmap_colored = apply_colormap(
        cv2.resize(heatmap_raw, (224, 224))
    )
    overlay = overlay_heatmap(image, heatmap_raw, alpha=0.4, target_size=(224, 224))

    return {
        "heatmap_raw": heatmap_raw,
        "heatmap_colored": heatmap_colored,
        "overlay": overlay,
        "pred_index": pred_index,
    }


def generate_similarity_gradcam(
    model: tf.keras.Model,
    image: np.ndarray,
    last_conv_layer_name: str,
    target_embedding: np.ndarray,
    embedding_layer_name: str = "embedding",
) -> dict:
    """Run the Grad-CAM pipeline for an embedding-verification match."""
    if tf is None or model is None:
        raise RuntimeError("A TensorFlow model is required for Grad-CAM.")

    img_batch = _prepare_image_batch(image)
    heatmap_raw = make_similarity_gradcam_heatmap(
        img_batch,
        model,
        last_conv_layer_name,
        target_embedding,
        embedding_layer_name=embedding_layer_name,
    )
    heatmap_colored = apply_colormap(cv2.resize(heatmap_raw, (224, 224)))
    overlay = overlay_heatmap(image, heatmap_raw, alpha=0.4, target_size=(224, 224))
    return {
        "heatmap_raw": heatmap_raw,
        "heatmap_colored": heatmap_colored,
        "overlay": overlay,
    }


# ------------------------------------------------------------------
# Encoding helper
# ------------------------------------------------------------------

def gradcam_to_base64(image: np.ndarray) -> str:
    """Encode a numpy image as a base64-encoded PNG string.

    Useful for embedding Grad-CAM visualizations directly into HTML
    ``<img>`` tags.

    Parameters
    ----------
    image : np.ndarray
        RGB image, dtype ``uint8``.

    Returns
    -------
    str
        Base64-encoded PNG data (no ``data:`` URI prefix).
    """
    # cv2 expects BGR for encoding
    if image.ndim == 3 and image.shape[-1] == 3:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        bgr = image

    success, buffer = cv2.imencode(".png", bgr)
    if not success:
        raise RuntimeError("Failed to encode image as PNG")

    return base64.b64encode(buffer).decode("utf-8")


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _contains_layer(model: tf.keras.Model, layer_name: str, *, _depth: int = 0) -> bool:
    """Return True when *layer_name* exists anywhere inside *model*."""
    if _depth > 8:  # guard against pathological nesting
        return False
    try:
        model.get_layer(layer_name)
        return True
    except ValueError:
        pass
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and _contains_layer(layer, layer_name, _depth=_depth + 1):
            return True
    return False


def _forward_capture(
    model: tf.keras.Model,
    layer_names: set[str],
) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
    """Replay ``model``'s top-level layers on ``model.inputs``.

    Returns the tensors produced by each requested top-level layer, plus the
    final output tensor. Every returned tensor is wired to ``model.inputs``.
    """
    if not isinstance(model.inputs, (list, tuple)) or len(model.inputs) != 1:
        raise ValueError("Grad-CAM expects a single-input model.")

    tensor = model.inputs[0]
    captured: dict[str, tf.Tensor] = {}
    remaining = set(layer_names)

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        tensor = layer(tensor)
        if layer.name in remaining:
            captured[layer.name] = tensor
            remaining.discard(layer.name)

    if remaining:
        missing = ", ".join(sorted(remaining))
        raise ValueError(f"Could not locate layer output(s) for: {missing}")
    return captured, tensor


def _grad_tensors(
    model: tf.keras.Model,
    layer_name: str,
    *,
    extra_layer_name: str | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return (activation, second_output) tensors connected to ``model.inputs``.

    A Keras functional sub-model keeps its symbolic ``.output`` attached to its
    *own* input tensor, so after reloading a model, building a grad-model from
    ``model.inputs`` to ``backbone.output`` fails with::

        Graph disconnected: cannot obtain value for tensor ... at layer "Conv1"

    Re-invoking the layers rebuilds the activations inside the outer graph,
    which fixes that without changing the model.
    """
    activation_name = _resolve_conv_layer(model, layer_name).name
    targets = {activation_name}
    if extra_layer_name:
        targets.add(extra_layer_name)

    captured, final_tensor = _forward_capture(model, targets)
    activation = captured[activation_name]
    if extra_layer_name:
        return activation, captured[extra_layer_name]
    return activation, final_tensor


def _resolve_conv_layer(
    model: tf.keras.Model,
    layer_name: str,
) -> tf.keras.layers.Layer:
    """Resolve a layer by name, always returning one wired into *model*.

    Keras functional sub-models (for example MobileNetV2 wrapped as the
    ``backbone`` layer) keep their own input tensor. Taking the output of a
    layer *inside* such a sub-model yields a grad-model whose inputs are not
    connected to ``model.inputs``, so building it fails with::

        Graph disconnected: cannot obtain value for tensor ... at layer "Conv1"

    When the requested layer lives inside a nested sub-model we therefore
    return the *sub-model itself*. Its output is a spatial feature map that is
    reachable from ``model.inputs``, which is all Grad-CAM needs.

    Parameters
    ----------
    model : tf.keras.Model
        Top-level model.
    layer_name : str
        Target layer name.

    Returns
    -------
    tf.keras.layers.Layer

    Raises
    ------
    ValueError
        If the layer cannot be found.
    """
    # Direct lookup first: a top-level layer is always safe to use.
    try:
        return model.get_layer(layer_name)
    except ValueError:
        pass

    # Otherwise fall back to the outermost sub-model that contains the layer.
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and _contains_layer(layer, layer_name):
            return layer

    raise ValueError(
        f"Layer '{layer_name}' not found in model or any nested sub-model."
    )
