"""Keras model architectures used by FootprintID.

Three architectures are supported so that models can be compared on the same
footprint dataset:

* ``mobilenetv2`` — ImageNet-pretrained MobileNetV2 transfer-learning model.
* ``efficientnetb0`` — ImageNet-pretrained EfficientNet-B0 transfer model.
* ``custom_cnn`` — compact convolutional network trained from scratch.

Every model exposes two important layers:

* ``embedding`` — a 256-dimensional feature vector used for biometric
  verification (cosine similarity) when an identity classifier cannot be
  trained (for example, a dataset with a single child).
* a final convolutional layer that Grad-CAM can target.

All models consume RGB images in the 0–255 range.  Architecture-specific
scaling is embedded in the model so training and inference use exactly the
same input contract.

The heads deliberately keep spatial information (a small feature grid is
flattened) instead of collapsing it with global average pooling.  A footprint
is identified by *where* its creases and arch are, not only by which textures
are present: on the 220-identity adult benchmark a GAP head never left chance
level, while the spatial head below reached ~90% held-out top-1.
"""

from __future__ import annotations

from typing import Iterable

import tensorflow as tf

SUPPORTED_ARCHITECTURES = ("mobilenetv2", "efficientnetb0", "custom_cnn")

ARCHITECTURE_LABELS = {
    "mobilenetv2": "MobileNetV2",
    "efficientnetb0": "EfficientNet-B0",
    "custom_cnn": "Custom CNN",
}


def normalize_architecture_name(value: str) -> str:
    """Return a canonical architecture key or raise a helpful error."""
    key = str(value or "").strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "mobilenetv2": "mobilenetv2",
        "mobilenet": "mobilenetv2",
        "efficientnetb0": "efficientnetb0",
        "efficientnet": "efficientnetb0",
        "customcnn": "custom_cnn",
        "custom": "custom_cnn",
        "cnn": "custom_cnn",
    }
    if key not in aliases:
        raise ValueError(
            f"Unsupported architecture '{value}'. Choose one of: "
            + ", ".join(SUPPORTED_ARCHITECTURES)
        )
    return aliases[key]


def _spatial_features(feature_map, name: str = "spatial_head"):
    """Compress a 7×7 backbone map to 64 channels but keep the 7×7 layout.

    Flattening the full backbone map (7 × 7 × 1280) would create a 16M-parameter
    dense layer; a 1×1 convolution first reduces channels while preserving the
    spatial arrangement the footprint identity lives in.
    """
    x = tf.keras.layers.Conv2D(
        64, 1, padding="same", use_bias=False, name=f"{name}_projection"
    )(feature_map)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn")(x)
    x = tf.keras.layers.Activation("relu", name=f"{name}_relu")(x)
    x = tf.keras.layers.Flatten(name=f"{name}_flatten")(x)
    return x


HEAD_LAYER_NAMES = (
    "spatial_head_projection",
    "spatial_head_bn",
    "head_dropout",
    "embedding",
    "embedding_dropout",
    "classifier",
)


def _classification_head(features, num_classes: int, dropout: float, embedding_dim: int):
    """Shared classifier head used by the full model and the pre-train model."""
    x = tf.keras.layers.Dropout(dropout, name="head_dropout")(features)
    x = tf.keras.layers.Dense(embedding_dim, activation="relu", name="embedding")(x)
    x = tf.keras.layers.Dropout(dropout / 2.0, name="embedding_dropout")(x)
    return tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)


def build_classification_head_model(
    feature_map_shape: tuple[int, int, int],
    num_classes: int,
    *,
    dropout: float = 0.3,
    embedding_dim: int = 256,
) -> tf.keras.Model:
    """Build the footprint head over an already-computed backbone feature map.

    Samples that are not written to disk (the identity signal is gentle) can be
    trained in seconds instead of minutes.  The layer names match
    :func:`build_classification_model`, so the trained weights can be copied
    straight into the full model with :func:`head_weight_names`.
    """
    if num_classes < 1:
        raise ValueError("num_classes must be at least 1")
    feature_map = tf.keras.Input(shape=feature_map_shape, name="backbone_features")
    features = _spatial_features(feature_map)
    outputs = _classification_head(features, num_classes, dropout, embedding_dim)
    return tf.keras.Model(feature_map, outputs, name="footprint_head")


def _build_mobilenetv2(inputs, weights: str | None):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=weights,
    )
    base._name = "backbone"
    # MobileNetV2 expects inputs scaled to [-1, 1].
    scaled = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="preprocess")(inputs)
    # Return the feature map *inside this graph*, not ``base.output``: the
    # latter belongs to the backbone's own input tensor and cannot be used to
    # build a sub-model over this model's inputs.
    return base, base(scaled, training=False)


def _build_efficientnetb0(inputs, weights: str | None):
    base = tf.keras.applications.EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=weights,
    )
    base._name = "backbone"
    # EfficientNet includes its own rescaling and normalisation layers and
    # therefore expects raw 0–255 RGB input.
    return base, base(inputs, training=False)


def _build_custom_cnn(inputs, weights: str | None):
    del weights  # The custom network is always trained from scratch.
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="preprocess")(inputs)

    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False, name="block1_conv")(x)
    x = tf.keras.layers.BatchNormalization(name="block1_bn")(x)
    x = tf.keras.layers.Activation("relu", name="block1_relu")(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False, name="block1_conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="block1_bn2")(x)
    x = tf.keras.layers.Activation("relu", name="block1_relu2")(x)
    x = tf.keras.layers.MaxPooling2D(name="block1_pool")(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False, name="block2_conv")(x)
    x = tf.keras.layers.BatchNormalization(name="block2_bn")(x)
    x = tf.keras.layers.Activation("relu", name="block2_relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False, name="block2_conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="block2_bn2")(x)
    x = tf.keras.layers.Activation("relu", name="block2_relu2")(x)
    x = tf.keras.layers.MaxPooling2D(name="block2_pool")(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", use_bias=False, name="block3_conv")(x)
    x = tf.keras.layers.BatchNormalization(name="block3_bn")(x)
    x = tf.keras.layers.Activation("relu", name="block3_relu")(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", use_bias=False, name="block3_conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="block3_bn2")(x)
    x = tf.keras.layers.Activation("relu", name="block3_relu2")(x)
    x = tf.keras.layers.MaxPooling2D(name="block3_pool")(x)

    x = tf.keras.layers.Conv2D(192, 3, padding="same", use_bias=False, name="custom_last_conv")(x)
    x = tf.keras.layers.BatchNormalization(name="block4_bn")(x)
    x = tf.keras.layers.Activation("relu", name="block4_relu")(x)
    # 224 → 112 → 56 → 28 after the three pools, then 7×7 so the dense head
    # keeps a coarse map of the crease layout.
    x = tf.keras.layers.AveragePooling2D(pool_size=4, name="spatial_pool")(x)
    x = tf.keras.layers.Flatten(name="spatial_flatten")(x)
    return None, x


def build_classification_model(
    architecture: str,
    num_classes: int,
    *,
    weights: str | None = "imagenet",
    dropout: float = 0.3,
    embedding_dim: int = 256,
) -> tf.keras.Model:
    """Build a complete softmax classifier with a named embedding layer."""
    architecture = normalize_architecture_name(architecture)
    if num_classes < 1:
        raise ValueError("num_classes must be at least 1")

    inputs = tf.keras.Input(shape=(224, 224, 3), name="footprint")

    if architecture == "mobilenetv2":
        base, feature_map = _build_mobilenetv2(inputs, weights)
        features = _spatial_features(feature_map, name="spatial_head")
    elif architecture == "efficientnetb0":
        base, feature_map = _build_efficientnetb0(inputs, weights)
        features = _spatial_features(feature_map, name="spatial_head")
    else:
        base, features = _build_custom_cnn(inputs, weights)
        feature_map = None

    outputs = _classification_head(features, num_classes, dropout, embedding_dim)

    model = tf.keras.Model(inputs, outputs, name=f"footprint_{architecture}")
    # Keep non-serialised references for the training loop.
    model._footprint_base_model = base  # type: ignore[attr-defined]
    model._footprint_feature_map = feature_map  # type: ignore[attr-defined]
    return model


def build_embedding_model(model: tf.keras.Model) -> tf.keras.Model:
    """Create a sub-model that returns the 256-D embedding vector."""
    return tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer("embedding").output,
        name=f"{model.name}_embedding",
    )


def find_last_conv_layer(model: tf.keras.Model) -> str:
    """Return the deepest Conv2D layer name, including nested sub-models."""
    layers: list[tf.keras.layers.Layer] = []

    def collect(current_model: tf.keras.Model) -> None:
        for layer in current_model.layers:
            if isinstance(layer, tf.keras.Model):
                collect(layer)
            else:
                layers.append(layer)

    collect(model)
    for layer in reversed(layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return ""


def unfreeze_for_fine_tuning(
    model: tf.keras.Model,
    number_of_layers: int = 30,
) -> int:
    """Unfreeze the last *N* backbone layers (BatchNorm stays frozen)."""
    base = getattr(model, "_footprint_base_model", None)
    if base is None:
        try:
            base = model.get_layer("backbone")
        except ValueError:
            return 0

    base.trainable = True
    for layer in base.layers[:-number_of_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    return sum(1 for layer in base.layers if layer.trainable)


def architecture_display_name(architecture: str) -> str:
    """Return a human-readable architecture name."""
    try:
        return ARCHITECTURE_LABELS[normalize_architecture_name(architecture)]
    except ValueError:
        return str(architecture)


def available_architectures() -> Iterable[str]:
    return SUPPORTED_ARCHITECTURES
