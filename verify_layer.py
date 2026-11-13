#!/usr/bin/env python
# Fix the last_conv_layer_name for the mobilenetv2 model
import tensorflow as tf
from app.inference.architectures import build_classification_model, find_last_conv_layer

# Create the model
model = build_classification_model("mobilenetv2", 5, weights=None, dropout=0.2)

# Get the current last_conv_layer_name
current = find_last_conv_layer(model)
print(f"Current last_conv_layer_name: '{current}'")

# List all layer names
print("Model layer names:")
for l in model.layers:
    print(f"  {l.name}")

# The 'backbone' layer is the main convolutional base
# Let's set last_conv_layer_name to 'backbone'
# We need to modify the classifier's attribute

# Actually, let's just directly set it
classifier_last = 'backbone'
print(f"\nSetting last_conv_layer_name to: '{classifier_last}'")

# Verify 'backbone' is a valid reference in the model
if hasattr(model, 'get_layer'):
    try:
        layer = model.get_layer('backbone')
        print(f'✓ Layer "backbone" exists and output shape: {layer.output.shape}')
    except Exception as e:
        print(f'✗ Layer "backbone" error: {e}')

# Also check if 'Conv_1' exists
try:
    layer = model.get_layer('Conv_1')
    print(f'✓ Layer "Conv_1" exists: {layer.output.shape}')
except Exception as e:
    print(f'✗ Layer "Conv_1" does NOT exist: {e}')

print(f'\nConclusion: Set last_conv_layer_name to "backbone"')
PYEOF