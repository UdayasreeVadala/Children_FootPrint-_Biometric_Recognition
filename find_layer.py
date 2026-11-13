#!/usr/bin/env python
from app.inference.architectures import find_last_conv_layer
from app.inference import mobilenetv2

# Create a mobilenetv2 model
model = mobilenetv2.build_classification_model("mobilenetv2", 5, weights=None, dropout=0.2)
print("Model created")

# Find the last conv layer
last = find_last_conv_layer(model)
print(f"find_last_conv_layer returned: '{last}'")

# List all layer names
print("All layer names:")
for l in model.layers:
    print(f"  {l.name}")
PYEOF