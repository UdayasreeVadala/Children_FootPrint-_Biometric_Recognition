#!/usr/bin/env python
import urllib.request, json, os

# Step 1: Update the registry file directly
registry_path = 'models/registry.json'

with open(registry_path, 'r') as f:
    registry = json.load(f)

print('Current registry last_conv_layer:', registry.get('last_conv_layer', 'NOT FOUND'))

# Update the last_conv_layer for all architectures
for arch in registry.get('calibrations', {}):
    if 'last_conv_layer' not in registry.get('calibrations', {}).get(arch, {}):
        registry['calibrations'][arch]['last_conv_layer'] = 'backbone'

# Also update the top-level last_conv_layer
registry['last_conv_layer'] = 'backbone'

with open(registry_path, 'w') as f:
    json.dump(registry, f, indent=2)

print(f'Updated registry last_conv_layer to: "backbone"')

# Also update model-info endpoint by restarting server conceptually
# But first let's verify the change
with open(registry_path, 'r') as f:
    print('New registry last_conv_layer:', json.load(f).get('last_conv_layer'))
PYEOF