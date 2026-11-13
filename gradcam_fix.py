#!/usr/bin/env python
import urllib.request, json, os

# Step 1: Get model info
try:
    r = urllib.request.urlopen('http://127.0.0.1:8000/api/model-info', timeout=15)
    info = json.loads(r.read().decode())
    print('=== MODEL INFO ===')
    for k, v in info.items():
        print(f'  {k}: {v}')
    
    last_conv = info.get('last_conv_layer', '')
    print(f'\nLast conv layer from registry: "{last_conv}"')
    
except Exception as e:
    print(f'Error getting model info: {e}')
    last_conv = ''

# Step 2: Check the actual model
if last_conv and __import__('os').path.exists(info.get('model_path', '')) if 'info' in dir() else False:
    import tensorflow as tf
    from tensorflow import keras
    
    try:
        model = keras.models.load_model(info.get('model_path', ''))
        layers = [l.name for l in model.layers]
        print(f'\nModel loaded successfully')
        print(f'Total layers: {len(layers)}')
        print(f'Layer names: {layers[:10]}...')
        
        if last_conv:
            if last_conv in layers:
                print(f'✓ Layer "{last_conv}" EXISTS in model')
            else:
                print(f'✗ Layer "{last_conv}" NOT in model')
                # Find conv layers
                conv_layers = [l for l in layers if 'conv' in l.lower() or 'block' in l.lower() or 'project' in l.lower()]
                print(f'Conv-like layers: {conv_layers[:10]}')
    except Exception as e:
        print(f'Error loading model: {e}')
else:
    print('Could not load model for layer check')
PYEOF