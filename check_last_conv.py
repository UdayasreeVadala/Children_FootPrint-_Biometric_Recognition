#!/usr/bin/env python
# Check last_conv_layer_name in classifier
with open('app/inference/classifier.py', 'r') as f:
    content = f.read()

# Find all occurrences of last_conv_layer_name
import re
matches = re.finditer(r'last_conv_layer_name', content)
for m in matches:
    start = max(0, m.start() - 150)
    end = min(len(content), m.end() + 150)
    print(f'Found at position {m.start()}:')
    print(content[start:end])
    print('---')
PYEOF