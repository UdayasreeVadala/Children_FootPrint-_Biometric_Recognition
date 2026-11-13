import urllib.request, json

# Check enrolled children
try:
    r = urllib.request.urlopen('http://127.0.0.1:8000/api/enrolled', timeout=15)
    enrolled = json.loads(r.read().decode())
    print('=== ENROLLED CHILDREN ===')
    children = enrolled.get('children', [])
    for c in children:
        cid = c.get('child_id', '')
        cname = c.get('child_name', '')
        ic = c.get('image_count', 0)
        print(f'  ID: {cid}, Name: {cname}, Images: {ic}')
    print(f'Total enrolled: {len(children)}')
except Exception as e:
    print('Error getting enrolled:', e)

# Check model-info
try:
    r = urllib.request.urlopen('http://127.0.0.1:8000/api/model-info', timeout=15)
    info = json.loads(r.read().decode())
    print()
    print('=== MODEL INFO ===')
    for k, v in info.items():
        print(f'  {k}: {v}')
except Exception as e:
    print('Error getting model-info:', e)
PYEOF