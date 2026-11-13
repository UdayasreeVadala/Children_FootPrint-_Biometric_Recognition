from app.inference.dataset import discover_dataset, resolve_task
from app.inference.training import train_dataset_models, grouped_split, summarize_samples, class_distribution, load_manifest, preprocess_dataset, save_manifest
import os

# Step 1: Discover dataset
samples = discover_dataset('Training_Data', include_sensors=('Paper', 'Scanner', 'Camera', 'Phone'))
print(f"1. Discovered {len(samples)} samples")

# Step 2: Resolve task
resolved_task = resolve_task(samples, 'auto')
print(f"2. Resolved task: {resolved_task}")

# Step 3: Check class names from samples
class_names_from_samples = sorted({s.label_for(resolved_task) for s in samples})
print(f"3. Class names from samples: {class_names_from_samples} ({len(class_names_from_samples)} classes)")

# Step 4: Check if manifest exists
manifest_path = 'data/cache/dataset_224/dataset_manifest.json'
if os.path.exists(manifest_path):
    manifest = load_manifest(manifest_path)
    print(f"3b. Manifest exists with {len(manifest)} samples")
    class_names_from_manifest = sorted({s.label_for(resolved_task) for s in manifest})
    print(f"3b. Class names from manifest: {class_names_from_manifest} ({len(class_names_from_manifest)} classes)")
else:
    print(f"3b. No manifest file at {manifest_path}")
    
    # Step 4: Generate manifest
    print("4. Generating manifest...")
    manifest = preprocess_dataset(samples, Path('data/cache/dataset_224'), force=False)
    print(f"4a. Preprocessed {len(manifest)} samples")
    save_manifest(manifest, manifest_path)
    print(f"4b. Saved manifest with {len(manifest)} samples")
    
    # Now check class names from the new manifest
    class_names_from_manifest = sorted({s.label_for(resolved_task) for s in manifest})
    print(f"4c. Class names from new manifest: {class_names_from_manifest} ({len(class_names_from_manifest)} classes)")

# Step 5: Try the actual training
print("\n5. Attempting training...")
try:
    result = train_dataset_models(
        dataset_root='Training_Data',
        models_dir='models',
        cache_dir='data/cache/dataset_224',
        task='auto',
        architectures=['mobilenetv2'],
        epochs=5,
        fine_tune_epochs=2,
        batch_size=16,
        include_sensors=('Paper', 'Scanner', 'Camera', 'Phone'),
        verbose=1,
    )
    print("5. Training succeeded!")
except Exception as e:
    print(f"5. Training failed: {type(e).__name__}: {e}")
PYEOF