from app.inference.dataset import discover_dataset, resolve_task

samples = discover_dataset('Training_Data', include_sensors=('Paper', 'Scanner', 'Camera', 'Phone'))
task = 'identity'

# Check label_for for each sample
for s in samples[:5]:
    lbl_identity = s.label_for(task)
    lbl_side = s.label_for("side")
    lbl_sensor = s.label_for("sensor")
    lbl_visit = s.label_for("visit")
    print(f'Sample: subject={s.subject}, visit={s.visit}, sensor={s.sensor}, side={s.side}')
    print(f'  label_for(identity)={lbl_identity}')
    print(f'  label_for(side)={lbl_side}')
    print(f'  label_for(sensor)={lbl_sensor}')
    print(f'  label_for(visit)={lbl_visit}')
    print()

# Now check all samples
subjects_with_labels = {}
for s in samples:
    lbl = s.label_for(task)
    if lbl not in subjects_with_labels:
        subjects_with_labels[lbl] = []
    subjects_with_labels[lbl].append(s)

print(f"Number of distinct identity labels: {len(subjects_with_labels)}")
for lbl, samp_list in subjects_with_labels.items():
    subj = samp_list[0].subject
    count = len(samp_list)
    print(f'  {lbl}: {count} images, subject={subj}')
PYEOF