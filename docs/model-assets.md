# Production model assets

The Git source contains [the asset manifest](../models/production-assets.json),
not the four binary checkpoints required by the default live/API pipeline:

- `artifacts/person_detector/yolo11n.pt`
- `artifacts/face_detector/face_detection_yunet_2023mar.onnx`
- `artifacts/gender_classifier/face_gender_classifier_mobilenet_v3_large.pth`
- `artifacts/body_gender_classifier/body_gender_classifier_mobilenet_v3_small.pth`

Each declared file has an expected size and SHA-256 checksum. Verify a prepared
checkout before starting the application or API:

```powershell
python tools/prepare_production_assets.py
```

To provision a new checkout, put the files in a trusted local directory using
the same relative paths, then copy only missing assets:

```powershell
python tools/prepare_production_assets.py --source-dir D:\trusted-crowd-models
```

The command never downloads a model and never overwrites an existing file by
default. If a local checkpoint is known to be invalid and the source is trusted,
replacement requires an explicit opt-in:

```powershell
python tools/prepare_production_assets.py --source-dir D:\trusted-crowd-models --overwrite
```

Use the manifest checksum as the source of truth; do not bypass a failed
verification by renaming or substituting a model file.
