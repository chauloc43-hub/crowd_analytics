# Local person-detector asset

`yolo11n.pt` is the detector required by the production FastTracker profile.
The checkpoint binary is intentionally ignored by Git. Provision and verify it
using the contract in [`docs/model-assets.md`](../../docs/model-assets.md),
then keep it at:

```text
artifacts/person_detector/yolo11n.pt
```

The runtime accepts only an existing local path and fails clearly if this file
is absent; it never asks Ultralytics to download a model during a request.
