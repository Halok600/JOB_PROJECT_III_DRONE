# Model Weights

`best.pt` — YOLOv8n-P2 fine-tuned on VisDrone persons dataset.

Training metrics and curves are in `results/aerial_guardian_p2_v1/`.

Not committed to git (binary file). To obtain weights, run training:
```
python scripts/02_train.py
```
or copy from a previous run:
```
Copy-Item "path\to\previous\best.pt" "models\weights\best.pt"
```
