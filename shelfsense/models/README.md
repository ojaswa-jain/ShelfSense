# Optional weights

Manual mode requires no weights. Place a trusted, retail-trained Ultralytics YOLO detection checkpoint here as `retail_best.pt`, or set its absolute path in the app. Do not commit model weights to Git.

The app checks that the local file exists before loading; it will not silently download or substitute a COCO detector. Train a checkpoint with `scripts/train_detector.py` as explained in the main README. SKU-110K generic product detection does not supply catalogue identities.

ResNet18 embeddings use torchvision's official IMAGENET1K_V1 weights, approximately 44.7 MB, stored in the PyTorch hub cache. The first download requires opt-in in Settings or the documented preload command. Subsequent cached inference is offline.

See the main README for model limitations, licensing sources and CPU/GPU setup. Load only trusted `.pt` files. No model weights are bundled, and optional neural inference has not been benchmarked on real shop images for this release.
