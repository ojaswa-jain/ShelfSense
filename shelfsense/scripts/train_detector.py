"""Fine-tune a one-class retail-facing detector on a user-supplied YOLO dataset."""
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Local YOLO dataset YAML")
    parser.add_argument("--weights", default="yolo11n.pt", help="Trusted local checkpoint or official YOLO11 starter")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.data.is_file() or args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
        parser.error("Supply a local dataset YAML and positive training values.")
    from ultralytics import YOLO, settings
    settings.update({"sync": False})
    model = YOLO(args.weights)
    model.train(data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, workers=0,
                project="outputs/training", name="shelfsense", seed=42)


if __name__ == "__main__":
    main()
