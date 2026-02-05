from pathlib import Path
import sys


def main() -> int:
    try:
        from ultralytics import YOLO
    except Exception as exc:
        print(
            "Failed to import ultralytics. Install it with: pip install ultralytics",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 1

    root = Path(__file__).resolve().parent
    model_path = root / "best.pt"
    if not model_path.exists():
        print(f"Model not found at {model_path}", file=sys.stderr)
        return 1

    input_dir = root / "input"
    output_dir = root / "output"
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = [p for p in input_dir.iterdir() if p.suffix.lower() in image_exts]
    if not images:
        print(f"No images found in {input_dir}")
        return 0

    model = YOLO(str(model_path))

    for img_path in images:
        results = model.predict(source=str(img_path), verbose=False)
        if not results:
            print(f"No result for {img_path.name}")
            continue

        out_path = output_dir / img_path.name
        # Save using Ultralytics helper to preserve correct color format
        results[0].save(filename=str(out_path))
        print(f"Saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
