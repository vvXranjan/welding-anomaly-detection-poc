from pathlib import Path
import sys
import numpy as np
import cv2
import torch
from PIL import Image
import torchvision.transforms.v2 as v2

from anomalib.models import Patchcore

# ---------------- CONFIG ----------------
CKPT_PATH = "results/Patchcore/weld/v1/weights/lightning/model.ckpt"

IN_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/test_normal")
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- UTILS ----------------
def overlay_heatmap(img_bgr, heatmap01):
    heatmap01 = cv2.resize(heatmap01, (img_bgr.shape[1], img_bgr.shape[0]))
    heat = (255 * heatmap01).clip(0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 0.65, heat_color, 0.35, 0)

# ---------------- MAIN ----------------
def main():
    print("Loading checkpoint:", CKPT_PATH)
    model = Patchcore.load_from_checkpoint(CKPT_PATH)
    model.eval()

    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((256, 256)),
        v2.ToDtype(torch.float32, scale=True),
    ])

    # Collect images from input directory
    img_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        img_paths.extend(IN_DIR.rglob(ext))

    if len(img_paths) == 0:
        print("No images found in:", IN_DIR)
        return

    print(f"Running inference on {len(img_paths)} images from {IN_DIR}")

    for img_path in img_paths:
        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        x = transform(img_rgb).unsqueeze(0)

        with torch.no_grad():
            out = model(x)

        score = float(out.pred_score.item())
        anomaly_map = out.anomaly_map.squeeze().cpu().numpy()
        anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)

        vis = overlay_heatmap(
            cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
            anomaly_map
        )

        out_path = OUT_DIR / f"{img_path.stem}_score_{score:.3f}.jpg"
        cv2.imwrite(str(out_path), vis)
        print("Saved:", out_path)

# ---------------- ENTRY ----------------
if __name__ == "__main__":
    main()


