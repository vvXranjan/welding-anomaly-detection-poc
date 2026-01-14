from pathlib import Path
import shutil, random

random.seed(42)

DS = Path("data/The Welding Defect Dataset - v2/The Welding Defect Dataset - v2")
if not DS.exists():
    raise SystemExit(f"❌ Dataset root not found: {DS}")

print("✅ Using dataset root:", DS)

out_train = Path("datasets/weld/train/good")
out_test_good = Path("datasets/weld/test/good")
out_test_bad = Path("datasets/weld/test/bad")

out_train.mkdir(parents=True, exist_ok=True)
out_test_good.mkdir(parents=True, exist_ok=True)
out_test_bad.mkdir(parents=True, exist_ok=True)

img_exts = {".jpg", ".jpeg", ".png"}

def gather(split_name):
    img_dir = DS / split_name / "images"
    lab_dir = DS / split_name / "labels"
    imgs = [p for p in img_dir.rglob("*") if p.suffix.lower() in img_exts]
    return imgs, lab_dir

def looks_good(img_path: Path) -> bool:
    n = img_path.name.lower()
    return ("good" in n) or ("mig-good-welds" in n)

def label_has_boxes(img_path: Path, labels_dir: Path) -> bool:
    lab = labels_dir / (img_path.stem + ".txt")
    if not lab.exists():
        return False
    return lab.read_text().strip() != ""

good, bad = [], []

for split in ["train", "valid", "test"]:
    if (DS / split / "images").exists():
        imgs, labdir = gather(split)
        for img in imgs:
            has_boxes = label_has_boxes(img, labdir)
            if looks_good(img):
                good.append(img)
            else:
                (bad if has_boxes else good).append(img)

print("Found good:", len(good))
print("Found bad :", len(bad))

random.shuffle(good)
split_idx = int(0.8 * len(good))
train_good = good[:split_idx]
test_good = good[split_idx:]

def copy_list(items, dst):
    for p in items:
        shutil.copy2(p, dst / p.name)

copy_list(train_good, out_train)
copy_list(test_good, out_test_good)
copy_list(bad, out_test_bad)

print("✅ Written anomalib folders:")
print("train good:", len(train_good))
print("test good :", len(test_good))
print("test bad  :", len(bad))

