import torch
import torchvision.transforms.v2 as v2

from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine

def main():
    img_size = 256

    # v2 transforms (what anomalib expects)
    common_aug = v2.Compose([
        v2.ToImage(),
        v2.Resize((img_size, img_size)),
        v2.ToDtype(torch.float32, scale=True),
    ])

    datamodule = Folder(
        name="weld",
        root="datasets/weld",
        normal_dir="train/good",
        abnormal_dir="test/bad",
        normal_test_dir="test/good",
        train_batch_size=16,
        eval_batch_size=1,
        num_workers=2,
        augmentations=common_aug,   # applied to all splits
    )

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
    )

    accelerator = "cpu"
    engine = Engine(accelerator=accelerator, devices=1, max_epochs=1)

    engine.fit(model=model, datamodule=datamodule)
    engine.test(model=model, datamodule=datamodule)

if __name__ == "__main__":
    main()

