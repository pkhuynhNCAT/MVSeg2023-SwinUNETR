# Minimal 2-D UNet training example from README
# Added to demonstrate bug fix: compute_meandice -> compute_dice

import os
import random
import numpy as np
import torch
import monai
from torch.utils.data import IterableDataset, DataLoader
from monai.transforms import (
    EnsureChannelFirstd,
    CastToTyped,
    Resized,
    NormalizeIntensityd,
    Compose,
    LoadImage,
)

# Example settings
DATA = "./data"  # path with *_image.nii and *_label.nii pairs
VALVE = [4, 5]   # labels for AV & MV
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class SliceStream(IterableDataset):
    """Yield 2D image slices one by one."""

    def __init__(self, stems: list[str], keep_empty_prob: float = 0.1) -> None:
        self.stems = stems
        self.keep_empty = keep_empty_prob
        self.pre = Compose(
            [
                EnsureChannelFirstd(keys=("img", "lab")),
                CastToTyped(keys="img", dtype=torch.float16),
                Resized(keys=("img", "lab"), spatial_size=(256, 256)),
                NormalizeIntensityd(keys="img", channel_wise=True),
            ]
        )

    def __iter__(self):
        for stem in self.stems:
            img_vol = LoadImage()(f"{DATA}/{stem}_image.nii")
            lab_vol = LoadImage()(f"{DATA}/{stem}_label.nii")
            depth = img_vol.shape[-1]
            for z in range(depth):
                lab_slice = np.asarray(lab_vol[..., z])
                if lab_slice.max() == 0 and random.random() > self.keep_empty:
                    continue
                img_slice = np.asarray(img_vol[..., z])
                sample = {"img": img_slice, "lab": lab_slice}
                sample = self.pre(sample)
                yield {
                    "img": torch.as_tensor(sample["img"]),
                    "lab": torch.as_tensor(sample["lab"]).long(),
                }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=3,
        channels=(8, 16, 32),
        strides=(2, 2),
        num_res_units=1,
    ).to(device)

    loss_fn = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.AdamW(model.parameters(), 3e-4)
    scaler = torch.cuda.amp.GradScaler()

    stems = []  # fill with case IDs without suffix
    dl = DataLoader(SliceStream(stems), batch_size=1, num_workers=0)

    dice_metric = monai.metrics.DiceMetric(include_background=False)

    model.train()
    for b in dl:
        x = b["img"].to(device)
        y = b["lab"].to(device)
        y = torch.where(torch.isin(y, torch.tensor(VALVE, device=device)), y, 0)
        with torch.cuda.amp.autocast():
            logits = model(x)
            loss = loss_fn(logits, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

    # Example evaluation
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast():
        dices = []
        for b in dl:
            x = b["img"].to(device)
            y = b["lab"].to(device)
            y = torch.where(torch.isin(y, torch.tensor(VALVE, device=device)), y, 0)
            preds = model(x)
            dice_val = monai.metrics.compute_dice(
                preds, y, include_background=False
            ).mean()
            dices.append(dice_val)
        if dices:
            print("Dice:", torch.stack(dices).mean().item())


if __name__ == "__main__":
    main()
