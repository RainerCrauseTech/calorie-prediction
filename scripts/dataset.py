import os
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torch.utils.data import Dataset


def map_ingredient_ids(raw: str, id2name: dict) -> str:
    """ingr_0000000122;ingr_0000000026 -> 'salmon, zucchini'."""
    names = []
    for token in str(raw).split(";"):
        token = token.strip()
        if not token:
            continue
        num = str(int(token.replace("ingr_", "")))
        names.append(id2name.get(num, token))
    return ", ".join(names)


def prepare_dish_table(config) -> pd.DataFrame:
    dish = pd.read_csv(config.DISH_CSV)
    ingredients = pd.read_csv(config.INGR_CSV)
    id2name = {str(int(i)): name for i, name in zip(ingredients["id"], ingredients["ingr"])}
    dish = dish.copy()
    dish["text"] = dish["ingredients"].apply(lambda x: map_ingredient_ids(x, id2name))
    images_dir = Path(config.IMAGES_DIR)

    def has_image(dish_id: str) -> bool:
        return (images_dir / str(dish_id) / "rgb.png").exists()

    dish = dish.loc[dish["dish_id"].apply(has_image)].reset_index(drop=True)
    return dish


class MultimodalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, config, transforms):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.transforms = transforms
        self.images_dir = Path(config.IMAGES_DIR)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["text"]
        label = float(row["total_calories"])
        mass = float(row["total_mass"])

        dish_id = str(row["dish_id"])
        img_path = os.path.join(self.images_dir, dish_id, "rgb.png")
        image = Image.open(img_path).convert("RGB")
        image = self.transforms(image=np.array(image))["image"]

        return {
            "label": label,
            "mass": mass,
            "image": image,
            "text": text,
            "dish_id": dish_id,
        }


def collate_fn(batch, tokenizer, max_length=64):
    texts = [item["text"] for item in batch]
    images = torch.stack([item["image"] for item in batch])
    labels = torch.FloatTensor([item["label"] for item in batch])
    mass = torch.FloatTensor([item["mass"] for item in batch])
    dish_ids = [item["dish_id"] for item in batch]

    tokenized_input = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    return {
        "label": labels,
        "mass": mass,
        "image": images,
        "input_ids": tokenized_input["input_ids"],
        "attention_mask": tokenized_input["attention_mask"],
        "text": texts,
        "dish_id": dish_ids,
    }


def get_transforms(config, ds_type="train"):
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)

    if ds_type == "train":
        transforms = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
                A.RandomCrop(
                    height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
                A.Affine(scale=(0.8, 1.2),
                         rotate=(-15, 15),
                         translate_percent=(-0.1, 0.1),
                         shear=(-10, 10),
                         fill=0,
                         p=0.8),
                A.CoarseDropout(num_holes_range=(2, 8),
                                hole_height_range=(int(0.07 * cfg.input_size[1]),
                                                   int(0.15 * cfg.input_size[1])),
                                hole_width_range=(int(0.1 * cfg.input_size[2]),
                                                  int(0.15 * cfg.input_size[2])),
                                fill=0,
                                p=0.5),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                ToTensorV2(p=1.0),
            ],
            seed=42,
        )
    else:
        transforms = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
                A.CenterCrop(
                    height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                ToTensorV2(p=1.0),
            ]
        )

    return transforms
