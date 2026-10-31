import os
import random
from functools import partial

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torchmetrics
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from dataset import (
    MultimodalDataset,
    collate_fn,
    get_transforms,
    prepare_dish_table,
)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True


def set_requires_grad(module: nn.Module, unfreeze_pattern="", verbose=False):
    if len(unfreeze_pattern) == 0:
        for _, param in module.named_parameters():
            param.requires_grad = False
        return

    pattern = unfreeze_pattern.split("|")
    n_on = 0
    n_all = 0
    for name, param in module.named_parameters():
        n_all += 1
        if any(name.startswith(p) for p in pattern):
            param.requires_grad = True
            n_on += 1
            if verbose:
                print(f"Разморожен слой: {name}", flush=True)
        else:
            param.requires_grad = False
    print(f"разморожено {n_on}/{n_all} тензоров", flush=True)


def _trainable(module: nn.Module):
    return [p for p in module.parameters() if p.requires_grad]


class MultimodalModel(nn.Module):
    """Как в Theme 3: два энкодера, две проекции, фьюжн, classifier.

    Голова Linear(..., 1) без активации. Калории = плотность * масса / MASS_SCALE,
    иначе сеть игнорирует размер порции (это и ломало MAE на жирных блюдах).
    """

    def __init__(self, config):
        super().__init__()
        self.mass_scale = float(getattr(config, "MASS_SCALE", 100.0))

        self.text_encoder = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.image_encoder = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0,
        )

        self.text_projection = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
        )
        self.image_projection = nn.Sequential(
            nn.Linear(self.image_encoder.num_features, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
        )

        fused_dim = config.HIDDEN_DIM * 3
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM // 2, 1),
        )

    def _text_features(self, input_ids, attention_mask):
        hidden = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        mask = attention_mask.unsqueeze(-1).type_as(hidden)
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def forward(self, input_ids, attention_mask, image, mass):
        text_features = self._text_features(input_ids, attention_mask)
        image_features = self.image_encoder(image)

        text_emb = self.text_projection(text_features)
        image_emb = self.image_projection(image_features)
        fused_mul = text_emb * image_emb
        fused_emb = torch.cat([text_emb, image_emb, fused_mul], dim=1)

        density = self.classifier(fused_emb).squeeze(-1)
        return density * (mass / self.mass_scale)


def split_train_val(train_df: pd.DataFrame, val_fraction: float, seed: int):
    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = max(1, int(len(train_df) * val_fraction))
    val_part = train_df.iloc[:n_val].reset_index(drop=True)
    train_part = train_df.iloc[n_val:].reset_index(drop=True)
    return train_part, val_part


def _batch_inputs(batch, device):
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "image": batch["image"].to(device),
        "mass": batch["mass"].to(device),
    }


def train(config):
    seed_everything(config.SEED)
    device = get_device()
    print("device:", device, flush=True)

    model = MultimodalModel(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    set_requires_grad(
        model.text_encoder,
        unfreeze_pattern=config.TEXT_MODEL_UNFREEZE,
        verbose=True,
    )
    set_requires_grad(
        model.image_encoder,
        unfreeze_pattern=config.IMAGE_MODEL_UNFREEZE,
        verbose=True,
    )

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_train:,} / {n_all:,}", flush=True)

    optimizer = AdamW(
        [
            {"params": _trainable(model.text_encoder), "lr": config.TEXT_LR},
            {"params": _trainable(model.image_encoder), "lr": config.IMAGE_LR},
            {"params": model.text_projection.parameters(), "lr": config.CLASSIFIER_LR},
            {"params": model.image_projection.parameters(), "lr": config.CLASSIFIER_LR},
            {"params": model.classifier.parameters(), "lr": config.CLASSIFIER_LR},
        ],
        weight_decay=getattr(config, "WEIGHT_DECAY", 0.01),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)
    criterion = nn.L1Loss()

    table = prepare_dish_table(config)
    train_full = table[table["split"] == "train"].reset_index(drop=True)
    train_df, val_df = split_train_val(train_full, config.VAL_FRACTION, config.SEED)

    transforms = get_transforms(config)
    val_transforms = get_transforms(config, ds_type="val")
    train_dataset = MultimodalDataset(train_df, config, transforms)
    val_dataset = MultimodalDataset(val_df, config, val_transforms)

    collate = partial(collate_fn, tokenizer=tokenizer, max_length=config.MAX_LENGTH)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate,
    )

    mae_train = torchmetrics.MeanAbsoluteError().to(device)
    mae_val = torchmetrics.MeanAbsoluteError().to(device)
    best_mae_val = float("inf")
    grad_clip = float(getattr(config, "GRAD_CLIP", 1.0))

    os.makedirs(os.path.dirname(config.SAVE_PATH), exist_ok=True)

    print("training started", flush=True)
    print(f"train: {len(train_dataset)} | val: {len(val_dataset)}", flush=True)

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            inputs = _batch_inputs(batch, device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            preds = model(**inputs)
            loss = criterion(preds, labels)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()
            mae_train.update(preds, labels)

        scheduler.step()
        train_mae = float(mae_train.compute().cpu().numpy())
        val_mae = validate(model, val_loader, device, mae_val)
        mae_train.reset()
        mae_val.reset()

        print(
            f"Epoch {epoch}/{config.EPOCHS - 1} | avg_Loss: {total_loss / len(train_loader):.4f} "
            f"| Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}",
            flush=True,
        )

        if val_mae < best_mae_val:
            print(f"New best model, epoch: {epoch}", flush=True)
            best_mae_val = val_mae
            torch.save(model.state_dict(), config.SAVE_PATH)

    return model, best_mae_val


def validate(model, val_loader, device, mae_metric):
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            inputs = _batch_inputs(batch, device)
            labels = batch["label"].to(device)
            preds = model(**inputs)
            mae_metric.update(preds, labels)
    return float(mae_metric.compute().cpu().numpy())
