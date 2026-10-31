from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
if not DATA_DIR.exists():
    DATA_DIR = Path("/Users/krlssnov/Desktop/ЯПрактикум/VS Code/DL/Sprint 4/Theme 4/data")


class Config:
    SEED = 42

    TEXT_MODEL_NAME = "bert-base-uncased"
    IMAGE_MODEL_NAME = "tf_efficientnet_b0"

    # последние слои BERT и хвост EfficientNet
    TEXT_MODEL_UNFREEZE = "encoder.layer.8|encoder.layer.9|encoder.layer.10|encoder.layer.11|pooler"
    IMAGE_MODEL_UNFREEZE = "blocks.4|blocks.5|blocks.6|conv_head|bn2"

    BATCH_SIZE = 4
    TEXT_LR = 2e-5
    IMAGE_LR = 5e-5
    CLASSIFIER_LR = 1e-3
    WEIGHT_DECAY = 0.01
    EPOCHS = 12
    DROPOUT = 0.1
    HIDDEN_DIM = 256
    MAX_LENGTH = 96
    MASS_SCALE = 100.0
    GRAD_CLIP = 1.0

    DISH_CSV = str(DATA_DIR / "dish.csv")
    INGR_CSV = str(DATA_DIR / "ingredients.csv")
    IMAGES_DIR = str(DATA_DIR / "images")
    SAVE_PATH = str(PROJECT_ROOT / "artifacts" / "best_model.pth")
    VAL_FRACTION = 0.15
    NUM_WORKERS = 0
