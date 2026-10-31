import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "scripts"))

from config import Config
from utils import train


if __name__ == "__main__":
    model, best_val_mae = train(Config())
    print("лучшая Val MAE:", round(best_val_mae, 3), flush=True)
