# Предсказание калорийности блюд

Мультимодальная регрессия: фото блюда + список ингредиентов (+ масса порции) → `total_calories`.

Целевая метрика: **MAE < 50** на `test`.

## Структура

```
calorie-prediction/
  scripts/
    config.py      гиперпараметры и пути
    dataset.py     MultimodalDataset, collate_fn, аугментации
    utils.py       модель, train/validate, разморозка слоёв
  notebook.ipynb   EDA, обучение, проверка на test
  train.py         опциональный запуск обучения из терминала
  artifacts/       сюда сохраняется best_model.pth
```

## Данные

Датасет курса Sprint 4 / Theme 4:

- `dish.csv`
- `ingredients.csv`
- `images/{dish_id}/rgb.png`

Положите их в папку `data/` рядом с ноутбуком или поправьте пути в `scripts/config.py`.

## Запуск

1. Открыть `notebook.ipynb` из корня проекта.
2. Выбрать ядро с установленными `torch`, `timm`, `transformers`, `albumentations`, `torchmetrics`.
3. Прогнать ячейки сверху вниз. Обучение на Mac идёт через MPS.

Или из терминала:

```bash
python train.py
```

## Что в решении

- блоки `text_encoder`, `image_encoder`, `text_projection`, `image_projection`, `classifier`
- голова `Linear(..., 1)`, лосс `nn.L1Loss`
- `BATCH_SIZE = 4`, разные lr в `AdamW`
- id ингредиентов заменяются на названия из `ingredients.csv`
