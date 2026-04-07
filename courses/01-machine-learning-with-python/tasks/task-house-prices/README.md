# House Prices Competition for Kaggle Learn Users

Це моя перша Kaggle задача і перший успішний submission. Я залишив структуру простою, але розділив код, нотатки і результати так, щоб проєкт було приємно перечитувати і мені, і людям, які дивляться портфоліо.

## Quick start

1. `cd courses/01-machine-learning-with-python/tasks/task-house-prices`
2. `make setup`
3. Поклади `train.csv` і `test.csv` у `data/raw/`
4. Пиши рішення в `main.py`
5. `make run`

## Kaggle

- Link: https://www.kaggle.com/competitions/home-data-for-ml-course
- Metric: RMSLE
- Target: `SalePrice`

## Idea

- Ця папка має бути зразком для копіювання під інші задачі.
- Якщо захочеш нову задачу, просто скопіюй цю структуру або `templates/task-simple/`.

## Current structure

- `main.py` - numeric baseline pipeline + experiment runner for feature comparisons
- `eda.ipynb` - основний EDA notebook з numeric і non-numeric analysis
- `notes/decision.txt` - чорнові рішення по групуванню фіч
- `notes/experiments.txt` - короткі замітки про попередні експерименти
- `notes/learning-journal.md` - ключові спостереження під час проходу по даних
- `notes/submission-log.md` - коротка історія сабмітів і змін між ними
- `submissions/` - згенеровані файли для Kaggle

## Current baseline

- використовує тільки явно оголошені numeric baseline features
- підтримує `log1p`, `Has*` indicators і явне виключення forgotten features
- заповнює пропуски медіаною
- масштабує фічі через `StandardScaler`
- навчає `LinearRegression`

Це все ще не "фінальна найкраща модель", а акуратний експериментальний baseline, від якого зручно рухатися далі.

## Experiment workflow

- базові фічі задаються через групи на початку `main.py`
- forgotten features не додаються мовчки в baseline
- експерименти описуються через `ExperimentConfig`
- можна тестувати:
  - додавання окремих фіч поверх baseline
  - `log1p` для конкретних колонок
  - `Has*` indicators
  - комбіновані фічі на кшталт `TotalArea`
- результати друкуються як компактна таблиця для порівняння

## Why this format works

- `main.py` не перетворюється на кладовище закоментованого коду
- навчальні спостереження не губляться
- еволюцію рішень можна показати як частину портфоліо, а не ховати в історії комітів
