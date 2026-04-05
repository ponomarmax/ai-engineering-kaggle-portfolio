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

- `main.py` - чистий baseline training script для локального запуску і генерації submission
- `notes/learning-journal.md` - ключові спостереження, які з'явилися під час першого проходу по даних
- `notes/submission-log.md` - коротка історія сабмітів і змін між ними
- `submissions/` - згенеровані файли для Kaggle

## Current baseline

- використовує тільки числові ознаки
- заповнює пропуски медіаною
- масштабує фічі через `StandardScaler`
- навчає `LinearRegression`

Це не "найкраща" модель, а свідомо простий, зрозумілий стартовий baseline, від якого легко рухатися далі.

## Why this format works

- `main.py` не перетворюється на кладовище закоментованого коду
- навчальні спостереження не губляться
- еволюцію рішень можна показати як частину портфоліо, а не ховати в історії комітів
