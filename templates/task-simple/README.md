# Kaggle Task Template

Скопіюй цю папку, перейменуй її в `task-<your-task-name>` і заміни заповнювачі під свою задачу. Цей шаблон відображає реальний робочий формат: чистий baseline у коді, нотатки окремо, і керування через `Makefile`.

## Structure

```text
task-<your-task-name>/
├── README.md
├── Makefile
├── requirements.txt
├── main.py
├── notes/
│   ├── learning-journal.md
│   └── submission-log.md
├── data/
│   └── raw/
└── submissions/
```

## Quick start

1. Скопіюй папку шаблону в нову задачу.
2. Онови `COMPETITION` у `Makefile`.
3. Замініть `TARGET` та `ID_COLUMN` у `main.py`.
4. Онови опис у цьому `README.md`.
5. З кореня репозиторію вкажи нову задачу в `.current-task`.
6. Запусти `make setup`, `make download`, `make run`.

## Kaggle

- Link: `https://www.kaggle.com/competitions/<competition-slug>`
- Metric:
- Target: `target_column`

## Current structure

- `main.py` - чистий baseline training script для локального запуску і генерації submission
- `notes/learning-journal.md` - спостереження під час першого проходу по даних
- `notes/submission-log.md` - коротка історія сабмітів і змін між ними
- `submissions/` - згенеровані файли для Kaggle

## Current baseline

- використовує тільки числові ознаки
- заповнює пропуски медіаною
- масштабує фічі через `StandardScaler`
- навчає `LinearRegression`

Це стартовий baseline, який легко читати, міняти й порівнювати з наступними ітераціями.

## Workflow

Після того як шлях до задачі записаний у `.current-task`, можна працювати з кореня репозиторію:

```text
make setup
make download
make run
make submit MESSAGE="first baseline"
```

## Why this format works

- `main.py` не перетворюється на кладовище закоментованого коду
- навчальні спостереження не губляться
- лог сабмітів показує еволюцію рішень
- нову задачу легко почати за однаковим шаблоном
