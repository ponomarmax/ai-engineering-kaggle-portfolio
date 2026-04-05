# AI Engineering Kaggle Portfolio

This repository is structured around the 13-course IBM AI Engineering Professional Certificate and the Kaggle projects used to practice each course topic.

## Repository structure

```text
.
├── courses/
│   ├── 01-machine-learning-with-python/
│   ├── 02-introduction-to-deep-learning-neural-networks-with-keras/
│   ├── 03-deep-learning-with-keras-and-tensorflow/
│   ├── ...
│   └── 13-project-generative-ai-applications-with-rag-and-langchain/
├── shared/
│   ├── docs/
│   ├── notebooks/
│   └── utils/
└── templates/
    ├── approach/
    └── task/
```

## How the portfolio is organized

Each course folder contains:

- `README.md`: course goals, target skills, and candidate Kaggle tasks
- `tasks/`: portfolio projects mapped to that course

Each Kaggle task contains:

- a task-level `README.md` with business framing, Kaggle link, metrics, and lessons learned
- `approaches/` with multiple solutions for the same task
- `notes/` for experiment summaries, error analysis, and course mapping
- `artifacts/` for exported results, plots, and reports

Each approach contains:

- code and notebooks for one specific solution strategy
- its own `README.md`
- isolated configs, reports, and predictions

## Suggested workflow

1. Pick a course.
2. Add 1-3 Kaggle tasks that match the course outcomes.
3. For each task, start with a simple baseline approach.
4. Add stronger or alternative approaches in separate `approaches/*` folders.
5. Summarize what the task demonstrates for your portfolio.

## Root workflow

The Python environment is shared for the whole repository and lives in the root as `.venv`.

Create it once:

```text
make setup
```

Open a shell with that environment activated:

```text
make shell
```

Then work inside any task folder directly:

```text
cd courses/01-machine-learning-with-python/tasks/task-house-prices
make setup
make run
make submit
```

## Debugging

Workflow:

```text
make setup
```

Then run the VS Code launch config:

```text
Python: Active File (root venv)
```

It will:

- use the root `.venv`
- start from the folder of the currently opened file
- run in the integrated terminal

## Naming conventions

- Course folders: `NN-short-course-name`
- Task folders: `task-<problem-slug>`
- Approach folders: `NN-<method-slug>`

Example:

```text
courses/01-machine-learning-with-python/tasks/task-house-prices/
courses/01-machine-learning-with-python/tasks/task-house-prices/approaches/01-linear-regression-baseline/
courses/01-machine-learning-with-python/tasks/task-house-prices/approaches/02-xgboost-feature-engineering/
```

## Next step

When you share the final course plan, we can map concrete Kaggle competitions and datasets into each course folder and prefill candidate tasks.
