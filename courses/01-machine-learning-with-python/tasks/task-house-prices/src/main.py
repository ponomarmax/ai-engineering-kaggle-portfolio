from __future__ import annotations

from config import CONFIG
from experiment_runner import run_experiment_suite


def main() -> None:
    summary_df = run_experiment_suite(CONFIG)
    print("Experiment summary for all models:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved experiment outputs under: {CONFIG.experiments_dir}")


if __name__ == "__main__":
    main()
