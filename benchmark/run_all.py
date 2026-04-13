import importlib
import sys
from pathlib import Path


# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd  # noqa: E402
from experiments import load_experiments  # noqa: E402


def main():
    experiments = load_experiments()
    results = []

    for exp_name, config in experiments.items():
        print(f"Running experiments for: {exp_name}")

        module = importlib.import_module(config["module"])
        entrypoint = getattr(module, config["entrypoint"])

        for factor in config["factors"]:
            print(f"  Testing {factor}...")

            try:
                res = entrypoint(factor)
                row = {"experiment": exp_name}
                row.update(factor)
                row.update(res)
                results.append(row)
            except Exception as e:
                print(f"Failed configuration: {factor} - {e}")

    df = pd.DataFrame(results)

    csv_path = Path(__file__).parent / "latest_results.csv"

    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
