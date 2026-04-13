from pathlib import Path

import yaml


# This file centrally defines the benchmark matrices.
# A run_all script will parse these configs and run the associated methods.

EXPERIMENTS = {
    # 1. circuit_benchmark.py
    "circuit": {
        "module": "benchmark.circuit_benchmark",
        "entrypoint": "run_benchmark_config",
        "factors": [
            {"num_features": 16, "batch_size": 512, "classes": 10},
            {"num_features": 32, "batch_size": 512, "classes": 10},
            {"num_features": 64, "batch_size": 512, "classes": 10},
            {"num_features": 128, "batch_size": 512, "classes": 10},
        ],
    },
    # 2. monarch_benchmark.py
    "monarch": {
        "module": "benchmark.monarch_benchmark",
        "entrypoint": "run_benchmark_config",
        "factors": [
            # configs: b, c, k, b1
            {"b": 16, "c": 16, "k": 16, "b1": 16, "batch_size": 128},
            {"b": 32, "c": 32, "k": 32, "b1": 32, "batch_size": 128},
            {"b": 64, "c": 64, "k": 64, "b1": 64, "batch_size": 128},
            {"b": 32, "c": 32, "k": 64, "b1": 64, "batch_size": 128},
        ],
    },
}


def load_experiments():
    return EXPERIMENTS
