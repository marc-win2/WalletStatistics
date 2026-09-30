#!/usr/bin/env bash

# Run all four algorithms on both workloads. NUM_ITER counts payments per run.
# Override NUM_RUNS, NUM_ITER, SEED, OUTPUT_ROOT, or PYTHON_BIN as needed.
set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_command="${PYTHON_BIN:-python3}"
num_runs="${NUM_RUNS:-100}"
num_iter="${NUM_ITER:-100000}"
seed="${SEED:-12345}"
output_root="${OUTPUT_ROOT:-$script_directory/Simulations/Crosscheck}"

for scenario in gaussian dirichletFloat; do
    for algorithm in lvf random boltzmann greedy; do
        echo "Running $scenario / $algorithm ($num_runs runs, $num_iter payments each)"
        "$python_command" "$script_directory/main.py" \
            --coin-selection-strategy "$algorithm" \
            --beta-adjustment-mode microcanonicalApprox \
            --transaction-scenario "$scenario" \
            --num_runs "$num_runs" \
            --num_iter "$num_iter" \
            --seed "$seed" \
            --output-path "$output_root/$scenario/$algorithm"
    done
done
