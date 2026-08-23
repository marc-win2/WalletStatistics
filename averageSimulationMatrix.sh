#!/usr/bin/env bash

set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
matrix_path="$script_directory/Simulations/CoinSelectionMatrix"
num_runs=100
num_payments=100000
chunk_size=20000
python_command="${PYTHON_BIN:-python3}"
configuration_paths=()

show_usage() {
    cat <<'EOF'
Usage: ./averageSimulationMatrix.sh [options]

Discover and average every simulation configuration below a matrix directory,
or average only explicitly selected configuration paths.

Options:
  --matrix-path PATH       Root searched for configurations containing Data/
  --configuration PATH    Configuration directory, repeatable; relative paths
                          are resolved below --matrix-path
  --num-runs NUMBER        Runs generated per configuration (default: 100)
  --num-payments NUMBER    Payments generated per run (default: 100000)
  --chunk-size NUMBER      Rows processed per chunk (default: 20000)
  -h, --help               Show this help

Examples:
  ./averageSimulationMatrix.sh \
    --matrix-path Simulations/CoinSelectionMatrix

  ./averageSimulationMatrix.sh \
    --matrix-path Simulations/CoinSelectionMatrix \
    --configuration RAGFit/Gaussian \
    --configuration BranchAndBound/DirichletFloat
EOF
}

require_value() {
    if [[ $# -lt 2 || -z "$2" ]]; then
        echo "Missing value for $1" >&2
        show_usage >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --matrix-path)
            require_value "$@"
            matrix_path="$2"
            shift 2
            ;;
        --configuration|--config)
            require_value "$@"
            configuration_paths+=("$2")
            shift 2
            ;;
        --num-runs)
            require_value "$@"
            num_runs="$2"
            shift 2
            ;;
        --num-payments)
            require_value "$@"
            num_payments="$2"
            shift 2
            ;;
        --chunk-size)
            require_value "$@"
            chunk_size="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            show_usage >&2
            exit 2
            ;;
    esac
done

configurations=()
if [[ ${#configuration_paths[@]} -eq 0 ]]; then
    if [[ ! -d "$matrix_path" ]]; then
        echo "Matrix directory does not exist: $matrix_path" >&2
        exit 1
    fi

    while IFS= read -r -d '' data_path; do
        configurations+=("${data_path%/Data}")
    done < <(find "$matrix_path" -type d -name Data -print0 | sort -z)
else
    for configuration_path in "${configuration_paths[@]}"; do
        if [[ "$configuration_path" = /* ]]; then
            resolved_path="$configuration_path"
        else
            resolved_path="$matrix_path/$configuration_path"
        fi

        if [[ "${resolved_path##*/}" == "Data" ]]; then
            resolved_path="${resolved_path%/Data}"
        fi
        configurations+=("$resolved_path")
    done
fi

if [[ ${#configurations[@]} -eq 0 ]]; then
    echo "No simulation configurations containing Data/ found." >&2
    exit 1
fi

for configuration_path in "${configurations[@]}"; do
    data_path="$configuration_path/Data"
    save_path="$configuration_path/DataGlobal"

    if [[ ! -d "$data_path" ]]; then
        echo "Missing simulation data directory: $data_path" >&2
        exit 1
    fi

    mkdir -p "$save_path"
    echo "Averaging $configuration_path"
    "$python_command" "$script_directory/averageSimulationPlots.py" \
        --num_runs "$num_runs" \
        --num_payments "$num_payments" \
        --chunk_size "$chunk_size" \
        --data_path "$data_path" \
        --save_path "$save_path"
done
