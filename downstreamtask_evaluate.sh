#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Stage 3: Train and evaluate downstream tasks from disentangled representations
REPRESENTATIONS_DIR="checkpoint"
OUTPUT_DIR="results/downstream"
LABELS_PATH="data/singapore_gdp.csv"
LABEL_COLUMN="gdp"
TASK="regression"
DOWNSTREAM_CONFIG="config/downstream.json"
SPLIT_PATH="data/sg_gdp_split.json"

usage() {
    cat <<EOF
Usage: bash downstreamtask_evaluate.sh [options]

Options:
  --representations-dir DIR   Disentangled representations directory (default: checkpoint)
  --output-dir DIR            Output directory (default: results/downstream)
  --labels PATH               Labels CSV path (default: data/singapore_gdp.csv)
  --label-column NAME         Target column name in labels CSV (default: gdp)
  --task TYPE                 regression or classification (default: regression)
  --config PATH               Downstream config JSON path (default: config/downstream.json)
  --split PATH                Split JSON path (default: data/sg_gdp_split.json)
  --no-split                  Disable preset split file and auto-split regions
  -h, --help                  Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --representations-dir)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
            REPRESENTATIONS_DIR="$2"
            shift 2
            ;;
        --representations-dir=*)
            REPRESENTATIONS_DIR="${1#*=}"
            shift
            ;;
        --output-dir)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --output-dir=*)
            OUTPUT_DIR="${1#*=}"
            shift
            ;;
        --labels)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
            LABELS_PATH="$2"
            shift 2
            ;;
        --labels=*)
            LABELS_PATH="${1#*=}"
            shift
            ;;
        --label-column)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
            LABEL_COLUMN="$2"
            shift 2
            ;;
        --label-column=*)
            LABEL_COLUMN="${1#*=}"
            shift
            ;;
        --task)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
            TASK="$2"
            shift 2
            ;;
        --task=*)
            TASK="${1#*=}"
            shift
            ;;
        --config)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
            DOWNSTREAM_CONFIG="$2"
            shift 2
            ;;
        --config=*)
            DOWNSTREAM_CONFIG="${1#*=}"
            shift
            ;;
        --split)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
            SPLIT_PATH="$2"
            shift 2
            ;;
        --split=*)
            SPLIT_PATH="${1#*=}"
            shift
            ;;
        --no-split)
            SPLIT_PATH=""
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

EXTRA_ARGS=()
if [[ -n "$SPLIT_PATH" ]]; then
    EXTRA_ARGS+=(--split "$SPLIT_PATH")
fi
if [[ "$TASK" == "regression" ]]; then
    EXTRA_ARGS+=(--log-target)
fi

python -m DownstreamTask.main     --data-dir "$REPRESENTATIONS_DIR"     --labels "$LABELS_PATH"     --label-column "$LABEL_COLUMN"     --task "$TASK"     --config "$DOWNSTREAM_CONFIG"     --output-dir "$OUTPUT_DIR"     "${EXTRA_ARGS[@]}"
