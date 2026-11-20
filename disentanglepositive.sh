#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Stage 2: Align 4-modality embeddings and train disentangled representations
EMBEDDINGS_DIR="data"
OUTPUT_DIR="results"
MODEL_CONFIG="config/model.json"

usage() {
    cat <<EOF
Usage: bash disentanglepositive.sh [options]

Options:
  --embeddings-dir DIR   4-modality embeddings directory (default: data)
  --output-dir DIR       Output directory (default: results)
  --config PATH          Model config JSON path (default: config/model.json)
  -h, --help             Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --embeddings-dir)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
            EMBEDDINGS_DIR="$2"
            shift 2
            ;;
        --embeddings-dir=*)
            EMBEDDINGS_DIR="${1#*=}"
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
        --config)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
            MODEL_CONFIG="$2"
            shift 2
            ;;
        --config=*)
            MODEL_CONFIG="${1#*=}"
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

# 1. Align and concatenate multimodal embeddings based on H3 indices
python -m IntraModality.concat     --data-dir "$EMBEDDINGS_DIR"     --output-dir "$OUTPUT_DIR/aligned"

# 2. Disentangle multimodal representations
python -m DisentanglePositive.main     --data-dir "$OUTPUT_DIR/aligned"     --config "$MODEL_CONFIG"     --output-dir "$OUTPUT_DIR/representations"
