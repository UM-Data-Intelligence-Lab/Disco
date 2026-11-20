#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Stage 1: Intra-modality H3 mapping and representation enhancement (raw data -> 4-modality embeddings)
RAW_DATA_DIR="raw_data"
OUTPUT_DIR="results"
CITY="singapore"

usage() {
    cat <<EOF
Usage: bash intramodality.sh [options]

Options:
  --raw-data-dir DIR   Raw input data directory (default: raw_data)
  --output-dir DIR     Output directory (default: results)
  --city NAME          City name (default: singapore)
  -h, --help           Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --raw-data-dir)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
            RAW_DATA_DIR="$2"
            shift 2
            ;;
        --raw-data-dir=*)
            RAW_DATA_DIR="${1#*=}"
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
        --city)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
            CITY="$2"
            shift 2
            ;;
        --city=*)
            CITY="${1#*=}"
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

# 1. Map raw data from four modalities to a unified H3 grid
python -m IntraModality.poi.map_h3 "$CITY"     --source-dir "$RAW_DATA_DIR/poi"     --regions "$RAW_DATA_DIR/regions.csv"     --output-root "$OUTPUT_DIR/processed"

python -m IntraModality.mobility_flow.map_h3 "$CITY"     --source-csv "$RAW_DATA_DIR/flow.csv"     --regions "$RAW_DATA_DIR/regions.csv"     --output-dir "$OUTPUT_DIR/processed"

python -m IntraModality.map_images_to_h3 streetview     "$RAW_DATA_DIR/streetview.csv"     "$RAW_DATA_DIR/regions.csv"     "$OUTPUT_DIR/processed/$CITY"

python -m IntraModality.map_images_to_h3 satellite     "$RAW_DATA_DIR/satellite.tif"     "$RAW_DATA_DIR/regions.csv"     "$OUTPUT_DIR/processed/$CITY"

# 2. Generate embeddings using four modality-specific encoders
python -m IntraModality.poi.main     --city "$CITY"     --data-dir "$OUTPUT_DIR/processed"     --output-dir "$OUTPUT_DIR/embeddings"

python -m IntraModality.mobility_flow.main     --city "$CITY"     --data-dir "$OUTPUT_DIR/processed"     --output-dir "$OUTPUT_DIR/embeddings"

python -m IntraModality.streetview.main     --city "$CITY"     --data-dir "$OUTPUT_DIR/processed"     --output-dir "$OUTPUT_DIR/embeddings"

python -m IntraModality.satellite.main     --city "$CITY"     --data-dir "$OUTPUT_DIR/processed"     --output-dir "$OUTPUT_DIR/embeddings"
