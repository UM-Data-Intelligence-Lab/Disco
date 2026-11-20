from __future__ import annotations

import argparse
import os
import json
import sys
from pathlib import Path
from typing import Any

import h3
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Polygon


BASE_HLY = Path(os.environ.get("DISCO_RAW_ROOT", "raw_data")).resolve()
H3_DIR = BASE_HLY / "1.15/H3_Regions_new"
POI_ROOT = (
    BASE_HLY
    / "Flexibleregion/osm_data/city_unclean_data"
)
OUTPUT_ROOT = (
    BASE_HLY
    / "new_pipeline/data_process"
)

CITY_CONFIGS = {
    "beijing": {
        "h3_candidates": ["beijing_h3_res7.csv"],
        "poi_dir": POI_ROOT / "beijing",
    },
    "shanghai": {
        "h3_candidates": ["shanghai_h3_res7.csv"],
        "poi_dir": POI_ROOT / "shanghai",
    },
    "sg": {
        "h3_candidates": [
            "sg_h3_res7.csv",
            "singapore_h3_res7.csv",
        ],
        "poi_dir": POI_ROOT / "sg",
    },
    "sydney": {
        "h3_candidates": ["sydney_h3_res7.csv"],
        "poi_dir": POI_ROOT / "sydney",
    },
}

CATEGORY_TO_COLUMN = {
    "Food & Drink": "poi_food_drink",
    "Shopping": "poi_shopping",
    "Finance": "poi_finance",
    "Education": "poi_education",
    "Healthcare": "poi_healthcare",
    "Lodging": "poi_lodging",
    "Recreation": "poi_recreation",
    "Transport": "poi_transport",
    "Services": "poi_services",
    "Religious": "poi_religious",
    "Government": "poi_government",
    "Office & Organization": (
        "poi_office_organization"
    ),
    "Other": "poi_other",
}

POI_COLUMNS = list(
    CATEGORY_TO_COLUMN.values()
)


def coord_to_h3(
    lat: float,
    lon: float,
    res: int = 7,
) -> str:
    try:
        return h3.latlng_to_cell(
            float(lat),
            float(lon),
            res,
        )
    except AttributeError:
        return h3.geo_to_h3(
            float(lat),
            float(lon),
            res,
        )


def first_existing(paths):
    return next(
        (
            path
            for path in paths
            if path.exists()
        ),
        None,
    )


def load_h3_list(
    city: str,
    path: Path | None = None,
) -> list[str]:
    if path is None:
        candidates = CITY_CONFIGS[city]["h3_candidates"]
        path = first_existing(H3_DIR / name for name in candidates) or H3_DIR / candidates[0]

    df = pd.read_csv(
        path,
        dtype=str,
    )

    h3_col = None
    for candidate in (
        "h3",
        "h3_index",
        "hex_id",
    ):
        if candidate in df.columns:
            h3_col = candidate
            break

    if h3_col is None:
        raise KeyError(
            f"[{city}] H3 file lacks an ID column. "
            f"found={df.columns.tolist()}"
        )

    values = (
        df[h3_col]
        .dropna()
        .astype(str)
        .str.strip()
    )

    return sorted({
        value
        for value in values
        if h3.is_valid_cell(value)
    })


def iter_jsonl(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def collect_coordinates(
    record: dict,
) -> list[tuple[float, float]]:
    coordinates = []

    geometry = record.get("geometry")

    if isinstance(geometry, list):
        for item in geometry:
            if isinstance(item, dict):
                lon = item.get("lon")
                lat = item.get("lat")
            elif (
                isinstance(item, (list, tuple))
                and len(item) >= 2
            ):
                lon = item[0]
                lat = item[1]
            else:
                continue

            try:
                coordinates.append(
                    (
                        float(lon),
                        float(lat),
                    )
                )
            except (TypeError, ValueError):
                continue

    members = record.get("members")

    if isinstance(members, list):
        for member in members:
            member_geometry = member.get(
                "geometry"
            )

            if not isinstance(
                member_geometry,
                list,
            ):
                continue

            for item in member_geometry:
                if not isinstance(item, dict):
                    continue

                try:
                    coordinates.append(
                        (
                            float(item["lon"]),
                            float(item["lat"]),
                        )
                    )
                except (
                    KeyError,
                    TypeError,
                    ValueError,
                ):
                    continue

    return coordinates


def representative_lonlat(
    record: dict,
) -> tuple[float, float] | None:
    lon = record.get("lon")
    lat = record.get("lat")

    if lon is not None and lat is not None:
        try:
            return float(lon), float(lat)
        except (TypeError, ValueError):
            pass

    coordinates = collect_coordinates(
        record
    )

    if not coordinates:
        return None

    layer = str(record.get("layer", ""))

    try:
        if (
            layer == "areas"
            and len(coordinates) >= 4
            and coordinates[0]
            == coordinates[-1]
        ):
            geometry = Polygon(coordinates)

            if not geometry.is_valid:
                geometry = geometry.buffer(0)
        else:
            geometry = LineString(
                coordinates
            )

        if geometry.is_empty:
            return None

        point = geometry.representative_point()

        return float(point.x), float(point.y)

    except Exception:
        array = np.asarray(
            coordinates,
            dtype=float,
        )

        return (
            float(array[:, 0].mean()),
            float(array[:, 1].mean()),
        )


def load_classifier(
    classifier_dir: Path,
):
    classifier_dir = classifier_dir.resolve()

    if str(classifier_dir) not in sys.path:
        sys.path.insert(
            0,
            str(classifier_dir),
        )

    try:
        from map2cat_cn import (
            CATEGORY_NAMES,
            category_id_to_name,
            classify_osm_tags,
        )
    except ImportError as exc:
        raise ImportError(
            "Cannot import map2cat_cn.py.\n"
            ""
            "\n"
            "--classifier-dir /path/to/classifier"
        ) from exc

    return (
        CATEGORY_NAMES,
        category_id_to_name,
        classify_osm_tags,
    )


def process_city(
    city: str,
    output_root: Path,
    classifier_dir: Path,
    include_lines: bool,
    regions: Path | None = None,
    source_dir: Path | None = None,
) -> None:
    print("\n" + "=" * 80)
    print(f"CITY: {city.upper()}")
    print("=" * 80)

    (
        category_names,
        category_id_to_name,
        classify_osm_tags,
    ) = load_classifier(classifier_dir)

    expected_categories = set(
        CATEGORY_TO_COLUMN
    )
    actual_categories = set(
        category_names
    )

    if expected_categories != actual_categories:
        raise ValueError(
            "map2cat/map2cat_cn categories differ from "
            "POIFeatureBuilder categories.\n"
            f"expected={sorted(expected_categories)}\n"
            f"actual={sorted(actual_categories)}"
        )

    h3_list = load_h3_list(city, regions)
    valid_h3 = set(h3_list)
    resolution = h3.get_resolution(h3_list[0])

    poi_dir = source_dir if source_dir is not None else CITY_CONFIGS[city]["poi_dir"]

    layers = [
        "points",
        "areas",
    ]

    if include_lines:
        layers.append("lines")

    counts = {
        h3_id: {
            column: 0
            for column in POI_COLUMNS
        }
        for h3_id in h3_list
    }

    seen_osm = set()
    source_count = 0
    mapped_count = 0
    outside_count = 0
    no_geometry_count = 0

    for layer in layers:
        path = poi_dir / f"{layer}.jsonl"

        if not path.exists():
            continue

        for record in iter_jsonl(path):
            source_count += 1

            osm_key = (
                record.get("osm_type"),
                record.get("osm_id"),
            )

            if (
                osm_key[0] is not None
                and osm_key[1] is not None
            ):
                if osm_key in seen_osm:
                    continue
                seen_osm.add(osm_key)

            lonlat = representative_lonlat(
                record
            )

            if lonlat is None:
                no_geometry_count += 1
                continue

            lon, lat = lonlat
            h3_id = coord_to_h3(
                lat,
                lon,
                resolution,
            )

            if h3_id not in valid_h3:
                outside_count += 1
                continue

            tags = record.get("tags")

            if not isinstance(tags, dict):
                tags = {}

            category_id = classify_osm_tags(
                tags
            )
            category_name = category_id_to_name(
                category_id
            )
            column = CATEGORY_TO_COLUMN.get(
                category_name,
                "poi_other",
            )

            counts[h3_id][column] += 1
            mapped_count += 1


    rows = []

    for h3_id in h3_list:
        row = {
            "h3": h3_id,
            **counts[h3_id],
        }
        row["poi_total"] = sum(
            row[column]
            for column in POI_COLUMNS
        )
        rows.append(row)

    result = pd.DataFrame(rows)


    result = result[
        [
            "h3",
            *POI_COLUMNS,
            "poi_total",
        ]
    ]

    city_output_dir = (
        output_root / city
    )
    city_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        city_output_dir
        / f"{city}_final_with_poi.csv"
    )
    result.to_csv(
        output_path,
        index=False,
    )

    h3_with_poi = int(
        (result["poi_total"] > 0).sum()
    )
    print(f"[{city}] Mapped {mapped_count} of {source_count} records to {h3_with_poi} H3 cells")
    print(f"[{city}] Outside the region: {outside_count} | Missing geometry: {no_geometry_count}")
    print(output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Classify OSM points and areas and aggregate them by H3"
        )
    )
    parser.add_argument(
        "cities",
        nargs="*",
        default=["sg"],
    )
    parser.add_argument("--regions", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument(
        "--output-root",
        default=str(OUTPUT_ROOT),
    )
    parser.add_argument(
        "--classifier-dir",
        default=str(
            Path(__file__).resolve().parent
        ),
        help=(
            "map2cat.py and map2cat_cn.py directory"
        ),
    )
    parser.add_argument(
        "--include-lines",
        action="store_true",
        help=(
            "Include lines.jsonl"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(
        args.output_root
    )
    classifier_dir = Path(
        args.classifier_dir
    )

    for city in args.cities:
        process_city(city, output_root, classifier_dir, args.include_lines, args.regions, args.source_dir)


if __name__ == "__main__":
    main()
