from __future__ import annotations

import argparse
import os
import ast
import json
import re
from pathlib import Path
from typing import Any, Iterable

import h3
import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except ImportError:
    gpd = None

try:
    from shapely import wkt
except ImportError:
    wkt = None


BASE_HLY = Path(os.environ.get("DISCO_RAW_ROOT", "raw_data")).resolve()
H3_DIR = BASE_HLY / "1.15/H3_Regions_new"
FLOW_ROOT = BASE_HLY / "1.15/flow"
OUTPUT_DIR = BASE_HLY / "new_pipeline/data_process"

H3_RES = 7
N_HOURS = 24

CITY_CONFIGS = {
    "beijing": {
        "h3_candidates": ["beijing_h3_res7.csv"],
        "flow_root": FLOW_ROOT / "China/data/beijing",
        "flow_type": "deepmobility",
    },
    "shanghai": {
        "h3_candidates": ["shanghai_h3_res7.csv"],
        "flow_root": FLOW_ROOT / "China/data/shanghai",
        "flow_type": "deepmobility",
    },
    "sg": {
        "h3_candidates": [
            "sg_h3_res7.csv",
            "singapore_h3_res7.csv",
        ],
        "flow_root": FLOW_ROOT / "sg/flow_data",
        "flow_type": "singapore_lta",
    },
    "sydney": {
        "h3_candidates": ["sydney_h3_res7.csv"],
        "flow_root": FLOW_ROOT / "sydney/flow_data",
        "flow_type": "sydney_opal",
    },
}


def normalize_name(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def normalize_code(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()

    if not text or text.lower() in {
        "nan",
        "none",
        "null",
    }:
        return None

    if re.fullmatch(r"-?\d+\.0", text):
        text = text[:-2]

    return text.upper()


def coord_to_h3(
    lat: float,
    lon: float,
    res: int = H3_RES,
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


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def pick_column(
    df: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    index = {
        normalize_name(column): column
        for column in df.columns
    }

    for candidate in candidates:
        normalized = normalize_name(candidate)
        if normalized in index:
            return index[normalized]

    if required:
        raise KeyError(
            f"Missing columns {candidates}. "
            f"found={df.columns.tolist()}"
        )

    return None


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(
            path,
            dtype=str,
            low_memory=False,
        )

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str)

    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Table format: {path}")


def parse_hour(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    try:
        number = float(text)
        hour = int(number)
        if 0 <= hour <= 23:
            return hour
    except ValueError:
        pass

    match = re.search(
        r"(?<!\d)([01]?\d|2[0-3])(?::\d{2})?",
        text,
    )

    if match:
        return int(match.group(1))

    return None


def load_h3_list(city: str, h3_path: Path | None = None) -> list[str]:
    if h3_path is None:
        candidates = CITY_CONFIGS[city]["h3_candidates"]
        h3_path = first_existing(H3_DIR / name for name in candidates) or H3_DIR / candidates[0]

    df = pd.read_csv(h3_path, dtype=str)

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
            f"[{city}] expected h3, h3_index or hex_id. "
            f"found={df.columns.tolist()}"
        )

    values = (
        df[h3_col]
        .dropna()
        .astype(str)
        .str.strip()
    )

    valid = [
        value
        for value in values
        if h3.is_valid_cell(value)
    ]

    h3_list = sorted(set(valid))


    return h3_list


def group_od(
    df: pd.DataFrame,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "origin_h3",
                "dest_h3",
                "time_bin",
                "trip_count",
            ]
        )

    df = df[
        [
            "origin_h3",
            "dest_h3",
            "time_bin",
            "trip_count",
        ]
    ].copy()

    df["origin_h3"] = df["origin_h3"].astype(str)
    df["dest_h3"] = df["dest_h3"].astype(str)
    df["time_bin"] = pd.to_numeric(
        df["time_bin"],
        errors="coerce",
    )
    df["trip_count"] = pd.to_numeric(
        df["trip_count"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "origin_h3",
            "dest_h3",
            "time_bin",
            "trip_count",
        ]
    )

    df["time_bin"] = df["time_bin"].astype(int)

    df = df[
        df["time_bin"].between(
            0,
            N_HOURS - 1,
        )
    ]

    return (
        df.groupby(
            [
                "origin_h3",
                "dest_h3",
                "time_bin",
            ],
            as_index=False,
        )["trip_count"]
        .sum()
        .sort_values(
            [
                "origin_h3",
                "dest_h3",
                "time_bin",
            ]
        )
        .reset_index(drop=True)
    )


CODE_CANDIDATES = [

    "pt_code",
    "busstopcode",
    "bus_stop_code",
    "stop_code",
    "station_code",
    "stn_no",
    "stn_code",


    "tzcode",
    "tz_code",
    "tz16",
    "tz16_code",
    "travel_zone",
    "travel_zone_code",
    "zone_id",
    "zone_code",
]

LAT_CANDIDATES = [
    "latitude",
    "lat",
    "centroid_latitude",
    "centroid_lat",
    "y",
]

LON_CANDIDATES = [
    "longitude",
    "lon",
    "lng",
    "centroid_longitude",
    "centroid_lon",
    "x",
]


def dataframe_to_spatial_lookup(
    df: pd.DataFrame,
    source: str,
) -> pd.DataFrame | None:
    code_col = pick_column(
        df,
        CODE_CANDIDATES,
        required=False,
    )

    if code_col is None:
        return None

    lat_col = pick_column(
        df,
        LAT_CANDIDATES,
        required=False,
    )
    lon_col = pick_column(
        df,
        LON_CANDIDATES,
        required=False,
    )

    if lat_col and lon_col:
        lookup = df[
            [code_col, lat_col, lon_col]
        ].copy()
        lookup.columns = [
            "code",
            "lat",
            "lon",
        ]

    elif "geometry" in df.columns:
        geometries = df["geometry"]

        if (
            len(geometries)
            and isinstance(
                geometries.iloc[0],
                str,
            )
            and wkt is not None
        ):
            def parse_wkt(value):
                try:
                    return wkt.loads(value)
                except Exception:
                    return None

            geometries = geometries.apply(parse_wkt)

        points = geometries.apply(
            lambda geom: (
                geom.representative_point()
                if geom is not None
                and not getattr(
                    geom,
                    "is_empty",
                    True,
                )
                else None
            )
        )

        lookup = pd.DataFrame({
            "code": df[code_col],
            "lat": points.apply(
                lambda point: (
                    point.y
                    if point is not None
                    else np.nan
                )
            ),
            "lon": points.apply(
                lambda point: (
                    point.x
                    if point is not None
                    else np.nan
                )
            ),
        })
    else:
        return None

    lookup["code"] = lookup["code"].apply(
        normalize_code
    )
    lookup["lat"] = pd.to_numeric(
        lookup["lat"],
        errors="coerce",
    )
    lookup["lon"] = pd.to_numeric(
        lookup["lon"],
        errors="coerce",
    )
    lookup["source"] = source

    lookup = (
        lookup.dropna(
            subset=[
                "code",
                "lat",
                "lon",
            ]
        )
        .drop_duplicates(
            subset=["code"],
            keep="first",
        )
    )

    if lookup.empty:
        return None

    return lookup


def load_spatial_lookup(
    root: Path,
) -> pd.DataFrame:
    lookups: list[pd.DataFrame] = []

    table_paths: list[Path] = []
    for pattern in (
        "*.csv",
        "*.xlsx",
        "*.xls",
        "*.parquet",
    ):
        table_paths.extend(root.rglob(pattern))

    for path in sorted(set(table_paths)):
        try:
            df = read_table(path)
            lookup = dataframe_to_spatial_lookup(
                df,
                str(path),
            )
            if lookup is not None:
                lookups.append(lookup)
        except Exception:
            continue

    geo_paths: list[Path] = []
    for pattern in (
        "*.shp",
        "*.geojson",
        "*.gpkg",
    ):
        geo_paths.extend(root.rglob(pattern))

    if gpd is not None:
        for path in sorted(set(geo_paths)):
            try:
                gdf = gpd.read_file(path)

                if gdf.crs is not None:
                    gdf = gdf.to_crs(
                        "EPSG:4326"
                    )

                lookup = dataframe_to_spatial_lookup(
                    gdf,
                    str(path),
                )
                if lookup is not None:
                    lookups.append(lookup)
            except Exception:
                continue

    if not lookups:
        return pd.DataFrame(
            columns=[
                "code",
                "lat",
                "lon",
                "source",
            ]
        )

    result = (
        pd.concat(
            lookups,
            ignore_index=True,
        )
        .drop_duplicates(
            subset=["code"],
            keep="first",
        )
    )

    return result


def map_codes_to_h3(
    od_df: pd.DataFrame,
    lookup: pd.DataFrame,
    valid_h3: set[str],
    city: str,
) -> pd.DataFrame:
    lookup = lookup.copy()
    lookup["h3"] = lookup.apply(
        lambda row: coord_to_h3(
            row["lat"],
            row["lon"],
        ),
        axis=1,
    )

    code_to_h3 = dict(
        zip(
            lookup["code"],
            lookup["h3"],
        )
    )

    od_df = od_df.copy()
    od_df["origin_code"] = od_df[
        "origin_code"
    ].apply(normalize_code)
    od_df["dest_code"] = od_df[
        "dest_code"
    ].apply(normalize_code)

    od_df["origin_h3"] = od_df[
        "origin_code"
    ].map(code_to_h3)
    od_df["dest_h3"] = od_df[
        "dest_code"
    ].map(code_to_h3)

    od_df = od_df.dropna(
        subset=[
            "origin_h3",
            "dest_h3",
        ]
    )

    od_df = od_df[
        od_df["origin_h3"].isin(valid_h3)
        & od_df["dest_h3"].isin(valid_h3)
    ]

    return od_df


def parse_lonlat(value: Any) -> tuple[float, float]:
    if isinstance(value, str):
        value = ast.literal_eval(value)

    if not isinstance(
        value,
        (list, tuple),
    ) or len(value) < 2:
        raise ValueError(
            f"Coordinates: {value!r}"
        )

    return float(value[0]), float(value[1])


def find_deepmobility_dir(
    root: Path,
) -> Path:
    if (root / "OD_6.npy").exists():
        return root

    matches = sorted(
        root.rglob("OD_6.npy")
    )

    if not matches:
        raise FileNotFoundError(
            f"{root}/OD_6.npy"
        )

    return matches[0].parent


def load_deepmobility_od(
    city: str,
    valid_h3: set[str],
) -> pd.DataFrame:
    root = find_deepmobility_dir(
        CITY_CONFIGS[city]["flow_root"]
    )

    od_path = root / "OD_6.npy"
    attr_path = root / "region_attribute.csv"

    od = np.load(od_path)
    attr = pd.read_csv(
        attr_path,
        low_memory=False,
    )

    region_col = pick_column(
        attr,
        [
            "RegionIndex",
            "region_index",
        ],
    )
    lonlat_col = pick_column(
        attr,
        ["lonlat"],
    )

    attr[region_col] = pd.to_numeric(
        attr[region_col],
        errors="raise",
    ).astype(int)

    region_count = od.shape[-1]

    expected = set(range(region_count))
    actual = set(attr[region_col].tolist())

    if expected != actual:
        raise ValueError(
            f"[{city}] RegionIndex and OD sizes differ. "
            f"missing={sorted(expected - actual)[:10]}，"
            f"extra={sorted(actual - expected)[:10]}"
        )

    attr = (
        attr.set_index(region_col)
        .loc[range(region_count)]
        .reset_index()
    )

    lonlat = attr[lonlat_col].apply(
        parse_lonlat
    )
    attr["lon"] = lonlat.apply(
        lambda value: value[0]
    )
    attr["lat"] = lonlat.apply(
        lambda value: value[1]
    )
    attr["h3"] = attr.apply(
        lambda row: coord_to_h3(
            row["lat"],
            row["lon"],
        ),
        axis=1,
    )

    if od.ndim == 4 and od.shape[1] == 4:

        slot_od = od.sum(axis=0)
    elif od.ndim == 3 and od.shape[0] == 4:
        slot_od = od
    else:
        raise ValueError(
            f"[{city}] OD_6.npy shape: "
            f"{od.shape}"
        )

    region_h3 = attr["h3"].to_numpy()

    frames: list[pd.DataFrame] = []

    for slot in range(4):
        matrix = np.asarray(
            slot_od[slot],
            dtype=np.float64,
        )

        rows, cols = np.nonzero(matrix)

        if len(rows) == 0:
            continue

        slot_df = pd.DataFrame({
            "origin_h3": region_h3[rows],
            "dest_h3": region_h3[cols],
            "trip_count": matrix[rows, cols],
        })

        slot_df = (
            slot_df.groupby(
                [
                    "origin_h3",
                    "dest_h3",
                ],
                as_index=False,
            )["trip_count"]
            .sum()
        )


        for hour in range(
            slot * 6,
            slot * 6 + 6,
        ):
            part = slot_df.copy()
            part["time_bin"] = hour
            part["trip_count"] = (
                part["trip_count"] / 6.0
            )
            frames.append(part)

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    result = result[
        result["origin_h3"].isin(valid_h3)
        & result["dest_h3"].isin(valid_h3)
    ]


    return group_od(result)


SG_ORIGIN_COLUMNS = [
    "ORIGIN_PT_CODE",
    "origin_pt_code",
    "origin_code",
]

SG_DEST_COLUMNS = [
    "DESTINATION_PT_CODE",
    "destination_pt_code",
    "destination_code",
]

TRIP_COLUMNS = [
    "TOTAL_TRIPS",
    "total_trips",
    "trips",
    "trip_count",
    "journeys",
    "journey_count",
    "passenger_count",
    "count",
]

HOUR_COLUMNS = [
    "TIME_PER_HOUR",
    "time_per_hour",
    "hour",
    "time",
    "time_bin",
]


def load_singapore_od(
    valid_h3: set[str],
) -> pd.DataFrame:
    root = CITY_CONFIGS["sg"]["flow_root"]

    month_dirs = [
        root / "202604",
        root / "202605",
        root / "202606",
    ]

    frames: list[pd.DataFrame] = []

    for month_dir in month_dirs:
        if not month_dir.exists():
            continue

        paths: list[Path] = []
        for pattern in (
            "*.csv",
            "*.xlsx",
            "*.xls",
            "*.parquet",
        ):
            paths.extend(
                month_dir.rglob(pattern)
            )

        for path in sorted(set(paths)):
            try:
                df = read_table(path)

                origin_col = pick_column(
                    df,
                    SG_ORIGIN_COLUMNS,
                    required=False,
                )
                dest_col = pick_column(
                    df,
                    SG_DEST_COLUMNS,
                    required=False,
                )
                trip_col = pick_column(
                    df,
                    TRIP_COLUMNS,
                    required=False,
                )
                hour_col = pick_column(
                    df,
                    HOUR_COLUMNS,
                    required=False,
                )

                if not all(
                    [
                        origin_col,
                        dest_col,
                        trip_col,
                        hour_col,
                    ]
                ):
                    continue

                part = df[
                    [
                        origin_col,
                        dest_col,
                        trip_col,
                        hour_col,
                    ]
                ].copy()
                part.columns = [
                    "origin_code",
                    "dest_code",
                    "trip_count",
                    "time_raw",
                ]

                part["time_bin"] = part[
                    "time_raw"
                ].apply(parse_hour)
                part["trip_count"] = pd.to_numeric(
                    part["trip_count"],
                    errors="coerce",
                )

                part = part.dropna(
                    subset=[
                        "origin_code",
                        "dest_code",
                        "trip_count",
                        "time_bin",
                    ]
                )

                frames.append(
                    part[
                        [
                            "origin_code",
                            "dest_code",
                            "time_bin",
                            "trip_count",
                        ]
                    ]
                )

            except Exception:
                continue

    if not frames:
        raise FileNotFoundError(
            "[sg] No table with ORIGIN_PT_CODE, "
            "DESTINATION_PT_CODE、TIME_PER_HOUR、"
            "TOTAL_TRIPS"
        )

    spatial_root = root / "spatial"
    lookup = load_spatial_lookup(
        spatial_root
    )

    if lookup.empty:
        raise FileNotFoundError(
            f"[sg] {spatial_root} lacks "
            "station IDs and coordinates or geometry"
        )


    od_df = pd.concat(
        frames,
        ignore_index=True,
    )

    od_df = map_codes_to_h3(
        od_df,
        lookup,
        valid_h3,
        "sg",
    )

    return group_od(od_df)


SYDNEY_ORIGIN_COLUMNS = [
    "TZcode_on",
    "tzcode_on",
    "origin",
    "origin_zone",
    "origin_travel_zone",
    "origin_tz",
    "origin_zone_id",
]

SYDNEY_DEST_COLUMNS = [
    "TZcode_off",
    "tzcode_off",
    "destination",
    "destination_zone",
    "destination_travel_zone",
    "destination_tz",
    "destination_zone_id",
]


def load_sydney_od(
    valid_h3: set[str],
) -> pd.DataFrame:
    root = CITY_CONFIGS["sydney"]["flow_root"]

    files = sorted(
        root.glob(
            "origin_destination_*.csv"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"[sydney] "
            f"{root}/origin_destination_*.csv"
        )

    frames: list[pd.DataFrame] = []

    for path in files:
        df = pd.read_csv(
            path,
            dtype=str,
            low_memory=False,
        )

        origin_col = pick_column(
            df,
            SYDNEY_ORIGIN_COLUMNS,
        )
        dest_col = pick_column(
            df,
            SYDNEY_DEST_COLUMNS,
        )
        trip_col = pick_column(
            df,
            TRIP_COLUMNS,
        )

        part = df[
            [
                origin_col,
                dest_col,
                trip_col,
            ]
        ].copy()
        part.columns = [
            "origin_code",
            "dest_code",
            "trip_count",
        ]

        part["trip_count"] = pd.to_numeric(
            part["trip_count"],
            errors="coerce",
        )
        part = part.dropna(
            subset=[
                "origin_code",
                "dest_code",
                "trip_count",
            ]
        )

        frames.append(part)



    lookup = load_spatial_lookup(root)

    if lookup.empty:
        raise FileNotFoundError(
            f"[sydney] Travel Zone coordinates: {root}"
        )


    static_od = pd.concat(
        frames,
        ignore_index=True,
    )

    static_od = map_codes_to_h3(
        static_od,
        lookup,
        valid_h3,
        "sydney",
    )

    static_od = (
        static_od.groupby(
            [
                "origin_h3",
                "dest_h3",
            ],
            as_index=False,
        )["trip_count"]
        .sum()
    )


    hourly_frames = []

    for hour in range(N_HOURS):
        part = static_od.copy()
        part["time_bin"] = hour
        part["trip_count"] = (
            part["trip_count"] / N_HOURS
        )
        hourly_frames.append(part)


    return group_od(
        pd.concat(
            hourly_frames,
            ignore_index=True,
        )
    )


def save_flow_outputs(
    city: str,
    h3_list: list[str],
    od_df: pd.DataFrame,
    output_dir: Path,
    time_semantics: str | None = None,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    valid_h3 = set(h3_list)

    od_df = od_df[
        od_df["origin_h3"].isin(valid_h3)
        & od_df["dest_h3"].isin(valid_h3)
    ].copy()

    od_df = group_od(od_df)

    od_path = (
        output_dir
        / f"{city}_flow_map_h3.csv"
    )
    od_df.to_csv(
        od_path,
        index=False,
    )

    h3_array = np.asarray(
        h3_list,
        dtype=str,
    )
    h3_list_path = (
        output_dir
        / f"{city}_flow_h3_list.npy"
    )
    np.save(
        h3_list_path,
        h3_array,
    )

    h3_to_idx = {
        h3_id: index
        for index, h3_id in enumerate(h3_list)
    }

    n_regions = len(h3_list)
    half = n_regions * N_HOURS
    feature_dim = 2 * half

    profile_path = (
        output_dir
        / f"{city}_flow_h3_profile.npy"
    )


    profile = np.lib.format.open_memmap(
        profile_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            n_regions,
            feature_dim,
        ),
    )
    profile[:] = 0.0

    for row in od_df.itertuples(
        index=False
    ):
        origin_index = h3_to_idx[
            str(row.origin_h3)
        ]
        dest_index = h3_to_idx[
            str(row.dest_h3)
        ]
        time_bin = int(row.time_bin)
        value = float(row.trip_count)


        profile[
            dest_index,
            origin_index * N_HOURS + time_bin,
        ] += value


        profile[
            origin_index,
            half + dest_index * N_HOURS + time_bin,
        ] += value

    profile.flush()

    total_flow = float(
        od_df["trip_count"].sum()
    )
    active_h3 = len(
        set(od_df["origin_h3"])
        | set(od_df["dest_h3"])
    )
    unique_od = od_df.groupby(
        [
            "origin_h3",
            "dest_h3",
        ]
    ).ngroups

    metadata = {
        "city": city,
        "h3_resolution": h3.get_resolution(h3_list[0]),
        "n_hours": N_HOURS,
        "n_regions": n_regions,
        "feature_dim": feature_dim,
        "active_h3": active_h3,
        "unique_od_pairs": int(unique_od),
        "total_flow": total_flow,
        "profile_shape": [
            n_regions,
            feature_dim,
        ],
    }

    if time_semantics is not None:
        metadata["time_semantics"] = time_semantics
    elif city in {
        "beijing",
        "shanghai",
    }:
        metadata["time_semantics"] = (
            "DeepMobility 4×6-hour slots uniformly "
            "expanded to 24 hours; each slot divided by 6"
        )
    elif city == "sg":
        metadata["time_semantics"] = (
            "LTA TIME_PER_HOUR aggregated to hour-of-day"
        )
    else:
        metadata["time_semantics"] = (
            "Static Opal OD uniformly expanded to 24 hours; "
            "each OD count divided by 24"
        )

    metadata_path = (
        output_dir
        / f"{city}_flow_metadata.json"
    )
    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "-" * 70)
    print(
        f"  Profile shape：({n_regions:,}, "
        f"{feature_dim:,}) = 2×N×24"
    )
    print(f"  OD CSV：        {od_path}")
    print(f"  Profile：       {profile_path}")
    print(f"  Metadata：      {metadata_path}")
    print("-" * 70)


def load_coordinate_od(path: Path, resolution: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for prefix, output in (("origin", "origin_h3"), ("destination", "dest_h3")):
        frame[output] = [coord_to_h3(lat, lon, resolution) for lat, lon in
                         zip(frame[f"{prefix}_lat"], frame[f"{prefix}_lon"], strict=True)]
    return frame.rename(columns={"hour": "time_bin"})


def process_city(
    city: str,
    output_dir: Path,
    regions: Path | None = None,
    source_csv: Path | None = None,
) -> None:
    print("\n" + "=" * 80)
    print(f"CITY: {city.upper()}")
    print("=" * 80)

    h3_list = load_h3_list(city, regions)
    if source_csv is not None:
        od_df = load_coordinate_od(source_csv, h3.get_resolution(h3_list[0]))
        save_flow_outputs(city, h3_list, od_df, output_dir, "Counts grouped by hour of day")
        return
    valid_h3 = set(h3_list)
    flow_type = CITY_CONFIGS[
        city
    ]["flow_type"]

    if flow_type == "deepmobility":
        od_df = load_deepmobility_od(
            city,
            valid_h3,
        )
    elif flow_type == "singapore_lta":
        od_df = load_singapore_od(
            valid_h3,
        )
    elif flow_type == "sydney_opal":
        od_df = load_sydney_od(
            valid_h3,
        )
    else:
        raise ValueError(
            f"Flow type: {flow_type}"
        )

    if od_df.empty:
        raise RuntimeError(
            f"[{city}] has no mapped flows"
        )

    save_flow_outputs(
        city,
        h3_list,
        od_df,
        output_dir,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Map flow to H3 "
            "and export an hourly OD profile"
        )
    )
    parser.add_argument(
        "cities",
        nargs="*",
        default=["sg"],
        help=(
            "Examples: beijing shanghai sg sydney. "
            "Default: sg"
        ),
    )
    parser.add_argument("--regions", type=Path)
    parser.add_argument("--source-csv", type=Path)
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for city in args.cities:
        process_city(city, output_dir, args.regions, args.source_csv)


if __name__ == "__main__":
    main()
