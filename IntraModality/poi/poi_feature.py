import os

import numpy as np
import pandas as pd


POI_COLUMNS = [
    "poi_food_drink",
    "poi_shopping",
    "poi_finance",
    "poi_education",
    "poi_healthcare",
    "poi_lodging",
    "poi_recreation",
    "poi_transport",
    "poi_services",
    "poi_religious",
    "poi_government",
    "poi_office_organization",
    "poi_other",
]


class POIFeatureBuilder:

    def __init__(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"{csv_path}"
            )

        df = pd.read_csv(csv_path)


        for column in POI_COLUMNS:
            if column not in df.columns:
                df[column] = 0

        self.poi_matrix = (
            df[POI_COLUMNS]
            .fillna(0)
            .astype(np.float32)
            .values
        )

        if "h3" in df.columns:
            h3_column = "h3"
        elif "h3_index" in df.columns:
            h3_column = "h3_index"
        else:
            raise ValueError(
                f"Missing h3 or h3_index: {csv_path}"
            )

        self.h3_list = (
            df[h3_column]
            .astype(str)
            .tolist()
        )

        if self.poi_matrix.shape[1] != 13:
            raise ValueError(
                "Expected 13 POI features. "
                f"Received {self.poi_matrix.shape[1]}"
            )

    @classmethod
    def from_city(
        cls,
        city: str,
        base_path: str,
    ):
        csv_path = os.path.join(
            base_path,
            city,
            f"{city}_final_with_poi.csv",
        )

        return cls(csv_path)
