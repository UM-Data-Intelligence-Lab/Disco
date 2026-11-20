from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


from map2cat import (
    CATEGORY_NAMES,
    CAT_NAME_TO_ID,
    NUM_COARSE_CATEGORIES,
    OTHER_ID,
    classify_single as classify_single_en,
)


CHINESE_COARSE_CATEGORY_RULES = json.loads(Path(__file__).with_name("category_keywords.json").read_text())


CATEGORY_PRIORITY: List[str] = [
    "Healthcare",
    "Education",
    "Government",
    "Religious",
    "Finance",
    "Transport",
    "Lodging",
    "Food & Drink",
    "Shopping",
    "Recreation",
    "Services",
    "Office & Organization",
]

_PRIORITY_INDEX = {
    category_name: index
    for index, category_name in enumerate(CATEGORY_PRIORITY)
}


def normalize_text(text: Any) -> str:


    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower().strip()


    text = re.sub(
        r"[，,。；;：:、/\\|_\-—–（）()\[\]【】{}<>《》“”\"'·]+",
        " ",
        text,
    )

    text = re.sub(r"\s+", " ", text)
    return text.strip()


_COMPILED_ZH_RULES: List[
    Tuple[re.Pattern, int, str, str]
] = []

for category_name in CATEGORY_PRIORITY:
    keywords = CHINESE_COARSE_CATEGORY_RULES.get(category_name, [])


    keywords = sorted(
        set(keywords),
        key=lambda keyword: len(normalize_text(keyword)),
        reverse=True,
    )

    for keyword in keywords:
        normalized_keyword = normalize_text(keyword)

        if not normalized_keyword:
            continue

        pattern = re.compile(
            re.escape(normalized_keyword),
            re.IGNORECASE,
        )

        _COMPILED_ZH_RULES.append(
            (
                pattern,
                CAT_NAME_TO_ID[category_name],
                category_name,
                normalized_keyword,
            )
        )


_COMPILED_ZH_RULES.sort(
    key=lambda item: (
        -len(item[3]),
        _PRIORITY_INDEX[item[2]],
    )
)


def classify_single_zh(category_text: Any) -> int:


    text = normalize_text(category_text)

    if not text:
        return OTHER_ID

    for pattern, category_id, _, _ in _COMPILED_ZH_RULES:
        if pattern.search(text):
            return category_id

    return OTHER_ID


def classify_single_zh_with_reason(
    category_text: Any,
) -> Tuple[int, Optional[str], Optional[str]]:


    text = normalize_text(category_text)

    if not text:
        return OTHER_ID, "Other", None

    for pattern, category_id, category_name, keyword in _COMPILED_ZH_RULES:
        if pattern.search(text):
            return category_id, category_name, keyword

    return OTHER_ID, "Other", None


def classify_single(category_text: Any) -> int:


    zh_category_id = classify_single_zh(category_text)

    if zh_category_id != OTHER_ID:
        return zh_category_id

    return classify_single_en(normalize_text(category_text))


def classify_batch(category_texts: Sequence[Any]) -> List[int]:

    return [
        classify_single(category_text)
        for category_text in category_texts
    ]


OSM_TAG_RULES: Dict[str, Dict[str, set[str]]] = {
    "Healthcare": {
        "amenity": {
            "hospital",
            "clinic",
            "doctors",
            "dentist",
            "pharmacy",
            "veterinary",
            "nursing_home",
            "social_facility",
        },
        "healthcare": {
            "hospital",
            "clinic",
            "doctor",
            "dentist",
            "pharmacy",
            "physiotherapist",
            "psychotherapist",
            "laboratory",
            "rehabilitation",
            "midwife",
            "optometrist",
            "podiatrist",
        },
        "shop": {
            "pharmacy",
            "medical_supply",
            "hearing_aids",
            "optician",
        },
    },

    "Education": {
        "amenity": {
            "school",
            "university",
            "college",
            "kindergarten",
            "childcare",
            "library",
            "training",
            "language_school",
            "music_school",
            "driving_school",
        },
    },

    "Government": {
        "amenity": {
            "police",
            "fire_station",
            "courthouse",
            "townhall",
            "embassy",
        },
        "office": {
            "government",
            "diplomatic",
        },
        "diplomatic": {
            "embassy",
            "consulate",
            "liaison",
        },
    },

    "Religious": {
        "amenity": {
            "place_of_worship",
            "monastery",
        },
        "building": {
            "church",
            "chapel",
            "mosque",
            "temple",
            "synagogue",
            "shrine",
        },
    },

    "Finance": {
        "amenity": {
            "bank",
            "atm",
            "bureau_de_change",
        },
        "office": {
            "financial",
            "insurance",
            "accountant",
            "tax_advisor",
        },
        "shop": {
            "pawnbroker",
        },
    },

    "Transport": {
        "amenity": {
            "parking",
            "parking_entrance",
            "parking_space",
            "fuel",
            "charging_station",
            "bus_station",
            "taxi",
            "ferry_terminal",
            "car_rental",
            "bicycle_rental",
            "car_wash",
            "vehicle_inspection",
        },
        "public_transport": {
            "station",
            "platform",
            "stop_position",
        },
        "railway": {
            "station",
            "halt",
            "tram_stop",
            "subway_entrance",
        },
        "aeroway": {
            "aerodrome",
            "terminal",
            "gate",
            "helipad",
        },
        "highway": {
            "bus_stop",
            "services",
            "rest_area",
            "toll_booth",
        },
    },

    "Lodging": {
        "tourism": {
            "hotel",
            "motel",
            "guest_house",
            "hostel",
            "apartment",
            "chalet",
            "resort",
            "alpine_hut",
            "camp_site",
            "caravan_site",
        },
        "building": {
            "hotel",
            "apartments",
            "residential",
            "dormitory",
        },
    },

    "Food & Drink": {
        "amenity": {
            "restaurant",
            "cafe",
            "fast_food",
            "food_court",
            "pub",
            "bar",
            "biergarten",
            "ice_cream",
        },
        "shop": {
            "bakery",
            "deli",
            "confectionery",
            "coffee",
            "tea",
            "beverages",
            "seafood",
            "butcher",
        },
    },

    "Recreation": {
        "amenity": {
            "cinema",
            "theatre",
            "arts_centre",
            "community_centre",
            "social_centre",
            "nightclub",
            "casino",
        },
        "leisure": {
            "park",
            "fitness_centre",
            "sports_centre",
            "stadium",
            "pitch",
            "swimming_pool",
            "golf_course",
            "garden",
            "playground",
            "marina",
            "dance",
        },
        "tourism": {
            "museum",
            "gallery",
            "attraction",
            "zoo",
            "theme_park",
            "aquarium",
            "viewpoint",
            "artwork",
            "picnic_site",
        },
    },

    "Services": {
        "amenity": {
            "post_office",
            "parcel_locker",
            "funeral_hall",
        },
        "shop": {
            "hairdresser",
            "beauty",
            "laundry",
            "dry_cleaning",
            "car_repair",
            "car",
            "travel_agency",
            "funeral_directors",
            "pet_grooming",
            "copyshop",
        },
        "office": {
            "lawyer",
            "real_estate_agent",
            "employment_agency",
            "travel_agent",
            "telecommunication",
            "property_management",
        },
    },

    "Shopping": {
        "amenity": {
            "marketplace",
        },
        "shop": {
            "supermarket",
            "convenience",
            "department_store",
            "mall",
            "clothes",
            "shoes",
            "jewelry",
            "furniture",
            "hardware",
            "electronics",
            "mobile_phone",
            "computer",
            "books",
            "stationery",
            "toys",
            "gift",
            "florist",
            "cosmetics",
            "sports",
            "pet",
            "variety_store",
            "wholesale",
        },
        "landuse": {
            "retail",
        },
    },

    "Office & Organization": {
        "office": {
            "company",
            "association",
            "ngo",
            "research",
            "it",
            "software",
            "consulting",
            "engineer",
            "architect",
        },
        "building": {
            "office",
            "commercial",
            "industrial",
            "warehouse",
        },
        "landuse": {
            "commercial",
            "industrial",
        },
        "industrial": {
            "factory",
            "warehouse",
            "manufacturing",
        },
    },
}


OSM_CATEGORY_PRIORITY: List[str] = [
    "Healthcare",
    "Education",
    "Government",
    "Religious",
    "Finance",
    "Transport",
    "Lodging",
    "Food & Drink",
    "Recreation",
    "Services",
    "Shopping",
    "Office & Organization",
]


def normalize_osm_value(value: Any) -> str:

    if value is None:
        return ""

    return str(value).strip().lower()


def classify_osm_tags(tags: Mapping[str, Any]) -> int:


    normalized_tags = {
        str(key).strip().lower(): normalize_osm_value(value)
        for key, value in tags.items()
        if value is not None
    }


    for category_name in OSM_CATEGORY_PRIORITY:
        rules = OSM_TAG_RULES[category_name]

        for osm_key, allowed_values in rules.items():
            osm_value = normalized_tags.get(osm_key, "")

            if osm_value in allowed_values:
                return CAT_NAME_TO_ID[category_name]


    healthcare_value = normalized_tags.get("healthcare", "")
    if healthcare_value not in {"", "no", "none"}:
        return CAT_NAME_TO_ID["Healthcare"]


    if normalized_tags.get("public_transport", "") not in {
        "",
        "no",
        "none",
    }:
        return CAT_NAME_TO_ID["Transport"]


    if normalized_tags.get("railway", "") not in {
        "",
        "no",
        "abandoned",
        "disused",
    }:
        return CAT_NAME_TO_ID["Transport"]

    if normalized_tags.get("aeroway", "") not in {
        "",
        "no",
    }:
        return CAT_NAME_TO_ID["Transport"]


    if normalized_tags.get("craft", "") not in {
        "",
        "no",
    }:
        return CAT_NAME_TO_ID["Services"]


    if normalized_tags.get("leisure", "") not in {
        "",
        "no",
    }:
        return CAT_NAME_TO_ID["Recreation"]


    if normalized_tags.get("tourism", "") not in {
        "",
        "no",
    }:
        return CAT_NAME_TO_ID["Recreation"]


    if normalized_tags.get("shop", "") not in {
        "",
        "no",
        "vacant",
    }:
        return CAT_NAME_TO_ID["Shopping"]


    if normalized_tags.get("office", "") not in {
        "",
        "no",
    }:
        return CAT_NAME_TO_ID["Office & Organization"]


    text_keys = [
        "name",
        "name:zh",
        "name:zh-hans",
        "official_name",
        "short_name",
        "brand",
        "brand:zh",
        "operator",
        "operator:zh",
        "description",
        "amenity",
        "shop",
        "office",
        "tourism",
        "leisure",
        "healthcare",
        "public_transport",
        "railway",
        "aeroway",
        "building",
        "landuse",
        "craft",
    ]

    combined_text = " ".join(
        normalized_tags[key]
        for key in text_keys
        if normalized_tags.get(key)
    )

    return classify_single(combined_text)


def classify_osm_batch(
    tags_list: Sequence[Mapping[str, Any]],
) -> List[int]:

    return [
        classify_osm_tags(tags)
        for tags in tags_list
    ]


def category_id_to_name(category_id: int) -> str:

    if category_id < 0 or category_id >= len(CATEGORY_NAMES):
        return "Other"

    return CATEGORY_NAMES[category_id]
