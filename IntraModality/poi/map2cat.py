import re
from typing import Dict, List, Tuple

COARSE_CATEGORY_RULES: Dict[str, List[str]] = {
    "Food & Drink": [
        "restaurant", "food", "cafe", "coffee", "pizza", "burger", "sushi",
        "bakery", "bar", "pub", "brewery", "diner", "grill", "steakhouse",
        "seafood", "taco", "noodle", "ramen", "thai", "chinese", "indian",
        "mexican", "italian", "japanese", "korean", "vietnamese", "mediterranean",
        "sandwich", "ice cream", "dessert", "donut", "bagel", "juice",
        "tea house", "brunch", "buffet", "fast food", "chicken", "bbq",
        "barbecue", "wing", "salad", "soup", "breakfast", "lunch",
        "catering", "meal", "dining", "eatery", "bistro", "brasserie",
        "cantina", "chophouse", "creperie", "deli", "delicatessen",
        "gastropub", "patisserie", "pizzeria", "trattoria", "winery",
        "distillery", "taproom", "lounge", "nightclub", "cocktail",
        "steak house", "caterer", "banquet hall",
    ],
    "Shopping": [
        "store", "shop", "market", "mall", "supermarket", "grocery",
        "boutique", "outlet", "retail", "wholesale", "plaza",
        "department store", "convenience store", "discount store",
        "thrift", "gift shop", "bookstore", "florist", "jewel",
        "clothing", "apparel", "fashion", "shoe", "furniture",
        "hardware", "electronics", "appliance", "pet store",
        "sporting goods", "toy", "antique", "art supply",
        "cosmetics", "beauty supply", "optical", "eyewear",
    ],
    "Finance": [
        "bank", "credit union", "atm", "financial", "insurance",
        "accounting", "tax", "investment", "mortgage", "loan",
    ],
    "Education": [
        "school", "university", "college", "academy", "preschool",
        "kindergarten", "daycare", "learning center", "tutor",
        "training", "education", "library", "institute",
    ],
    "Healthcare": [
        "hospital", "clinic", "doctor", "dentist", "pharmacy",
        "medical", "health", "urgent care", "veterinar", "optometrist",
        "chiropract", "therapist", "psycholog", "psychiatr",
        "dermatolog", "pediatr", "orthoped", "cardio", "surgeon",
        "nursing", "rehab", "physical therapy", "mental health",
    ],
    "Lodging": [
        "hotel", "motel", "inn", "hostel", "resort", "lodge",
        "bed and breakfast", "airbnb", "guest house", "suites",
        "condominium", "apartment complex", "apartment building",
        "housing development",
    ],
    "Recreation": [
        "park", "museum", "theater", "theatre", "cinema", "movie",
        "gym", "fitness", "sport", "stadium", "arena", "pool",
        "bowling", "golf", "yoga", "spa", "salon", "nail",
        "massage", "recreation", "amusement", "zoo", "aquarium",
        "garden", "playground", "trail", "beach", "gallery",
        "entertainment", "casino", "arcade", "club",
        "tourist attraction", "historical landmark", "landmark",
        "sculpture", "pier", "picnic ground", "soccer field",
        "tennis court", "mountain peak", "lake",
        "live music venue", "event venue", "cultural center",
        "senior citizen center", "community center",
        "sightseeing", "tour operator", "ferry terminal",
    ],
    "Transport": [
        "gas station", "parking", "bus", "metro", "subway", "train",
        "airport", "taxi", "car rental", "car wash", "auto repair",
        "mechanic", "tire", "towing", "transit", "station",
        "charging station", "fuel",
        "airline", "railroad", "trucking", "chauffeur",
        "car leasing", "oil change service", "bicycle rental",
    ],
    "Services": [
        "laundry", "dry clean", "barber", "hair", "beauty",
        "repair", "plumb", "electric", "hvac", "locksmith",
        "moving", "storage", "print", "copy", "post office",
        "shipping", "courier", "notary", "legal", "lawyer",
        "attorney", "real estate", "property", "contractor",
        "cleaning", "landscap", "pest control", "tailor",
        "alteration", "car dealer", "dealership",
        "laundromat", "pet groomer", "groomer",
        "handyman", "employment agency", "travel agency",
        "recording studio", "donations center", "homeless shelter",
    ],
    "Religious": [
        "church", "mosque", "temple", "synagogue", "chapel",
        "cathedral", "worship", "religious", "ministry",
        "kingdom hall", "jehovah",
    ],
    "Government": [
        "city hall", "courthouse", "police", "fire department",
        "government", "embassy", "consulate", "dmv", "social security",
        "public",
    ],
    "Office & Organization": [
        "corporate office", "office", "software company", "company",
        "manufacturer", "non-profit", "nonprofit", "organization",
        "association", "warehouse",
    ],
}


CATEGORY_NAMES = [
    "Food & Drink",
    "Shopping",
    "Finance",
    "Education",
    "Healthcare",
    "Lodging",
    "Recreation",
    "Transport",
    "Services",
    "Religious",
    "Government",
    "Office & Organization",
    "Other",
]

CAT_NAME_TO_ID = {
    name: index
    for index, name in enumerate(CATEGORY_NAMES)
}

CAT_ID_TO_NAME = {
    index: name
    for name, index in CAT_NAME_TO_ID.items()
}

NUM_COARSE_CATEGORIES = len(CATEGORY_NAMES)
OTHER_ID = CAT_NAME_TO_ID["Other"]
CAT_NAME_TO_ID = {name: i for i, name in enumerate(CATEGORY_NAMES)}
NUM_COARSE_CATEGORIES = len(CATEGORY_NAMES)


_COMPILED_RULES: List[Tuple[re.Pattern, int]] = []
for cat_name in CATEGORY_NAMES[:-1]:

    keywords = COARSE_CATEGORY_RULES[cat_name]

    keywords_sorted = sorted(keywords, key=len, reverse=True)
    for kw in keywords_sorted:

        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        _COMPILED_RULES.append((pattern, CAT_NAME_TO_ID[cat_name]))

OTHER_ID = CAT_NAME_TO_ID["Other"]


def classify_single(category_text: str) -> int:

    for pattern, cat_id in _COMPILED_RULES:
        if pattern.search(category_text):
            return cat_id
    return OTHER_ID


def classify_batch(category_texts: List[str]) -> List[int]:

    return [classify_single(t) for t in category_texts]
