"""Domain constants shared across pipeline scripts."""

# Price level → label mapping  (single source of truth)
# Level 0 = free/very cheap, 4 = luxury
PRICE_LABELS = {
    0: "Free",
    1: "Budget",
    2: "Moderate",
    3: "Expensive",
    4: "Luxury",
}

# Rating bucket boundaries
RATING_BUCKETS = [
    (0.0,  3.5,  "Poor"),
    (3.5,  4.0,  "Average"),
    (4.0,  4.5,  "Good"),
    (4.5,  5.01, "Excellent"),
]

# Synthetic user generation
N_USERS         = 200
RATINGS_PER_USER = (20, 50)

# Top-restaurants threshold
MIN_REVIEWS_THRESHOLD = 10
TOP_RESTAURANTS_LIMIT = 50
