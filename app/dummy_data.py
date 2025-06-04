import json
from datetime import datetime
import random

# Major categories and their item types
major_categories = {
    "top": ["t-shirt", "blouse", "hoodie", "shirt", "cardigan", "tank top", "sweater"],
    "bottom": ["jeans", "skirt", "shorts", "pants", "leggings", "trousers"],
    "dress": ["dress", "gown"],
    "shoes": ["sneakers", "boots", "loafers", "flats", "sandals"],
    "outerwear": ["coat", "blazer", "jacket", "vest"],
    "accessories": ["necklace", "watch", "scarf", "belt", "hat", "ring", "bracelet"],
}

colors = ["white", "black", "blue", "red", "green", "yellow", "grey", "brown", "navy", "pink"]
genders = ["male", "female", "unisex"]
seasons = ["summer", "spring", "fall", "winter", "all"]

expanded_products = []
unique_id = 1
for major, item_types in major_categories.items():
    for item_type in item_types:
        for color in colors:
            for gender in genders:
                for season in seasons:
                    prod = {
                        "item_type": item_type,
                        "base_name": item_type.replace("-", " ").title(),
                        "category": major,
                        "tags": list(set([item_type, major, color, gender, season])),
                        "price": round(random.uniform(20, 120), 2),
                        "image_url": f"https://example.com/{color}-{item_type}.jpg",
                        "description": f"{color.capitalize()} {item_type.replace('-', ' ')} for {season} ({gender}).",
                        "color": color,
                        "gender": gender,
                        "season": season,
                        "slug": f"{color}-{item_type}-{gender}-{season}-{unique_id:03d}",
                        "name": f"{color.capitalize()} {item_type.replace('-', ' ').title()} ({gender.capitalize()}, {season.capitalize()})",
                        "created_at": datetime.now().isoformat()
                    }
                    expanded_products.append(prod)
                    unique_id += 1

# Shuffle and limit to 120 for performance, but ensure at least 10 per major category
random.shuffle(expanded_products)
final_products = []
cat_counts = {cat: 0 for cat in major_categories}
for prod in expanded_products:
    cat = prod["category"]
    if cat_counts[cat] < 10 or len(final_products) < 120:
        final_products.append(prod)
        cat_counts[cat] += 1
    if len(final_products) >= 120 and all(v >= 10 for v in cat_counts.values()):
        break

def load_dummy_data(collection):
    """Load realistic dummy data into MongoDB collection"""
    collection.delete_many({})
    collection.insert_many(final_products)
    print(f"Loaded {len(final_products)} realistic products into the database") 