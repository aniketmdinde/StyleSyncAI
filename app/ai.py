from flask import Blueprint, jsonify, request
from app import collection
from PIL import Image
import io
from .outfit_transformer import outfit_recommender
import logging
from itertools import product, combinations

# Major categories and their semantic subcategories (removed bag, activewear, intimates)
MAJOR_CATEGORY_MAP = {
    "top": ["blouse", "shirt", "t-shirt", "sweater", "cardigan", "jacket", "vest", "tank top", "polo", "hoodie", "tunic", "outerwear", "sweatshirt"],
    "bottom": ["jeans", "pants", "skirt", "shorts", "leggings", "trousers", "suit", "jumpsuit", "romper", "capri", "cropped pants"],
    "dress": ["dress", "gown", "wedding dress", "day dress", "cocktail dress"],
    "shoes": ["boots", "pumps", "sandals", "flats", "flip flops", "sneakers", "slippers", "loafers", "moccasins", "oxfords", "clogs", "athletic shoes"],
    "outerwear": ["coat", "jacket", "vest", "blazer"],
    "accessories": ["belt", "glove", "hat", "tie", "sunglasses", "hair accessory", "umbrella", "scarf", "watch", "jewelry", "necklace", "earring", "ring", "brooch", "hosiery", "sock"],
}

# Color harmony (simplified): analogous, monochromatic, or same family
COLOR_GROUPS = [
    {"white", "grey", "black", "navy", "brown"},
    {"red", "pink", "magenta", "purple"},
    {"blue", "cyan", "turquoise", "azure"},
    {"green", "olive", "emerald", "mint", "sage"},
    {"yellow", "gold", "beige", "cream"},
    {"orange", "peach", "apricot", "tan"},
]

def color_group(color):
    color = color.lower()
    for group in COLOR_GROUPS:
        if color in group:
            return group
    return {color}

# Flatten all semantic subcategories for quick lookup
SEMANTIC_TO_MAJOR = {}
for major, semantics in MAJOR_CATEGORY_MAP.items():
    for sem in semantics:
        SEMANTIC_TO_MAJOR[sem] = major

def get_major_and_semantic_category(product):
    tags = [t.lower() for t in product.get("tags", [])]
    cat = product.get("category", "").lower()
    for tag in tags + [cat]:
        for sem, major in SEMANTIC_TO_MAJOR.items():
            if sem in tag or tag in sem:
                return major, sem
    if cat in MAJOR_CATEGORY_MAP:
        return cat, cat
    return "other", cat

ai_bp = Blueprint("ai", __name__)
MAJOR_CATEGORIES = list(MAJOR_CATEGORY_MAP.keys())

# Helper: check if two items are compatible by gender and season
def compatible_items(item1, item2):
    return (
        (item1.get("gender", "unisex") == item2.get("gender", "unisex") or "unisex" in [item1.get("gender"), item2.get("gender")]) and
        (item1.get("season", "all") == item2.get("season", "all") or "all" in [item1.get("season"), item2.get("season")])
    )

# Helper: check color harmony for a bundle
def harmonious_colors(bundle):
    colors = [item.get("color", "") for item in bundle]
    groups = [color_group(c) for c in colors if c]
    # All colors should be in at most 2 groups (analogous/monochrome)
    unique_groups = set(tuple(sorted(g)) for g in groups)
    return len(unique_groups) <= 2

# Helper: ensure bundles are diverse (at least 3 different items between bundles)
def bundles_are_diverse(bundle1, bundle2):
    slugs1 = set(item["slug"] for item in bundle1)
    slugs2 = set(item["slug"] for item in bundle2)
    return len(slugs1.symmetric_difference(slugs2)) >= 3

# Main function to generate top N diverse, harmonious bundles
def generate_bundles(recommendations, n=3):
    # Group by major category and pick top 3 for each
    category_items = {cat: [] for cat in MAJOR_CATEGORIES}
    for rec in recommendations:
        major, semantic = get_major_and_semantic_category(rec)
        if major in MAJOR_CATEGORIES:
            rec['major_category'] = major
            rec['semantic_category'] = semantic
            if len(category_items[major]) < 3:
                category_items[major].append(rec)
    # Only use one item per category in each bundle
    bundle_candidates = [category_items[cat] for cat in MAJOR_CATEGORIES if category_items[cat]]
    if len(bundle_candidates) < 2:
        return []
    all_bundles = [list(bundle) for bundle in product(*bundle_candidates)]
    # Filter for unique categories in each bundle (should always be true, but double check)
    valid_bundles = []
    for bundle in all_bundles:
        majors = [item['major_category'] for item in bundle]
        if len(set(majors)) != len(majors):
            continue  # skip bundles with duplicate categories
        # All items must be compatible by gender/season
        if not all(compatible_items(bundle[i], bundle[j]) for i, j in combinations(range(len(bundle)), 2)):
            continue
        # Color harmony
        if not harmonious_colors(bundle):
            continue
        valid_bundles.append(bundle)
    # Sort by average compatibility score
    valid_bundles.sort(key=lambda b: sum(item['compatibility_score'] for item in b) / len(b), reverse=True)
    # Ensure diversity: pick top N bundles with at least 3 different items between them
    diverse_bundles = []
    for bundle in valid_bundles:
        if all(bundles_are_diverse(bundle, prev) for prev in diverse_bundles):
            diverse_bundles.append(bundle)
        if len(diverse_bundles) == n:
            break
    # If not enough, fill with best available
    while len(diverse_bundles) < n and valid_bundles:
        candidate = valid_bundles.pop(0)
        if candidate not in diverse_bundles:
            diverse_bundles.append(candidate)
    return [
        {
            "items": b,
            "score": round(sum(item['compatibility_score'] for item in b) / len(b), 4)
        } for b in diverse_bundles
    ]

@ai_bp.route('/')
def home():
    return jsonify({"response": "Hello"}), 200

@ai_bp.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' in JSON body"}), 400
    query = data['query']
    try:
        recommendations = outfit_recommender.get_outfit_recommendations(query, top_k=50)
        bundles = generate_bundles(recommendations, n=3)
        if not bundles:
            return jsonify({"error": "No valid outfits found for the query."}), 404
        return jsonify({
            "query": query,
            "outfits": bundles
        })
    except Exception as e:
        logging.exception("Error in /recommend endpoint")
        return jsonify({"error": str(e)}), 500
    
@ai_bp.route("/recommend/image", methods=["POST"])
def recommend_from_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    try:
        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read()))
        recommendations = outfit_recommender.get_outfit_recommendations(image, is_image=True, top_k=50)
        bundles = generate_bundles(recommendations, n=3)
        if not bundles:
            return jsonify({"error": "No valid outfits found for the image."}), 404
        return jsonify({
            "outfits": bundles
        })
    except Exception as e:
        logging.exception("Error in /recommend/image endpoint")
        return jsonify({"error": str(e)}), 500

@ai_bp.route("/bundling", methods=["POST"])
def recommend_bundles():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' in JSON body"}), 400
    query = data['query']
    try:
        recommendations = outfit_recommender.get_outfit_recommendations(query, top_k=50)
        bundles = generate_bundles(recommendations, n=3)
        if not bundles:
            return jsonify({"error": "No valid bundles could be created."}), 404
        return jsonify({
            "query": query,
            "bundles": bundles
        })
    except Exception as e:
        logging.exception("Error in /bundling endpoint")
        return jsonify({"error": str(e)}), 500

@ai_bp.route("/bundling/image", methods=["POST"])
def bundle_from_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    try:
        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read()))
        recommendations = outfit_recommender.get_outfit_recommendations(image, is_image=True, top_k=50)
        bundles = generate_bundles(recommendations, n=3)
        if not bundles:
            return jsonify({"error": "No valid bundles could be created."}), 404
        return jsonify({
            "bundles": bundles
        })
    except Exception as e:
        logging.exception("Error in /bundling/image endpoint")
        return jsonify({"error": str(e)}), 500