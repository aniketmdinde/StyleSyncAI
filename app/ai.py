from flask import Blueprint, jsonify, request
from . import gemini_model, embedding_model, system_prompt, collection
from sentence_transformers import util
from PIL import Image

ai_bp = Blueprint("audio", __name__)

@ai_bp.route('/')
def home():
    return jsonify({"response": "Hello"}), 200

@ai_bp.route("/ai", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        image_file = request.files.get("image")

        if not query and not image_file:
            return jsonify({"error": "Please provide a description or image."}), 400

        try:
            inputs = [system_prompt]
            if query:
                inputs.append("User Query:\n" + query)

            if image_file:
                image = Image.open(image_file.stream)
                inputs.append(image)

            response = gemini_model.generate_content(inputs, stream=False)
            return jsonify({"result": response.text}), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"message": "Send a POST request with a 'query' or an 'image'."}), 200


@ai_bp.route("/add", methods=["POST"])
def add_data():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data received"}), 400
        
        result = collection.insert_one(data)
        return jsonify({
            "message": "Data inserted successfully!",
            "inserted_id": str(result.inserted_id)
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@ai_bp.route("/example", methods=["POST"])
def example():
    return jsonify({"message": "true"}), 200


# Recommendation
def get_product_tags(slug):
    product = collection.find_one({"slug": slug})
    if product and 'tags' in product:
        return product['tags']
    else:
        raise ValueError("Product with given slug not found or no tags.")

def get_all_products_except(slug):
    products = list(collection.find({"slug": {"$ne": slug}}, {"slug": 1, "tags": 1}))
    return products

def recommend_similar_products(slug, top_k=3):
    base_tags = get_product_tags(slug)
    base_text = " ".join(base_tags)
    base_embedding = embedding_model.encode(base_text, convert_to_tensor=True)

    candidates = get_all_products_except(slug)
    similarities = []

    for prod in candidates:
        prod_tags = prod.get('tags', [])
        if not prod_tags:
            continue
        prod_text = " ".join(prod_tags)
        prod_embedding = embedding_model.encode(prod_text, convert_to_tensor=True)
        sim_score = util.cos_sim(base_embedding, prod_embedding).item()
        similarities.append((prod['slug'], sim_score))

    top_matches = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    return [slug for slug, score in top_matches]

@ai_bp.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    if not data or 'slug' not in data:
        return jsonify({"error": "Missing 'slug' in JSON body"}), 400

    slug = data['slug']

    try:
        top_slugs = recommend_similar_products(slug)
        return jsonify({"recommended_slugs": top_slugs})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500
    

# Outfit Bundling
category_map = { 
    'top': ['top', 'shirt'],
    'bottom': ['bottom', 'pant', 'skirt', 'short'],
    'accessories': ['accessories'],
    'shoes': ['shoes', 'sandals', 'slippers']
}

def detect_category(category_name):
    category_name = category_name.lower()
    for canonical, keywords in category_map.items():
        if any(keyword in category_name for keyword in keywords):
            return canonical
    return None

def get_product_by_slug(slug):
    doc = collection.find_one({"slug": slug})
    if doc:
        tags = doc.get("tags", [])
        raw_category = doc.get("category", "")
        canonical_category = detect_category(raw_category)
        return {"tags": tags, "category": canonical_category}
    return None

def find_best_bundles(target_tags, exclude_category):
    best_matches = {}
    target_embedding = embedding_model.encode(" ".join(target_tags), convert_to_tensor=True)

    for doc in collection.find():
        doc_category_raw = doc.get("category", "")
        doc_category = detect_category(doc_category_raw)

        if not doc_category or doc_category == exclude_category:
            continue

        tags = doc.get("tags", [])
        if not tags:
            continue

        doc_embedding = embedding_model.encode(" ".join(tags), convert_to_tensor=True)
        score = util.cos_sim(target_embedding, doc_embedding).item()

        if doc_category not in best_matches or score > best_matches[doc_category][1]:
            best_matches[doc_category] = (doc, score)

    return best_matches

@ai_bp.route("/bundling", methods=["POST"])
def recommend_bundles():
    data = request.get_json()
    slug = data.get("slug")

    if not slug:
        return jsonify({"error": "Missing 'slug' in request body"}), 400

    product = get_product_by_slug(slug)
    if not product:
        return jsonify({"error": "Product not found"}), 404

    recommendations = find_best_bundles(product["tags"], product["category"])

    result = {
        "target_product": {
            "slug": slug,
            "category": product["category"],
            "tags": product["tags"]
        },
        "recommendations": []
    }

    for cat, (rec, score) in recommendations.items():
        result["recommendations"].append({
            "category": cat,
            "slug": rec.get("slug"),
            "tags": rec.get("tags"),
            "score": round(score, 4)
        })

    return jsonify(result), 200


# Based on user query
@ai_bp.route("/query/recommend", methods=["POST"])
def recommend_from_query():
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Missing 'query' in JSON body"}), 400

    try:
        query_embedding = embedding_model.encode(query, convert_to_tensor=True)
        similarities = []

        for doc in collection.find({}, {"slug": 1, "tags": 1}):
            tags = doc.get("tags", [])
            if not tags:
                continue
            tags_text = " ".join(tags)
            doc_embedding = embedding_model.encode(tags_text, convert_to_tensor=True)
            sim_score = util.cos_sim(query_embedding, doc_embedding).item()
            similarities.append((doc['slug'], sim_score))

        top_k = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]
        return jsonify({"recommended_slugs": [slug for slug, _ in top_k]}), 200

    except Exception as e:
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

@ai_bp.route("/query/bundling", methods=["POST"])
def bundle_from_query():
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Missing 'query' in JSON body"}), 400

    try:
        query_embedding = embedding_model.encode(query, convert_to_tensor=True)
        best_matches = {}

        for doc in collection.find():
            raw_category = doc.get("category", "")
            canonical_category = detect_category(raw_category)
            if not canonical_category:
                continue

            tags = doc.get("tags", [])
            if not tags:
                continue

            doc_embedding = embedding_model.encode(" ".join(tags), convert_to_tensor=True)
            score = util.cos_sim(query_embedding, doc_embedding).item()

            if canonical_category not in best_matches or score > best_matches[canonical_category][1]:
                best_matches[canonical_category] = (doc, score)

        response = {
            "query": query,
            "recommendations": [
                {
                    "category": cat,
                    "slug": rec.get("slug"),
                    "tags": rec.get("tags"),
                    "score": round(score, 4)
                }
                for cat, (rec, score) in best_matches.items()
            ]
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500