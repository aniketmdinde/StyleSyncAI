from flask import Blueprint, jsonify, request
from . import model, system_prompt
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

            response = model.generate_content(inputs, stream=False)
            return jsonify({"result": response.text}), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"message": "Send a POST request with a 'query' or an 'image'."}), 200
