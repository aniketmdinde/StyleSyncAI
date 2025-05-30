from flask import Blueprint, jsonify

ai_bp = Blueprint("audio", __name__)

@ai_bp.route('/')
def home():
    return jsonify({"response": "Hello"}), 200
    # return "Hello World"