from flask import Flask
import os
import google.generativeai as genai
from pymongo import MongoClient
import urllib.parse
from sentence_transformers import SentenceTransformer, util
import torch

genai.configure(api_key="AIzaSyD6NSWFspgQgBOHt2F08VZStEvc37xMBZ4")
gemini_model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
system_prompt = """
You are a fashion enrichment and analysis assistant for a fashion recommendation engine.

Your input will be either:
- A user query describing a desired outfit or styling preference
- A single fashion product
- An image showing an entire outfit or clothing item

Your job is to extract and infer detailed metadata that can be used to understand the fashion context, and assist recommendation engines.

Analyze the text and/or image and return a single structured JSON object with:

- `title`: A concise, descriptive title (e.g., "Monochrome Winter Streetwear Outfit", "Beige Cotton Trench Coat").
- `description`: A complete natural-language paragraph that summarizes the outfit or product in fashion-oriented language.
- `items`: A list of one or more detected/described fashion items. Each item should include:
  - `type`: (e.g., jacket, boots, crop top, saree, bag)
  - `color`: Dominant visible color(s)
  - `material`: (e.g., denim, cotton, leather, chiffon)
  - `fit`: (e.g., slim fit, oversized, relaxed)
  - `gender`: Target gender (Men, Women, Unisex)
  - `aesthetic`: Overall aesthetic (e.g., grunge, streetwear, techwear, boho, minimalist, formal).
  - `season`: Season(s) it’s suitable for (e.g., summer, winter, all-season).
  - `occasion`: Best suited occasion(s) (e.g., casual outing, wedding, formal dinner, beach vacation)
  - `tags`: A list of 15–30 rich, relevant, search-optimized fashion tags that combine aesthetics, materials, categories, colors, seasons, style trends.
  - `semantic_category`: "",
  - `categories`: []
"""

username = "aniketmdinde100"
raw_password = "Aniket*99"
encoded_password = urllib.parse.quote_plus(raw_password)

uri = f"mongodb+srv://{username}:{encoded_password}@stylesyncaicluster.jr7xedc.mongodb.net/stylesync?retryWrites=true&w=majority&appName=StyleSyncAICluster"
client = MongoClient(uri)
db = client["stylesync"]
collection = db["products"]

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = "123"

    from .ai import ai_bp
    app.register_blueprint(ai_bp, url_prefix='')

    return app