import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, BertModel, BertTokenizer
from PIL import Image
import numpy as np
from app import collection
from app.dummy_data import load_dummy_data

class OutfitTransformer(nn.Module):
    def __init__(self, hidden_size=768, num_heads=8, num_layers=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Image encoder
        self.image_encoder = AutoModel.from_pretrained("google/vit-base-patch16-224")
        self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        
        # Text encoder
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        # Transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=0.1
            ),
            num_layers=num_layers
        )
        
        # Category embeddings
        self.category_embeddings = nn.Embedding(5, hidden_size)  # 5 main categories
        
        # Compatibility prediction head
        self.compatibility_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def encode_image(self, image):
        # Process image
        inputs = self.image_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.image_encoder(**inputs).last_hidden_state[:, 0, :]
        return image_features
    
    def encode_text(self, text):
        # Process text
        encoded = self.text_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.text_encoder(**encoded)
            text_features = outputs.last_hidden_state[:, 0, :]
        return text_features
    
    def forward(self, item_embeddings, category_ids):
        # Add category embeddings
        category_emb = self.category_embeddings(category_ids)
        combined_emb = item_embeddings + category_emb
        
        # Transformer processing
        transformer_out = self.transformer(combined_emb)
        
        return transformer_out

class OutfitRecommender:
    def __init__(self):
        self.model = OutfitTransformer()
        self.category_map = {
            'top': 0,
            'bottom': 1,
            'shoes': 2,
            'accessories': 3,
            'other': 4
        }
    
    def get_category_id(self, category):
        return self.category_map.get(category.lower(), 4)
    
    def calculate_compatibility(self, item1, item2):
        try:
            # Get embeddings for both items
            if isinstance(item1, Image.Image):
                item1_emb = self.model.encode_image(item1)
            else:
                text1 = item1.get('text', '') if isinstance(item1, dict) else str(item1)
                item1_emb = self.model.encode_text(text1)
                
            if isinstance(item2, Image.Image):
                item2_emb = self.model.encode_image(item2)
            else:
                text2 = " ".join(item2.get('tags', [])) if isinstance(item2, dict) else str(item2)
                item2_emb = self.model.encode_text(text2)
            
            # Get category IDs
            cat1_id = self.get_category_id(item1.get('category', 'other') if isinstance(item1, dict) else 'other')
            cat2_id = self.get_category_id(item2.get('category', 'other') if isinstance(item2, dict) else 'other')
            
            # Process through transformer
            combined_emb = torch.stack([item1_emb, item2_emb])
            cat_ids = torch.tensor([cat1_id, cat2_id])
            transformer_out = self.model(combined_emb, cat_ids)
            
            # Calculate compatibility score
            pair = torch.cat([transformer_out[0], transformer_out[1]], dim=-1).unsqueeze(0)  # [1, 1536]
            compatibility = self.model.compatibility_head(pair)  # [1, 1]
            
            # Ensure scalar output
            if compatibility.numel() == 1:
                return compatibility.item()
            else:
                return compatibility.view(-1)[0].item()
        except Exception as e:
            print(f"Error in calculate_compatibility: {str(e)}")
            return 0.0
    
    def get_outfit_recommendations(self, query, is_image=False, top_k=5):
        try:
            # Get all products
            all_products = list(collection.find())
            
            # Calculate compatibility scores
            recommendations = []
            for product in all_products:
                if is_image:
                    compatibility = self.calculate_compatibility(query, product)
                else:
                    compatibility = self.calculate_compatibility(
                        {"text": query, "category": "query"},
                        product
                    )
                
                recommendations.append({
                    'slug': product['slug'],
                    'compatibility_score': compatibility,
                    'category': product.get('category', ''),
                    'tags': product.get('tags', [])
                })
            
            # Sort by compatibility score
            recommendations.sort(key=lambda x: x['compatibility_score'], reverse=True)
            return recommendations[:top_k]
        except Exception as e:
            print(f"Error in get_outfit_recommendations: {str(e)}")
            return []

# Initialize the recommender
outfit_recommender = OutfitRecommender()

# Load dummy data
load_dummy_data(collection) 