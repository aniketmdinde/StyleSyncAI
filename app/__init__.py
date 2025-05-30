from flask import Flask
import os

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = "123"

    from .ai import ai_bp
    app.register_blueprint(ai_bp, url_prefix='/')

    return app