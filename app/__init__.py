from flask import Flask
from flask_cors import CORS
from .service import service_bp

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Register blueprints
    app.register_blueprint(service_bp)

    return app
