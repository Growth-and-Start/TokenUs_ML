from flask import Flask, jsonify
from config import Config
import os

# route blueprint import
from routes.video_routes import video_bp
from routes.faiss_routes import faiss_bp
from routes.test_routes import test_bp

def create_app():

    os.makedirs(os.path.dirname(Config.FAISS_INDEX_PATH), exist_ok=True)

    app = Flask(__name__)
    app.config.from_object(Config)

    # Blueprint register
    app.register_blueprint(video_bp)
    app.register_blueprint(faiss_bp)
    app.register_blueprint(test_bp)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"}), 200

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=Config.FLASK_PORT, debug=Config.DEBUG)
