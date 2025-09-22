from flask import Blueprint, jsonify
from extensions.faiss_index import load_faiss_index, save_faiss_index, FAISSIndex

faiss_bp = Blueprint("faiss", __name__)

@faiss_bp.route("/faiss_info", methods=["GET"])
def faiss_info():
    try:
        faiss_index = load_faiss_index()
        total_vectors = faiss_index.index.ntotal
        return jsonify({"message": "FAISS index info", "total_vectors": total_vectors})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@faiss_bp.route("/reset_faiss_index", methods=["POST"])
def reset_faiss_index():
    try:
        faiss_index = FAISSIndex(dim=2048)
        save_faiss_index(faiss_index)
        return jsonify({"message": "FAISS index reset complete"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
