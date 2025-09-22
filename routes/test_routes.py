from flask import Blueprint, request, jsonify
from extensions.db import get_db_connection

test_bp = Blueprint("test", __name__)

@test_bp.route("/test_db_insert", methods=["POST"])
def test_db_insert():
    data = request.json
    video_id = data.get("video_id")
    faiss_index = data.get("faiss_index")
    if not video_id or faiss_index is None:
        return jsonify({"error": "video_id and faiss_index are required"}), 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        sql = "INSERT INTO video_vectors (video_id, faiss_index) VALUES (%s, %s)"
        cursor.execute(sql, (video_id, faiss_index))
        conn.commit()

        cursor.execute("SELECT * FROM video_vectors ORDER BY id DESC LIMIT 1")
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return jsonify({"message": "insert complete", "last_inserted": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@test_bp.route("/health", methods=["GET"])
def health():
    return "ok", 200
