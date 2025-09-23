from flask import Blueprint, request, jsonify
from services.video_service import download_video_from_s3
from services.similarity import perform_similarity_check
from services.notify import notify_springboot

video_bp = Blueprint("video", __name__)

@video_bp.route("/download", methods=["POST"])
def download_video():
    data = request.json
    video_url = data.get("file_url")
    if not video_url:
        return jsonify({"error": "No video_url provided"}), 400

    try:
        file_path = download_video_from_s3(video_url)
        similarity_result = perform_similarity_check(file_path, video_url)
        notify_backend(similarity_result)
        return jsonify({"message": "Download & similarity check complete",
                        "result": similarity_result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
