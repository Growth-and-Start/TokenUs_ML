import requests
from config import Config

def notify_backend(similarity_result):
    payload = {
        "max_similarity": similarity_result.get("max_similarity"),
        "avg_similarity": similarity_result.get("avg_similarity"),
        "message": similarity_result.get("message"),
        "passed": similarity_result.get("passed"),
        "similar_video_url": similarity_result.get("similar_video_url"),
        "video_url": similarity_result.get("video_url"),
    }
    headers = {"Content-Type": "application/json"}
    try:
        url = Config.BACKEND_URL + Config.API_PATH
        response = requests.post(url, json=payload, headers=headers)
        print(f"Backend response: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Backend notify error: {e}")
