import numpy as np
import faiss
from extensions.faiss_index import load_faiss_index, save_faiss_index
from extensions.db import get_db_connection
from services.video_service import extract_frames, extract_features, delete_local_file

def perform_similarity_check(video_path, video_url):
    faiss_index = load_faiss_index()
    total_vectors = faiss_index.index.ntotal

    frames = extract_frames(video_path)
    feature_vectors = extract_features(frames)

    if total_vectors == 0:
        faiss_index.add_vectors(feature_vectors)
        save_faiss_index(faiss_index)
        insert_vector_metadata(video_url, 0, len(feature_vectors))
        delete_local_file(video_path)
        return {"message": "First video save complete", "passed": True}

    query_vectors = np.array(feature_vectors).astype('float32')
    faiss.normalize_L2(query_vectors)
    similarity_scores = []
    for q in query_vectors:
        q = q.reshape(1, -1)
        faiss.normalize_L2(q)
        distances, _ = faiss_index.index.search(q, total_vectors)
        similarity_scores.extend(distances[0].tolist())

    max_similarity = max(similarity_scores)
    avg_similarity = sum(similarity_scores) / len(similarity_scores)

    result = {"max_similarity": max_similarity,
              "avg_similarity": avg_similarity,
              "video_url": video_url}

    if max_similarity >= 0.9 and avg_similarity >= 0.8:
        result["message"] = "Similarity check failed"
        result["passed"] = False
    else:
        result["message"] = "Similarity check passed"
        result["passed"] = True

        start_idx = faiss_index.index.ntotal
        faiss_index.add_vectors(feature_vectors)
        save_faiss_index(faiss_index)
        insert_vector_metadata(video_url, start_idx, len(feature_vectors))

    delete_local_file(video_path)
    return result

def insert_vector_metadata(video_url, start_idx, count):
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            for i in range(count):
                sql = "INSERT INTO video_vectors (video_url, faiss_index) VALUES (%s, %s)"
                cursor.execute(sql, (video_url, start_idx + i))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f" MySQL insert error: {e}")
