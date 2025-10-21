import numpy as np
import faiss
import time
from collections import Counter
from extensions.faiss_index import load_faiss_index, save_faiss_index
from extensions.db import get_db_connection
from services.video_service import extract_frames, extract_features, delete_local_file


def perform_similarity_check(video_path, video_url):

    print(f"Performing similarity check for video: {video_url}")

    start_time = time.time()

    faiss_index = load_faiss_index()
    total_vectors = faiss_index.index.ntotal

    frames = extract_frames(video_path, interval=1)

    feature_vectors = extract_features(frames)

    if total_vectors == 0:
        faiss_index.add_vectors(feature_vectors)
        save_faiss_index(faiss_index)
        insert_vector_metadata(video_url, 0, len(feature_vectors))
        delete_local_file(video_path)
        elapsed_time = time.time() - start_time
        return {
            "message": "First video saved successfully",
            "passed": True,
            "elapsed_time": round(elapsed_time, 2)
        }

    query_vectors = np.array(feature_vectors).astype('float32')
    faiss.normalize_L2(query_vectors)

    similarity_scores = []
    all_indices = []

    for i, q in enumerate(query_vectors):
        q = q.reshape(1, -1)
        faiss.normalize_L2(q)
        distances, indices = faiss_index.index.search(q, total_vectors)
        similarity_scores.extend(distances[0].tolist())
        all_indices.extend(indices[0].tolist())

    window_size = 10
    stride = 1
    segment_similarities = []
    segment_indices = []

    if len(similarity_scores) >= window_size:
        for start in range(0, len(similarity_scores) - window_size + 1, stride):
            window_scores = similarity_scores[start:start + window_size]
            window_indices = all_indices[start:start + window_size]
            avg_seg_sim = sum(window_scores) / len(window_scores)
            segment_similarities.append(avg_seg_sim)
            segment_indices.append(window_indices)
        max_idx = int(np.argmax(segment_similarities))
        max_segment_similarity = segment_similarities[max_idx]
        most_similar_indices = segment_indices[max_idx]
    else:
        max_segment_similarity = sum(similarity_scores) / len(similarity_scores)
        most_similar_indices = all_indices

    most_common_video_url = get_most_common_video_url(most_similar_indices)

    result = {
        "video_url": video_url,
        "similar_video_url": most_common_video_url,
        "max_segment_similarity": max_segment_similarity
    }

    if max_segment_similarity >= 0.85:
        result["message"] = "Similarity check failed"
        result["passed"] = False
    else:
        result["message"] = "Similarity check passed"
        result["passed"] = True
        result["similar_video_url"] = None

        start_idx = faiss_index.index.ntotal
        faiss_index.add_vectors(feature_vectors)
        save_faiss_index(faiss_index)
        insert_vector_metadata(video_url, start_idx, len(feature_vectors))

    delete_local_file(video_path)
    result["elapsed_time"] = round(time.time() - start_time, 2)

    return result


def insert_vector_metadata(video_url, start_idx, count):

    print(f"Inserting vector metadata for video: {video_url}, starting at index: {start_idx}, count: {count}")

    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            for i in range(count):
                sql = "INSERT INTO video_vectors (video_url, faiss_index) VALUES (%s, %s)"
                cursor.execute(sql, (video_url, start_idx + i))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB ERROR] MySQL insert error: {e}")


def get_most_common_video_url(faiss_indices):
    try:
        if not faiss_indices or len(faiss_indices) == 0:
            print("[WARN] No FAISS indices provided, skipping DB lookup")
            return None

        faiss_indices = [int(i) for i in faiss_indices if i is not None]

        ids_str = ",".join(str(i) for i in faiss_indices)
        sql = f"SELECT video_url, faiss_index FROM video_vectors WHERE faiss_index IN ({ids_str})"

        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor._defer_warnings = True
            cursor.arraysize = 100

            cursor.execute(sql)
            try:
                rows = cursor.fetchall()
            except Exception as fe:
                print(f"[WARN] fetchall() failed, trying fetchmany(10): {fe}")
                rows = cursor.fetchmany(10)

        conn.close()

        if not rows:
            print("[WARN] No matching rows found for given FAISS indices")
            return None


        # Count which video_url appears most frequently
        from collections import Counter
        urls = [r[0] if isinstance(r, (list, tuple)) else r["video_url"] for r in rows]
        most_common = Counter(urls).most_common(1)[0][0]
        return most_common

    except Exception as e:
        print(f"[DB ERROR] Query failed: {e}")
        return None
