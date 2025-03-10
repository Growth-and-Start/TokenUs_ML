from flask import Flask, request, jsonify
import os
import requests
import boto3
from dotenv import load_dotenv
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import faiss
import numpy as np

app = Flask(__name__)

# 환경 변수 로드
load_dotenv(".env")

# 환경 변수 로드
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET_NAME]):
    raise ValueError("환경 변수가 제대로 설정되지 않았습니다!")


# S3 클라이언트 생성
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# 다운로드된 영상을 저장할 폴더 생성
DOWNLOAD_FOLDER = "downloads"
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

def delete_s3_file(s3_url):
    """
    S3에 업로드된 영상을 삭제하는 함수
    """
    try:
        # S3 URL에서 버킷명과 파일 키 추출
        bucket_name = "tokenus-storage"  # ✅ S3 버킷명
        key = s3_url.replace(f"https://{bucket_name}.s3.ap-northeast-2.amazonaws.com/", "")

        # S3 객체 삭제
        s3_client.delete_object(Bucket=bucket_name, Key=key)
        print(f"✅ S3에서 파일 삭제 성공: {s3_url}")

    except Exception as e:
        print(f"❌ S3에서 파일 삭제 실패: {s3_url}, 오류: {e}")

@app.route("/download", methods=['POST'])
def download_video():
    data = request.json
    video_url = data.get('file_url')
    
    if not video_url:
        return jsonify({"error": "No video_url provided"}), 400
    
    # S3 객체 키(파일 경로) 추출
    object_key = video_url.split(".com/")[-1]  # "videos/test3.mov"
    
    try:
        # 다운로드할 파일 경로 설정
        filename = object_key.split("/")[-1]
        file_path = os.path.join(DOWNLOAD_FOLDER, filename)
        
        # S3에서 파일 다운로드
        s3_client.download_file(S3_BUCKET_NAME, object_key, file_path)
        
        #return jsonify({"message": "Download successful", "file_path": file_path})
        
        # ✅ 자동으로 check_similarity 수행
        similarity_result = perform_similarity_check(file_path)

        return jsonify({
            "message": "Download successful",
            "file_path": file_path,
            "similarity_check_result": similarity_result
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def perform_similarity_check(video_path):
    """
    저장된 모든 벡터와 입력된 영상의 벡터 간 코사인 유사도 비교.
    같은 영상인지 판단하는 기준:
    - 한 개 이상의 벡터가 0.8 이상
    - 평균 유사도가 0.75 이상
    """
    if not video_path or not os.path.exists(video_path):
        return {"error": "Invalid video path"}

    try:
        # 1️⃣ FAISS 인덱스 로드
        load_faiss_index()

        # 2️⃣ 저장된 벡터 개수 확인
        total_vectors = faiss_index.index.ntotal
        if total_vectors == 0:
            similarity_result = {
                "message": "유사도 검사를 통과하였습니다",
                "max_similarity": 0,
                "avg_similarity": 0
            }
            notify_springboot(video_path, similarity_result, passed=True)
            return similarity_result
            

        # 3️⃣ 비교할 영상에서 프레임 추출
        frames = extract_frames(video_path)

        # 4️⃣ 프레임의 특징 벡터 추출
        feature_vectors = extract_features(frames)

        # 5️⃣ 벡터 정규화 (코사인 유사도 기반 비교)
        query_vectors = np.array(feature_vectors).astype('float32')
        faiss.normalize_L2(query_vectors)

        # 6️⃣ 각 프레임의 벡터를 FAISS에 저장된 모든 벡터와 비교
        similarity_scores = []
        for query_vector in query_vectors:
            query_vector = query_vector.reshape(1, -1)
            faiss.normalize_L2(query_vector)  # ✅ 검색 벡터 정규화
            distances, _ = faiss_index.index.search(query_vector, total_vectors)
            similarity_scores.extend(distances[0].tolist())

        # 7️⃣ 검사 기준 적용
        max_similarity = max(similarity_scores)
        avg_similarity = sum(similarity_scores) / len(similarity_scores)

        similarity_result = {
            "max_similarity": max_similarity,
            "avg_similarity": avg_similarity
        }

        if max_similarity >= 1.0 or (max_similarity >= 0.9 and avg_similarity >= 0.8):
            similarity_result["message"] = "유사도 검사를 실패하였습니다"
            # ❌ 로컬 파일 삭제
            delete_file(video_path)

            # ❌ S3에서도 삭제
            delete_s3_file(video_path)

            # ❌ Spring Boot 서버에 유사도 검사 실패 알림
            notify_springboot(video_path, similarity_result, passed=False)
            return similarity_result
        else:
            similarity_result["message"] = "유사도 검사를 통과하였습니다"
            # ✅ FAISS에 벡터 저장
            faiss_index.add_vectors(feature_vectors)
            save_faiss_index()  # 저장된 FAISS 인덱스 파일 업데이트

            # ❌ 로컬 파일 삭제
            delete_file(video_path)

            # ✅ Spring Boot 서버에 유사도 검사 성공 알림
            notify_springboot(video_path, similarity_result, passed=True)
            
            return similarity_result

    except Exception as e:
        return {"error": str(e)}

def delete_file(file_path):
    """로컬에 저장된 영상 파일 삭제"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"🗑 파일 삭제 완료: {file_path}")
    except Exception as e:
        print(f"🚨 파일 삭제 오류: {e}")
        
        
SPRINGBOOT_URL = "http://127.0.0.1:8080/video/similarity-check"  # Spring Boot 서버 URL

def notify_springboot(video_path, similarity_result, passed):
    """
    Spring Boot 서버에 유사도 검사 결과 전송
    :param video_path: 검사한 영상 경로
    :param similarity_result: 유사도 검사 결과
    :param passed: 유사도 검사를 통과했는지 여부 (True: 통과, False: 실패)
    """
    payload = {
        "video_path": video_path,
        "max_similarity": similarity_result["max_similarity"],
        "avg_similarity": similarity_result["avg_similarity"],
        "message": similarity_result["message"],
        "passed": passed  # ✅ 통과 여부 추가
    }

    try:
        print("payload:",payload)
        headers = {'Content-Type': 'application/json'}
        response = requests.post(SPRINGBOOT_URL, json=payload, headers=headers)
        print(f"📡 Spring Boot 응답: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"🚨 Spring Boot 전송 오류: {e}")


#영상 프레임 추출
def extract_frames(video_path, interval=1):
    """
    주어진 영상에서 일정 간격마다 프레임을 추출하는 함수
    :param video_path: 영상 파일 경로
    :param interval: 초 단위 간격 (1초마다 프레임 추출)
    :return: 프레임 이미지 리스트
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Error: Unable to open video file.")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 초당 프레임 수
    frame_interval = fps * interval  # 간격 설정
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 영상 종료 시 반복 중지

        if frame_count % frame_interval == 0:
            frames.append(frame)

        frame_count += 1

    cap.release()
    return frames
    

# ResNet-50 모델 로드
resnet = resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # 마지막 FC 레이어 제거
resnet.eval()  # 모델을 평가 모드로 설정

# 이미지 전처리 변환 정의
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(frames):
    """
    ResNet-50을 사용해 주어진 프레임 리스트에서 특징 벡터를 추출
    :param frames: 영상 프레임 리스트
    :return: 특징 벡터 리스트 (2048차원)
    """
    features = []
    with torch.no_grad():  # 그래디언트 계산 비활성화
        for frame in frames:
            img_tensor = transform(frame).unsqueeze(0)  # 배치 차원 추가
            feature_vector = resnet(img_tensor)  # (1, 2048, 1, 1) 형태 출력
            feature_vector = feature_vector.view(-1).numpy()  # 1D 벡터로 변환
            features.append(feature_vector)

    return features
    

class FAISSIndex:
    def __init__(self, dim=2048):
        """
        FAISS를 사용해 특징 벡터를 저장하는 클래스
        :param dim: 벡터 차원 (ResNet-50은 2048)
        """
        # self.index = faiss.IndexFlatL2(dim)  # L2 거리 기반 인덱스 생성
        self.index = faiss.IndexFlatIP(dim) # 내적 기반 인덱스(코사인 유사도에 적합)

    def add_vectors(self, vectors):
        """
        벡터 추가
        :param vectors: 특징 벡터 리스트 (numpy array)
        """
        vectors = np.array(vectors).astype('float32')
        faiss.normalize_L2(vectors) # 벡터 정규화
        self.index.add(vectors)

    def search(self, query_vector, top_k=5):
        """
        가장 유사한 벡터 검색
        :param query_vector: 검색할 벡터 (2048차원)
        :param top_k: 상위 K개 검색
        :return: 유사한 벡터 인덱스와 거리
        """
        query_vector = np.array(query_vector).astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_vector) # 정규화
        distances, indices = self.index.search(query_vector, top_k)
        return indices, distances
    

def convert_faiss_to_cosine():
    """L2 거리 기반 FAISS를 코사인 유사도 기반으로 변환"""
    global faiss_index

    total_vectors = faiss_index.index.ntotal
    if total_vectors == 0:
        print("📢 FAISS에 저장된 벡터가 없습니다. 변환할 필요 없음.")
        return

    print(f"🔄 기존 L2 기반 벡터 {total_vectors}개 변환 중...")

    # 기존 벡터 가져오기
    stored_vectors = np.zeros((total_vectors, 2048), dtype=np.float32)
    for i in range(total_vectors):
        stored_vectors[i] = faiss_index.index.reconstruct(i)

    # 벡터 정규화 (코사인 유사도 계산을 위해)
    faiss.normalize_L2(stored_vectors)

    # 새로운 코사인 유사도 기반 FAISS 인덱스 생성 (IndexFlatIP 사용)
    new_index = faiss.IndexFlatIP(2048)
    new_index.add(stored_vectors)  # 정규화된 벡터 추가

    # 기존 FAISS 인덱스를 새로운 인덱스로 교체
    faiss_index.index = new_index

    # 변환된 FAISS 인덱스를 저장
    save_faiss_index()

    print("✅ L2 → 코사인 유사도 변환 완료 및 저장됨!")

 
FAISS_INDEX_PATH = "faiss_index.bin"

def save_faiss_index():
    """FAISS 인덱스를 파일로 저장"""
    faiss.write_index(faiss_index.index, FAISS_INDEX_PATH)

def load_faiss_index():
    """FAISS 인덱스를 파일에서 불러오기"""
    global faiss_index
    if os.path.exists(FAISS_INDEX_PATH):
        print("🔄 FAISS 인덱스 로드 완료!")
        index = faiss.read_index(FAISS_INDEX_PATH)
        faiss_index.index = index  # 기존 객체를 덮어씌우지 않고 유지

# FAISS 인덱스 전역 변수로 생성
faiss_index = FAISSIndex(dim=2048)

# 서버 시작 시 FAISS 인덱스 로드
load_faiss_index()
convert_faiss_to_cosine()  # ✅ 기존 데이터가 있으면 변환

@app.route('/process_video', methods=['POST'])
def process_video():
    data = request.json
    video_path = data.get("video_path")  # 다운로드된 영상 경로

    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "Invalid video path"}), 400

    try:
        # 1️⃣ 영상에서 프레임 추출
        frames = extract_frames(video_path)

        # 2️⃣ 프레임에서 특징 벡터 추출
        feature_vectors = extract_features(frames)

        # 3️⃣ FAISS에 벡터 추가
        faiss_index.add_vectors(feature_vectors)
        
        #4️⃣ FAISS 인덱스 저장(벡터 추가 후 자동 저장)
        save_faiss_index()

        return jsonify({"message": "Video processed successfully", "num_frames": len(frames)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/faiss_info', methods=['GET'])
def faiss_info():
    """
    현재 FAISS에 저장된 벡터 개수를 확인하는 API
    """
    try:
        load_faiss_index()
        total_vectors = faiss_index.index.ntotal
        print(f"📊 현재 저장된 벡터 개수: {total_vectors}")  # ✅ 디버깅 로그 추가
        return jsonify({"message": "FAISS index info", "total_vectors": total_vectors})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/search_similar', methods=['POST'])
def search_similar():
    data = request.json
    video_path = data.get("video_path")  # 검색할 영상 파일 경로

    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "Invalid video path"}), 400

    try:
        # 1️⃣ FAISS 인덱스 로드 (혹시 파일이 존재하지 않을 경우 대비)
        load_faiss_index()
        
        # 1️⃣ 영상에서 프레임 추출
        frames = extract_frames(video_path)

        # 2️⃣ 프레임에서 특징 벡터 추출
        feature_vectors = extract_features(frames)
        
        # FAISS에 저장된 벡터 개수 확인
        total_vectors = faiss_index.index.ntotal
        
        if total_vectors ==0:
            return jsonify({"error":"No vectors stored in FAISS"}), 400

        # 3️⃣ FAISS에서 유사한 벡터 검색 (첫 번째 프레임 기준)
        query_vector = feature_vectors[0]
        
        top_k = min(5, total_vectors) #저장된 벡터 개수보다 많지 않게 제한
        
        indices, distances = faiss_index.search(query_vector, top_k)


        return jsonify({"message": "Search completed", "indices": indices.tolist(), "distances": distances.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def normalize_faiss_index():
    """FAISS에 저장된 벡터를 정규화하여 코사인 유사도 기반 검색이 가능하게 함"""
    total_vectors = faiss_index.index.ntotal
    if total_vectors > 0:
        stored_vectors = np.zeros((total_vectors, 2048), dtype=np.float32)
        for i in range(total_vectors):
            stored_vectors[i] = faiss_index.index.reconstruct(i)
        faiss.normalize_L2(stored_vectors)  # 정규화 수행
        faiss_index.index = faiss.IndexFlatIP(2048)  # 내적을 사용하도록 설정
        faiss_index.index.add(stored_vectors)  # 정규화된 벡터 추가


@app.route('/check_similarity', methods=['POST'])
def check_similarity():
    """
    저장된 모든 벡터와 입력된 영상의 벡터 간 코사인 유사도 비교
    - 한 개 이상의 벡터가 0.95 이상이면서, 평균 유사도가 0.8 이상이면 같은 영상으로 판단
    """
    data = request.json
    video_path = data.get("video_path")  # 비교할 영상 파일 경로

    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "Invalid video path"}), 400

    try:
        # 1️⃣ FAISS 인덱스 로드 (파일이 있다면 불러오기)
        load_faiss_index()

        # 2️⃣ 저장된 벡터 개수 확인
        total_vectors = faiss_index.index.ntotal
        if total_vectors == 0:
            return jsonify({"message": "유사도 검사를 통과하였습니다"}), 200

        # 3️⃣ 비교할 영상에서 프레임 추출
        frames = extract_frames(video_path)

        # 4️⃣ 프레임의 특징 벡터 추출
        feature_vectors = extract_features(frames)

        # 5️⃣ 벡터 정규화 (코사인 유사도 기반 비교)
        query_vectors = np.array(feature_vectors).astype('float32')
        faiss.normalize_L2(query_vectors)

        # 6️⃣ 각 프레임의 벡터를 FAISS에 저장된 모든 벡터와 비교
        similarity_scores = []
        for query_vector in query_vectors:
            query_vector = query_vector.reshape(1, -1)
            
            faiss.normalize_L2(query_vector) # ✅ 검색 벡터 정규화
            # FAISS에서 저장된 모든 벡터와 유사도 검사
            distances, _ = faiss_index.index.search(query_vector, total_vectors)

            # FAISS는 내적을 반환하므로, 값이 클수록 유사한 것 (1.0에 가까울수록 유사)함
            similarity_scores.extend(distances[0].tolist())

        # 7️⃣ 검사 기준 적용
        max_similarity = max(similarity_scores)  # 가장 높은 유사도
        avg_similarity = sum(similarity_scores) / len(similarity_scores)  # 평균 유사도

        if max_similarity >= 0.9 and avg_similarity >= 0.8:
            return jsonify({
                "message": "같은 영상이 이미 존재합니다",
                "max_similarity": max_similarity,
                "avg_similarity": avg_similarity
            }), 200
        else:
            return jsonify({
                "message": "유사도 검사를 통과하였습니다",
                "max_similarity": max_similarity,
                "avg_similarity": avg_similarity
            }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500




if __name__ == '__main__':
    app.run(debug=True)