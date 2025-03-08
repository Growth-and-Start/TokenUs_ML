import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from scipy.spatial.distance import cosine

import boto3
from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, AWS_S3_BUCKET_NAME

import requests

import os
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Flask 영상 유사도 검사 프로젝트 시작!"

#-----파일 업로드-------

# 업로드된 파일을 저장할 폴더 설정
UPLOAD_FOLDER = 'uploads'
FRAMES_FOLDER = 'frames'  # 프레임 저장 폴더
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['FRAMES_FOLDER'] = FRAMES_FOLDER

# 폴더가 없으면 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)

# 허용된 확장자인지 확인하는 함수
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 로컬 업로드 엔드포인트()
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     if file and allowed_file(file.filename):
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#         file.save(filepath)
#         return jsonify({'message': 'File uploaded successfully', 'filename': file.filename}), 200

#     return jsonify({'error': 'Invalid file format'}), 400


# ----------프레임 추출-------------
# 프레임 추출 함수
def extract_frames(video_path, output_folder, frame_interval=30):
    """
    동영상에서 일정 간격으로 프레임을 추출하는 함수
    - video_path: 동영상 파일 경로
    - output_folder: 프레임 저장 폴더
    - frame_interval: N프레임마다 한 개의 프레임 저장 (기본값: 30)
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_output_path = os.path.join(output_folder, video_name)
    os.makedirs(frame_output_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    extracted_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(frame_output_path, f"frame_{extracted_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1

        frame_count += 1

    cap.release()
    return extracted_count  # 추출된 프레임 개수 반환

@app.route('/extract_frames', methods=['POST'])
def extract_video_frames():
    if 'filename' not in request.json:
        return jsonify({'error': 'Filename is required'}), 400

    filename = request.json['filename']
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(video_path):
        return jsonify({'error': 'File not found'}), 404

    extracted_count = extract_frames(video_path, app.config['FRAMES_FOLDER'])
    return jsonify({'message': 'Frames extracted successfully', 'frames_extracted': extracted_count}), 200

#------------벡터 추출 및 유사도 검사---------------
# 사전 학습된 모델 로드 (ResNet-50 사용)
model = models.resnet50(pretrained=True)
model.eval()  # 평가 모드로 설정

# 입력 이미지를 모델에 맞게 변환하는 함수
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet 입력 크기로 변환
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_feature_vector(image_path):
    """이미지를 로드하고 특징 벡터를 추출하는 함수"""
    image = Image.open(image_path).convert('RGB')  # 이미지를 RGB로 변환
    image = transform(image).unsqueeze(0)  # 배치 차원 추가

    with torch.no_grad():
        features = model(image)  # 특징 벡터 추출

    return features.numpy().flatten()  # 벡터를 1차원으로 변환

def compare_feature_vectors(image1_path, image2_path):
    """ 두 이미지의 특징 벡터를 비교하여 유사도를 계산 """
    vector1 = extract_feature_vector(image1_path)
    vector2 = extract_feature_vector(image2_path)

    similarity = 1 - cosine(vector1, vector2)  # 코사인 유사도 계산
    return similarity

@app.route('/compare', methods=['POST'])
def compare_videos():
    """ 두 영상의 프레임을 비교하여 유사도를 계산 """
    data = request.json
    if 'video1' not in data or 'video2' not in data:
        return jsonify({'error': 'Two filenames are required'}), 400

    video1 = data['video1']
    video2 = data['video2']

    video1_path = os.path.join(app.config['FRAMES_FOLDER'], video1)
    video2_path = os.path.join(app.config['FRAMES_FOLDER'], video2)

    if not os.path.exists(video1_path) or not os.path.exists(video2_path):
        return jsonify({'error': 'One or both videos not found'}), 404

    frame_list1 = sorted(os.listdir(video1_path))
    frame_list2 = sorted(os.listdir(video2_path))

    if not frame_list1 or not frame_list2:
        return jsonify({'error': 'No frames extracted for comparison'}), 400

    similarities = []
    for frame1, frame2 in zip(frame_list1, frame_list2):
        image1_path = os.path.join(video1_path, frame1)
        image2_path = os.path.join(video2_path, frame2)

        similarity = compare_feature_vectors(image1_path, image2_path)
        similarities.append(similarity)

    avg_similarity = np.mean(similarities)

    return jsonify({
        "average_feature_similarity": float(avg_similarity)
    }), 200
    
    
#--------S3 업로드 기능-----------

# AWS S3 클라이언트 설정
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

def upload_to_s3(file_path, s3_key):
    """ S3에 파일 업로드 """
    try:
        s3_client.upload_file(file_path, AWS_S3_BUCKET_NAME, s3_key)
        return f"https://{AWS_S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
    except Exception as e:
        print(f"S3 업로드 오류: {e}")
        return None
    
@app.route('/upload', methods=['POST'])
def upload_file():
    """파일을 업로드하고 S3에 저장"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    local_file_path = os.path.join("uploads", file.filename)
    file.save(local_file_path)

    # S3에 업로드
    s3_url = upload_to_s3(local_file_path, f"videos/{file.filename}")

    if not s3_url:
        return jsonify({'error': 'Failed to upload to S3'}), 500

    return jsonify({'message': 'File uploaded successfully', 's3_url': s3_url}), 200


#--------check similarity-----



# @app.route('/check_similarity', methods=['POST'])
# def check_similarity_and_upload():
#     """ 새로 업로드된 영상과 기존 모든 영상의 유사도를 비교한 후, 너무 유사하지 않으면 S3에 저장 """
#     data = request.json
#     if 'filename' not in data:
#         return jsonify({'error': 'Filename is required'}), 400

#     new_video = data['filename']
#     new_video_path = os.path.join(app.config['UPLOAD_FOLDER'], new_video)

#     if not os.path.exists(new_video_path):
#         return jsonify({'error': 'File not found'}), 404

#     # 기존 업로드된 영상 목록 가져오기 (Spring Boot에서 조회)
#     existing_videos = get_existing_videos()

#     if not existing_videos:
#         # 기존 영상이 없으면 바로 S3에 업로드
#         s3_url = upload_to_s3(new_video_path, f"videos/{new_video}")
#         return jsonify({'message': 'File uploaded successfully', 's3_url': s3_url}), 200

#     # 새 영상 프레임 벡터 추출
#     new_vectors = extract_video_vectors(new_video_path)

#     # 기존 영상과 비교
#     similarities = []
#     for existing_video in existing_videos:
#         existing_video_path = os.path.join(app.config['FRAMES_FOLDER'], existing_video)

#         if not os.path.exists(existing_video_path):
#             continue

#         existing_vectors = extract_video_vectors(existing_video_path)

#         # 벡터 간 유사도 비교 (평균값)
#         similarity_scores = [
#             1 - cosine(new_vector, existing_vector)
#             for new_vector, existing_vector in zip(new_vectors, existing_vectors)
#         ]

#         avg_similarity = np.mean(similarity_scores)
#         similarities.append(avg_similarity)

#     # 가장 높은 유사도 찾기
#     max_similarity = max(similarities) if similarities else 0

#     # 유사도가 너무 높으면 업로드하지 않음 (예: 90% 이상)
#     SIMILARITY_THRESHOLD = 0.9
#     if max_similarity >= SIMILARITY_THRESHOLD:
#         return jsonify({'error': 'Too similar to existing videos', 'similarity': max_similarity}), 400

#     # 유사도가 낮다면 S3에 저장
#     s3_url = upload_to_s3(new_video_path, f"videos/{new_video}")
#     return jsonify({'message': 'File uploaded successfully', 's3_url': s3_url}), 200


#-------main-------------

if __name__ == '__main__':
    app.run(debug=True)
