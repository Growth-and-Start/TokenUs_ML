import os
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from config import Config
from extensions.s3 import get_s3_client

# S3 클라이언트
s3_client = get_s3_client()

# 다운로드 폴더
os.makedirs(Config.DOWNLOAD_FOLDER, exist_ok=True)

def download_video_from_s3(s3_url):
    object_key = s3_url.split(".com/")[-1]
    filename = object_key.split("/")[-1]
    file_path = os.path.join(Config.DOWNLOAD_FOLDER, filename)
    s3_client.download_file(Config.S3_BUCKET_NAME, object_key, file_path)
    return file_path

def delete_s3_file(s3_url):
    try:
        key = s3_url.replace(f"https://{Config.S3_BUCKET_NAME}.s3.{Config.AWS_REGION}.amazonaws.com/", "")
        s3_client.delete_object(Bucket=Config.S3_BUCKET_NAME, Key=key)
    except Exception as e:
        print(f"S3 delete error: {e}")

def extract_frames(video_path, interval=1):
    """Extract frames from video at specified interval"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error: Unable to open video file.")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = fps * interval
    frames, frame_count = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append(frame)
        frame_count += 1
    cap.release()
    return frames

# ResNet-50 Model Load
resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def extract_features(frames):
    """Extract feature vectors using ResNet-50"""
    features = []
    with torch.no_grad():
        for frame in frames:
            img_tensor = transform(frame).unsqueeze(0)
            feature_vector = resnet(img_tensor)
            feature_vector = feature_vector.view(-1).numpy()
            features.append(feature_vector)
    return features

def delete_local_file(file_path):
    """Delete local file"""
    if os.path.exists(file_path):
        os.remove(file_path)
