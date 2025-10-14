import os
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from config import Config
from extensions.s3 import get_s3_client
import subprocess
import numpy as np
import tempfile

s3_client = get_s3_client()

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
    """
    Extract frames using ffmpeg at every `interval` seconds.
    Returns: list of frames (as numpy arrays, in RGB)
    """
    tmp_dir = tempfile.mkdtemp()
    output_pattern = os.path.join(tmp_dir, "frame_%06d.jpg")

    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps=1/{interval}",
        "-q:v", "2",
        output_pattern,
        "-hide_banner", "-loglevel", "error"
    ]

    subprocess.run(cmd, check=True)

    frames = []
    for file in sorted(os.listdir(tmp_dir)):
        if file.endswith(".jpg"):
            frame_path = os.path.join(tmp_dir, file)
            frame = cv2.imread(frame_path)
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

    for file in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, file))
    os.rmdir(tmp_dir)

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
