# 🪙TokenUs🪙

## Team Info
| Heejae An | Jimin Seo | Wonyoung Kim |
| --- | --- | --- |
| @AnyJae | @SeoJimin1234    | @lasagna10 |


## Video Similarity Checker(TokenUs ML)

### Overview
 This project provides a video similarity detection system based on deep learning and vectr search.
 It was originally developed as part of the TokenUs platfor, but is released here as a standalone open-source component.

 The system ensures video originality by comparing uploaded videos against an index of previously processed content, preventing duplication and enabling downstream applications such as copyright protection and content verification.

-----

### Features
- **Frame-based feature extraction** using a pretrained **ResNet-50** model.
- **Efficient vector similarity search** powered by **FAISS**.
- **Cosine similarity scoring** with configurable thresholds.
-** AWS S3 integration** for video storage and retrieval.
- **REST API(Flask)** for easy integration with external services.

-----
### Workflow
1. **Download** video from S3 using its URL.
2. **Extract frames** and compute feature embeddings bia ResNet-50.
3. **Search FAISS index** and claculate cosine similarities.(You can control the Duplicate criteria)
4. **Return result** via REST API and update storage/index accordingly.

-----
### Tech Stack
<img src="https://img.shields.io/badge/flask-000000?style=for-the-badge&logo=flask&logoColor=white"><br>
<img src="https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"><br>
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"><br>
<img src="https://img.shields.io/badge/docker-2496ED?style=for-the-badge&logo=docker&logoColor=white"><br>
<img src="https://img.shields.io/badge/amazons3-569A31?style=for-the-badge&logo=amazons3&logoColor=white"><br>
<img src="https://img.shields.io/badge/amazonec2-FF9900?style=for-the-badge&logo=amazonec2&logoColor=white"><br>
<img src="https://img.shields.io/badge/opencv-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"><br>
<img src="https://img.shields.io/badge/FAISS-000000?style=for-the-badge&logo=&logoColor=white"><br>

-------

### Installation
#### 1. Clone the repository
```
git clone https://github.com/Growth-and-Start/TokenUs_ML.git
cd TokenUs_ML
```
#### 2. Create a .env file

```
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_S3_BUCKET=
AWS_DEFAULT_REGION=
AWS_ACCOUNT_ID=

MYSQL_ROOT_PASSWORD=
MYSQL_DATABASE=
MYSQL_USER=
MYSQL_PASSWORD=

BACKEND_URL=
API_PATH=

CONTAINER_NAME=
IMAGE_NAME=
NETWORK_NAME=
VOLUME_NAME=
FLASK_PORT=

CPU_LIMIT=
MEMORY_LIMIT=

FAISS_INDEX_PATH=
DOWNLOAD_FOLDER=

FLASK_PORT=
```
#### 3. Run with Docker Compose
```
cd docker/deploy

docker-compose up -d
```

---------

### API Example
```
POST /similarity-check
Content-Type: application/json

{
  "s3_url": "https://s3.amazonaws.com/bucket/video.mp4"
}

```
#### Response
```
{
  "is_duplicate": false,
  "max_similarity": 0.62,
  "avg_similarity": 0.45
}
```

------
###License
MIT License © 2025 TokenUs Team
