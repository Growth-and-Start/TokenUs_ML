import os

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = "ap-northeast-2"  # 서울 리전
AWS_S3_BUCKET_NAME = "tokenus-storage"  # S3 버킷 이름
