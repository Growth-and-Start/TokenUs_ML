# AWS ECR 로그인
aws ecr get-login-password --region ap-northeast-2 | docker login --username AWS --password-stdin 343218215881.dkr.ecr.ap-northeast-2.amazonaws.com

# ECR에서 최신 이미지 pull
echo "ECR에 있는 이미지 불러오기"
if ! docker pull 343218215881.dkr.ecr.ap-northeast-2.amazonaws.com/tokenus/ml:latest; then
    echo "이미지 불러오기에 실패했습니다."
    exit 1
fi

# Docker compose down으로 기존 컨테이너 중지 및 삭제
echo "Docker compose down 실행"
docker compose down --remove-orphans

# 볼륨이 없으면 생성
docker volume create faiss_data || true

# Docker compose up 실행
echo "Docker compose up 실행"
if ! docker compose up -d; then
    echo "컨테이너 실행에 실패했습니다"
    exit 1
fi

# dangling 이미지 삭제
echo "dangling 이미지 삭제"
docker image prune -f

echo "멈춘 container 삭제"
docker container prune -f

for i in {1..10}; do
    if [ "$i" -eq 10 ]; then
       echo "Health check failed"
       docker compose down
       exit 1
    fi

    if curl "http://localhost:5000/health"; then
        echo "Flask 컨테이너가 정상적으로 실행되었습니다..."
        break
    fi

    echo "Flask 서버 health check 중..."
    sleep 15
done

echo "모든 작업이 완료되었습니다."