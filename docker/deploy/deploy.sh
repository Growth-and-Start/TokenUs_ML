#!/bin/bash
set -e

# ==============================
# .env 파일 불러오기
# ==============================
if [ -f .env ]; then
  export $(cat .env | grep -v '^#' | xargs)
else
  echo ".env 파일이 존재하지 않습니다. 먼저 생성해주세요."
  exit 1
fi

# ==============================
# AWS ECR 로그인
# ==============================
aws ecr get-login-password --region $AWS_DEFAULT_REGION | \
docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com

# ==============================
# Docker network 확인 및 생성
# ==============================
if ! docker network ls | grep -q "$NETWORK_NAME"; then
  echo "$NETWORK_NAME 네트워크가 없어 새로 생성합니다..."
  docker network create $NETWORK_NAME
else
  echo "$NETWORK_NAME 네트워크가 이미 존재합니다."
fi

# ==============================
# ECR에서 최신 이미지 pull
# ==============================
echo "ECR에 있는 이미지 불러오기"
if ! docker pull $IMAGE_NAME; then
    echo "이미지 불러오기에 실패했습니다."
    exit 1
fi

# ==============================
# 기존 컨테이너 종료 및 볼륨 확인
# ==============================
echo "Docker compose down 실행"
docker compose down --remove-orphans

docker volume create faiss_data || true

# ==============================
# Docker compose up 실행
# ==============================
echo "Docker compose up 실행"
if ! docker compose up -d; then
    echo "컨테이너 실행에 실패했습니다"
    exit 1
fi

# ==============================
# 불필요한 리소스 정리
# ==============================
echo "dangling 이미지 삭제"
docker image prune -f

echo "멈춘 container 삭제"
docker container prune -f

# ==============================
# DB Migration
# ==============================
echo "DB 마이그레이션 실행..."
for i in {1..10}; do
    if docker exec tokenus-mysql-flask mysql -u${FLASK_DB_USER} -p${FLASK_DB_PASSWORD} -e "SELECT 1;" >/dev/null 2>&1; then
        echo "MySQL 컨테이너 준비 완료. 마이그레이션 실행..."
        docker exec -i tokenus-mysql-flask mysql -uroot -p${MYSQL_ROOT_PASSWORD} flask_db < migrations/schema.sql
        echo "DB 마이그레이션 완료!"
        break
    fi
    echo "MySQL 준비 대기 중... ($i/10)"
    sleep 10
fi

# ==============================
# Health Check
# ==============================
for i in {1..10}; do
    if [ "$i" -eq 10 ]; then
       echo "Health check failed"
       docker compose down
       exit 1
    fi

    if curl "http://localhost:$FLASK_PORT/health"; then
        echo "Flask 컨테이너가 정상적으로 실행되었습니다..."
        break
    fi

    echo "Flask 서버 health check 중..."
    sleep 15
done

echo "모든 작업이 완료되었습니다."
