#!/bin/bash
set -e

# ==============================
# .env file load
# ==============================
if [ -f .env ]; then
  export $(cat .env | grep -v '^#' | xargs)
else
  echo "There are no .env file"
  exit 1
fi

# ==============================
# AWS ECR login
# ==============================
aws ecr get-login-password --region $AWS_DEFAULT_REGION | \
docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com

# ==============================
# Docker network
# ==============================
if ! docker network ls | grep -q "$NETWORK_NAME"; then
  echo "$NETWORK_NAME network does not exist, creating it..."
  docker network create $NETWORK_NAME
else
  echo "$NETWORK_NAME network already exists."
fi

# ==============================
# ECR image pull
# ==============================
echo "ECR image pull"
if ! docker pull $IMAGE_NAME; then
    echo "Failed to pull image."
    exit 1
fi

# ==============================
# Docker compose down
# ==============================
echo "Docker compose down"
docker compose down --remove-orphans

docker volume create faiss_data || true

# ==============================
# Docker compose up
# ==============================
echo "Docker compose up"
if ! docker compose up -d; then
    echo "Failed to start containers"
    exit 1
fi

# ==============================
# Resource cleanup
# ==============================
echo "dangling image delete"
docker image prune -f

echo "Stopped container delete"
docker container prune -f

# ==============================
# DB Migration
# ==============================
echo "DB migration in progress..."
for i in {1..10}; do
    if docker exec tokenus-mysql-flask mysql -u${FLASK_DB_USER} -p${FLASK_DB_PASSWORD} -e "SELECT 1;" >/dev/null 2>&1; then
        echo "MySQL container is ready. Starting migration..."
        docker exec -i tokenus-mysql-flask mysql -uroot -p${MYSQL_ROOT_PASSWORD} flask_db < migrations/schema.sql
        echo "DB migration completed!"
        break
    fi
    echo "Waiting for MySQL to be ready... ($i/10)"
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
        echo "Flask container is running normally..."
        break
    fi

    echo "Flask server health check in progress..."
    sleep 15
done

echo "All tasks are completed."
