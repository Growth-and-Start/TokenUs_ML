#!/bin/bash
set -e
source .env

echo "Running DB migration for flask_db..."

# MySQL 준비 확인 루프
for i in $(seq 1 10); do
  if docker exec tokenus-mysql-flask mysql -u"${FLASK_DB_USER}" -p"${FLASK_DB_PASSWORD}" -e "SELECT 1;" >/dev/null 2>&1; then
      echo "MySQL is ready."
      break
  fi
  echo "Waiting for MySQL to be ready... ($i/10)"
  sleep 5
done

# DB 생성 및 권한 부여
docker exec -i tokenus-mysql-flask mysql -uroot -p"${MYSQL_ROOT_PASSWORD}" <<EOF
CREATE DATABASE IF NOT EXISTS flask_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS '${FLASK_DB_USER}'@'%' IDENTIFIED BY '${FLASK_DB_PASSWORD}';
GRANT ALL PRIVILEGES ON flask_db.* TO '${FLASK_DB_USER}'@'%';
FLUSH PRIVILEGES;
EOF

# 테이블 생성 (schema.sql을 주입)
docker cp migrations/schema.sql tokenus-mysql-flask:/schema.sql
docker exec -i tokenus-mysql-flask mysql -u"${FLASK_DB_USER}" -p"${FLASK_DB_PASSWORD}" flask_db < /schema.sql

echo "DB migration completed successfully!"
