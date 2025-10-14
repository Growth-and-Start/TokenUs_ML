                                                                             migrations/schema.sql                                                                                         CREATE DATABASE IF NOT EXISTS flask_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE flask_db;

CREATE TABLE IF NOT EXISTS video_vectors (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    video_url VARCHAR(255) NOT NULL,
    faiss_index INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);