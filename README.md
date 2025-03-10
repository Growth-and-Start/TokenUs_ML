# 🪙TokenUs🪙
- 이화여자대학교 컴퓨터공학과 캡스톤디자인과창업프로젝트A,B
- 개발 기간: 2024.09 ~ 진행 중

## 팀 소개 : 8시 스쿼시 연맹
| 안희재 | 서지민 | 김원영 |
| --- | --- | --- |
| 1 | ![seojimin](https://ibb.co/gM8sWp6G)    | 3 |
| -FE 개발<br>-SmartContract개발 | -BE 개발<br>-ML 개발<br>-SmartContract 개발| -UX/UI 디자인<br>-FE개발<br>-SmartContract 개발 |
| 7 | @Seojimin1234 | 9 |


## 프로젝트 소개
 영상을 NFT로 발행하여 영상의 고유 가치를 지키고, 불법 복제를 방지하며, 원저작자의 권리를 보호하고 투자의 기회까지 제공하는 영상 플랫폼.
### 💡주요 기능1 - 영상 유사도 검사
사전 학습된 ResNet-50 모델과 Cosine Similarity를 활용한 유사도 검사. 영상의 고유성과 NFT의 가치를 보호하고, 불법 복제 방지.
### 💡주요 기능2 - NFT 발행
Ethereum을 기반으로 한 NFT 발행
### 💡주요 기능3 - NFT 거래
유저 간 자유로운 NFT 거래. 수익을 기대할 수 있음
## Stacks
<img src="https://img.shields.io/badge/flask-000000?style=for-the-badge&logo=flask&logoColor=white"><br>
<img src="https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"><br>
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"><br>
<img src="https://img.shields.io/badge/docker-2496ED?style=for-the-badge&logo=docker&logoColor=white"><br>
<img src="https://img.shields.io/badge/amazons3-569A31?style=for-the-badge&logo=amazons3&logoColor=white"><br>
<img src="https://img.shields.io/badge/amazonec2-FF9900?style=for-the-badge&logo=amazonec2&logoColor=white"><br>
<img src="https://img.shields.io/badge/opencv-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"><br>
<img src="https://img.shields.io/badge/FAISS-000000?style=for-the-badge&logo=&logoColor=white"><br>
## 유사도 검사 과정
##### 1. S3 url을 받아 영상 다운로드
##### 2. 영상을 프레임 단위로 나누고 Res-Net50을 이용하여 특징 벡터 추출
##### 3. FAISS에 저장된 특징 벡터값과 cosine similarity 수행
###### 3.1. max_similarity가 1 이상이거나, average_similarity가 0.75 이상이고, max_similarity가 0.8이상인 경우, 같은 영상이 있다고 판단.
##### 4.1. 유사도 검사를 통과하였다면, S3에서 다운받은 영상 데이터 삭제/ FAISS에 특징 벡터 값 저장 / SpringBoot 백엔드 서버로 결과 반환
##### 4.2. 유사도 검사를 실패하였다면, S3에서 다운받은 영상 데이터 삭제 및 S3 스토리지 삭제 / SpringBoot 백엔드 서버로 결과 반환
