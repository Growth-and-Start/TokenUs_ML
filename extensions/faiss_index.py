import os
import faiss
import numpy as np
from config import Config

class FAISSIndex:
    def __init__(self, dim=2048):
        self.index = faiss.IndexFlatIP(dim)

    def add_vectors(self, vectors):
        vectors = np.array(vectors).astype('float32')
        faiss.normalize_L2(vectors)
        self.index.add(vectors)

    def search(self, query_vector, top_k=5):
        query_vector = np.array(query_vector).astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, top_k)
        return indices, distances

# File save/load
def save_faiss_index(faiss_index):
    os.makedirs(os.path.dirname(Config.FAISS_INDEX_PATH), exist_ok=True)
    faiss.write_index(faiss_index.index, Config.FAISS_INDEX_PATH)

def load_faiss_index():
    index = FAISSIndex(dim=2048)
    if os.path.exists(Config.FAISS_INDEX_PATH):
        index.index = faiss.read_index(Config.FAISS_INDEX_PATH)
    return index
