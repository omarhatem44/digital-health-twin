import faiss, numpy as np, pickle

def build_index(emb_path="data/processed/embeddings.pkl",
                idx_path="data/processed/faiss.index"):
    with open(emb_path,'rb') as f: data = pickle.load(f)

    embs = data['embeddings'].astype(np.float32)
    # Normalize → inner product == cosine similarity
    faiss.normalize_L2(embs)

    dim   = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    faiss.write_index(index, idx_path)
    print(f"FAISS index: {index.ntotal} vectors, dim={dim}")
    return index

def load_index(idx_path="data/processed/faiss.index"):
    return faiss.read_index(idx_path)

if __name__=="__main__":
    build_index()