import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

class HealthRAGRetriever:
    def __init__(self,
                 emb_path="data/processed/embeddings.pkl",
                 idx_path="data/processed/faiss.index"):
        with open(emb_path,'rb') as f:
            data = pickle.load(f)
        self.docs  = data['docs']
        self.model = SentenceTransformer(data['model_name'])
        self.index = faiss.read_index(idx_path)

    def retrieve(self, query: str, top_k: int = 3) -> list:
        q = self.model.encode([query],
                show_progress_bar=False).astype(np.float32)
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, top_k)
        return [
            {
                'patient_id': self.docs[i]['patient_id'],
                'text':       self.docs[i]['text'],
                'score':      float(s)
            }
            for s, i in zip(scores[0], indices[0])
        ]

if __name__=="__main__":
    r = HealthRAGRetriever()
    hits = r.retrieve("high cardiovascular risk elevated BMI hypertension")
    for h in hits:
        print(f"\n{h['patient_id']}  score={h['score']:.4f}")
        print(h['text'][:180])