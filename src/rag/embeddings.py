import pandas as pd, pickle, os
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

def build_patient_documents(df: pd.DataFrame) -> list:
    docs = []
    for _, r in df.iterrows():
        text = (
            f"Patient {r['patient_id']}: Age {r['age']}, "
            f"Gender {r['gender']}, BMI {r['bmi']:.1f}, "
            f"BP {r['systolic_bp']}/{r['diastolic_bp']}, "
            f"HR {r['heart_rate']} bpm, "
            f"Glucose {r['glucose']:.1f}, Cholesterol {r['cholesterol']:.1f}. "
            f"Activity: {r['activity_level']}, "
            f"Smoker: {'Yes' if r['smoking'] else 'No'}, "
            f"Diabetes: {'Yes' if r['diabetes'] else 'No'}. "
            f"Notes: {r['notes']} "
            f"Risk classification: {'HIGH' if r['high_risk'] else 'LOW'}."
        )
        docs.append({
            'patient_id': r['patient_id'],
            'text': text,
            'high_risk': int(r['high_risk'])
        })
    return docs

def embed_documents(docs: list,
                    save_path="data/processed/embeddings.pkl"):
    model = SentenceTransformer(MODEL_NAME)
    texts = [d['text'] for d in docs]
    embeddings = model.encode(
        texts, show_progress_bar=True, batch_size=32)

    payload = {'docs':docs,'embeddings':embeddings,
               'model_name':MODEL_NAME}
    os.makedirs("data/processed",exist_ok=True)
    with open(save_path,'wb') as f: pickle.dump(payload,f)
    print(f"Embedded {len(docs)} docs  shape={embeddings.shape}")
    return payload

if __name__=="__main__":
    df = pd.read_csv("data/processed/patients_processed.csv")
    docs = build_patient_documents(df)
    embed_documents(docs)