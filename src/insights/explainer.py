import os
from groq import Groq   # pip install groq
# Swap for: from openai import OpenAI

class InsightGenerator:
    def __init__(self, api_key=None):
        self.client = Groq(
            api_key=api_key or os.environ["GROQ_API_KEY"])
        self.model  = "llama-3.3-70b-versatile"

    # ── Called from POST /predict ──────────────────────────────
    def generate_insight(self,
                         patient: dict,
                         prediction: dict,
                         context: list) -> str:
        ctx = "\n".join(
            f"- {r['text'][:280]}" for r in context)

        prompt = f"""You are a clinical AI assistant embedded in a
Digital Health Twin platform.

PATIENT DATA:
{patient}

ML PREDICTION:
  Risk level : {'HIGH' if prediction['prediction']==1 else 'LOW'}
  Confidence : {prediction['confidence']:.1%}

SIMILAR PATIENT CONTEXT (retrieved via RAG):
{ctx}

Write a 3-5 sentence clinical insight that:
1. States the predicted risk and top contributing factors
2. References patterns seen in similar patients
3. Recommends what a clinician should investigate next
Use precise but accessible language. No bullet points."""

        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"user","content":prompt}],
            max_tokens=300, temperature=0.3)
        return r.choices[0].message.content.strip()

    # ── Called from POST /ask ──────────────────────────────────
    def answer_question(self,
                        question: str,
                        patient: dict,
                        context: list) -> str:
        ctx = "\n".join(
            f"- {r['text'][:280]}" for r in context)

        prompt = f"""You are a clinical AI assistant.
Answer this question about the patient concisely (2-4 sentences).

QUESTION: {question}

PATIENT DATA:
{patient}

SIMILAR PATIENT CONTEXT:
{ctx}"""

        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"user","content":prompt}],
            max_tokens=200, temperature=0.3)
        return r.choices[0].message.content.strip()