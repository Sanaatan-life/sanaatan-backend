
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
from pinecone import Pinecone
from openai import OpenAI

# ── Initialise app ────────────────────────────────────────────────────────────
app = FastAPI(title="Sanaatan API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://app.sanaatan.life",
        "https://sanaatan.life",
        "https://sanaatan-app.pages.dev",
        "http://localhost:3000",
        "*"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ── Load API keys from environment ────────────────────────────────────────────
OPENAI_API_KEY    = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY  = os.environ["PINECONE_API_KEY"]
PINECONE_HOST     = os.environ["PINECONE_HOST"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

# ── Initialise clients ────────────────────────────────────────────────────────
pc     = Pinecone(api_key=PINECONE_API_KEY)
index  = pc.Index(host=PINECONE_HOST)
client = OpenAI(api_key=OPENAI_API_KEY)
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ── Request model ─────────────────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5

# ── Health check endpoint ─────────────────────────────────────────────────────
@app.get("/")
def health_check():
    return {"status": "Sanaatan API is running", "version": "1.0.0"}

# ── Main Q&A endpoint ─────────────────────────────────────────────────────────
@app.post("/ask")
def ask_sanaatan(request: QuestionRequest):
    try:
        # Step 1: Embed the question
        q_embedding = client.embeddings.create(
            input=[request.question],
            model="text-embedding-3-small"
        ).data[0].embedding

        # Step 2: Retrieve relevant verses
        results = index.query(
            vector=q_embedding,
            top_k=request.top_k,
            include_metadata=True
        )

        # Step 3: Build context
        context = ""
        sources = []
        for match in results.matches:
            m = match.metadata
            context += f"\n---\n"
            context += f"Reference: {match.id} "
            context += f"(Bhagavad Gita, Chapter {m.get('chapter','?')}, Verse {m.get('verse','?')})\n"
            context += f"Sanskrit: {m.get('sanskrit','')[:200]}\n"
            context += f"Meaning: {m.get('english','')}\n"
            sources.append({
                "id": match.id,
                "chapter": m.get("chapter"),
                "verse": m.get("verse"),
                "score": round(match.score, 3)
            })

        # Step 4: Claude synthesis
        system_prompt = """You are Sanaatan — a wise, compassionate guide to Hindu scriptures.
Answer questions using only the Bhagavad Gita verses provided.
Always cite the specific verse (e.g. Bhagavad Gita, Chapter 2, Verse 47).
Start with the Sanskrit transliteration of the most relevant verse.
Give a clear, practical explanation grounded in the verse.
End with: 💭 For your contemplation: [one reflective question]
Never fabricate or go beyond what the verses say."""

        response = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": f"Question: {request.question}\n\nRelevant verses:\n{context}"
            }]
        )

        return {
            "question": request.question,
            "answer": response.content[0].text,
            "sources": sources,
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
