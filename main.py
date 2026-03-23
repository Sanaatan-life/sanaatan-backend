import os
import traceback
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Sanaatan API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5

@app.get("/")
def health():
    return {"status": "Sanaatan API is running", "version": "1.0.0"}

@app.post("/ask")
def ask(request: QuestionRequest):
    try:
        # Step 1: Import and connect
        from pinecone import Pinecone
        from openai import OpenAI
        import anthropic

        OPENAI_KEY    = os.environ.get("OPENAI_API_KEY", "")
        PINECONE_KEY  = os.environ.get("PINECONE_API_KEY", "")
        PINECONE_HOST = os.environ.get("PINECONE_HOST", "").replace("https://", "").replace("http://", "").strip("/")
        ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

        # Log what we have (masked)
        print(f"OPENAI_KEY present: {bool(OPENAI_KEY)}")
        print(f"PINECONE_KEY present: {bool(PINECONE_KEY)}")
        print(f"PINECONE_HOST: {PINECONE_HOST[:30] if PINECONE_HOST else 'MISSING'}")
        print(f"ANTHROPIC_KEY present: {bool(ANTHROPIC_KEY)}")

        pc     = Pinecone(api_key=PINECONE_KEY)
        index  = pc.Index(host=PINECONE_HOST)
        client = OpenAI(api_key=OPENAI_KEY)
        claude = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        print("✅ All clients connected")

        # Step 2: Embed
        q_embedding = client.embeddings.create(
            input=[request.question],
            model="text-embedding-3-small"
        ).data[0].embedding
        print("✅ Embedding generated")

        # Step 3: Retrieve
        results = index.query(
            vector=q_embedding,
            top_k=request.top_k,
            include_metadata=True
        )
        print(f"✅ Retrieved {len(results.matches)} matches")

        # Step 4: Build context
        context = ""
        for match in results.matches:
            m = match.metadata
            context += f"\nReference: {match.id} "
            context += f"(Chapter {m.get('chapter','?')}, Verse {m.get('verse','?')})\n"
            context += f"Sanskrit: {m.get('sanskrit','')[:200]}\n"
            context += f"Meaning: {m.get('english','')}\n"

        # Step 5: Claude
        response = claude.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            system="""You are Sanaatan, a wise guide to Hindu scriptures.
Always cite the specific verse (Bhagavad Gita, Chapter X, Verse Y).
Start with the Sanskrit transliteration.
Give a clear practical explanation.
End with: 💭 For your contemplation: [one reflective question]""",
            messages=[{
                "role": "user",
                "content": f"Question: {request.question}\n\nVerses:\n{context}"
            }]
        )
        print("✅ Claude responded")

        return {
            "question": request.question,
            "answer": response.content[0].text,
            "status": "success"
        }

    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"❌ ERROR: {error_detail}")
        return {"error": str(e), "detail": error_detail, "status": "failed"}
