import os
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

# Manual CORS — handles ALL responses including errors
@app.middleware("http")
async def add_cors(request: Request, call_next):
    if request.method == "OPTIONS":
        return JSONResponse(
            content={},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Max-Age": "86400",
            }
        )
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5

@app.get("/")
def health():
    return {"status": "Sanaatan API is running", "version": "1.0.0"}

@app.post("/ask")
def ask(request: QuestionRequest):
    try:
        from pinecone import Pinecone
        from openai import OpenAI
        import anthropic

        OPENAI_KEY    = os.environ.get("OPENAI_API_KEY", "")
        PINECONE_KEY  = os.environ.get("PINECONE_API_KEY", "")
        PINECONE_HOST = os.environ.get("PINECONE_HOST", "").replace("https://", "").strip("/")
        ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

        pc     = Pinecone(api_key=PINECONE_KEY)
        index  = pc.Index(host=PINECONE_HOST)
        client = OpenAI(api_key=OPENAI_KEY)
        claude = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

        q_embedding = client.embeddings.create(
            input=[request.question],
            model="text-embedding-3-small"
        ).data[0].embedding

        results = index.query(
            vector=q_embedding,
            top_k=request.top_k,
            include_metadata=True
        )

        context = ""
        for match in results.matches:
            m = match.metadata
            context += f"\nReference: {match.id} (Chapter {m.get('chapter','?')}, Verse {m.get('verse','?')})\n"
            context += f"Sanskrit: {m.get('sanskrit','')[:200]}\n"
            context += f"Meaning: {m.get('english','')}\n"

        response = claude.messages.create(
            model="claude-sonnet-4-20250514",
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

        return {"question": request.question, "answer": response.content[0].text, "status": "success"}

    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={"error": str(e), "detail": traceback.format_exc(), "status": "failed"},
            headers={"Access-Control-Allow-Origin": "*"}
        )
