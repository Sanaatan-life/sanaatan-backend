import os
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from collections import defaultdict
import time
from pinecone import Pinecone
from openai import OpenAI
import anthropic

# ── Clients (initialized once at startup) ─────────────────────────────────────
OPENAI_KEY    = os.environ.get("OPENAI_API_KEY", "")
PINECONE_KEY  = os.environ.get("PINECONE_API_KEY", "")
PINECONE_HOST = os.environ.get("PINECONE_HOST", "").replace("https://", "").strip("/")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

pc     = Pinecone(api_key=PINECONE_KEY)
index  = pc.Index(host=PINECONE_HOST)
client = OpenAI(api_key=OPENAI_KEY)
claude = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

# ── Model routing ──────────────────────────────────────────────────────────────
SONNET = "claude-sonnet-4-20250514"
HAIKU  = "claude-haiku-4-5-20251001"
SONNET_TIERS = {"unlimited", "annual", "free", "starter", "daily"}

def get_model(tier: str) -> str:
    return SONNET if tier.lower() in SONNET_TIERS else HAIKU

# ── Rate limiting ──────────────────────────────────────────────────────────────
request_counts = defaultdict(list)
RATE_LIMIT = 10
WINDOW_SECS = 60

app = FastAPI()

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
    response.headers["Access-Control-Expose-Headers"] = "*"
    return response

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5
    tier: str = "free"

@app.get("/")
def health():
    return {"status": "Sanaatan API is running", "version": "1.0.0"}

@app.get("/debug-ip")
async def debug_ip(req: Request):
    forwarded_for = req.headers.get("X-Forwarded-For") or ""
    client_ip = forwarded_for.split(",")[0].strip() or "unknown"
    return {
        "CF-Connecting-IP": req.headers.get("CF-Connecting-IP"),
        "X-Forwarded-For": req.headers.get("X-Forwarded-For"),
        "client_ip_resolved": client_ip
    }

@app.post("/ask")
async def ask(question_request: QuestionRequest, req: Request):
    forwarded_for = req.headers.get("X-Forwarded-For") or ""
    client_ip = forwarded_for.split(",")[0].strip() or "unknown"
    now = time.time()
    request_counts[client_ip] = [t for t in request_counts[client_ip] if now - t < WINDOW_SECS]
    if len(request_counts[client_ip]) >= RATE_LIMIT:
        return JSONResponse(
            status_code=429,
            content={"error": "Too many requests. Please wait a moment.", "status": "rate_limited"},
            headers={"Access-Control-Allow-Origin": "*"}
        )
    request_counts[client_ip].append(now)
    try:
        model = get_model(question_request.tier)

        q_embedding = client.embeddings.create(
            input=[question_request.question],
            model="text-embedding-3-small"
        ).data[0].embedding

        results = index.query(
            vector=q_embedding,
            top_k=question_request.top_k,
            include_metadata=True
        )

        context = ""
        for match in results.matches:
            m = match.metadata
            context += f"\nReference: {match.id} (Chapter {m.get('chapter','?')}, Verse {m.get('verse','?')})\n"
            context += f"Sanskrit: {m.get('sanskrit','')[:200]}\n"
            context += f"Meaning: {m.get('english','')}\n"

        response = claude.messages.create(
            model=model,
            max_tokens=1000,
system="""You are Sanaatan, a wise guide to Hindu scriptures including the Bhagavad Gita and the Upanishads.

If a question is not related to spirituality, dharma, life guidance, philosophy, or Hindu scriptures, respond with exactly:
"I am here to share wisdom from the Hindu scriptures. Please ask me something about dharma, life, relationships, purpose, or spiritual growth and I will find the answer in the Gita or Upanishads for you. 🙏"

For all relevant questions, structure every response exactly like this:

1. A clear, direct answer in 4-6 sentences in plain English. Be warm, wise, and substantive — not a one-liner but not an essay.

2. Then on a new line: "The scriptures say:"

3. For each of 2-3 supporting verses, use this exact citation format:
[Scripture Name], Chapter [X], Verse [Y]
[Devanagari Sanskrit]
[One clear sentence explaining what this verse means and why it is relevant]

4. End with: 💭 For your contemplation: [one reflective question]

Citation format must always be: "Bhagavad Gita, Chapter 2, Verse 47" or "Katha Upanishad, Chapter 1, Verse 3" — never abbreviate or vary this format.

No markdown headers. No bullet points. Wisdom should feel like a conversation with a knowledgeable guide.""",
        return {
            "question": question_request.question,
            "answer": response.content[0].text,
            "status": "success",
            "model_used": model
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "detail": traceback.format_exc(), "status": "failed"},
            headers={"Access-Control-Allow-Origin": "*"}
        )
