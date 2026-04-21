import os
import traceback
import hmac
import hashlib
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from collections import defaultdict
import time
from datetime import date
from pinecone import Pinecone
from openai import OpenAI
import anthropic
import httpx

# ── Clients ────────────────────────────────────────────────────────────────────
OPENAI_KEY             = os.environ.get("OPENAI_API_KEY", "")
PINECONE_KEY           = os.environ.get("PINECONE_API_KEY", "")
PINECONE_HOST          = os.environ.get("PINECONE_HOST", "").replace("https://", "").strip("/")
ANTHROPIC_KEY          = os.environ.get("ANTHROPIC_API_KEY", "")
RAZORPAY_WEBHOOK_SECRET= os.environ.get("RAZORPAY_WEBHOOK_SECRET", "")
SUPABASE_URL           = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY   = os.environ.get("SUPABASE_SERVICE_KEY", "")

pc     = Pinecone(api_key=PINECONE_KEY)
index  = pc.Index(host=PINECONE_HOST)
client = OpenAI(api_key=OPENAI_KEY)
claude = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

# ── Plan ID → Tier mapping ─────────────────────────────────────────────────────
PLAN_TIER_MAP = {
    "plan_DAILY_ID_HERE":     "daily",
    "plan_UNLIMITED_ID_HERE": "unlimited",
    "plan_ANNUAL_ID_HERE":    "annual",
}

# ── Daily query limits per tier ────────────────────────────────────────────────
TIER_LIMITS = {
    "free":      3,
    "starter":   15,   # one-time pack, tracked same way
    "daily":     10,
    "unlimited": 999999,
    "annual":    999999,
}

# ── Model routing ──────────────────────────────────────────────────────────────
SONNET = "claude-sonnet-4-20250514"
HAIKU  = "claude-haiku-4-5-20251001"

def get_model(tier: str) -> str:
    return SONNET if tier.lower() in {"unlimited", "annual"} else HAIKU

# ── In-memory rate limiting (Cloudflare WAF handles real limiting) ─────────────
request_counts = defaultdict(list)
RATE_LIMIT = 10
WINDOW_SECS = 60

app = FastAPI()

# ── CORS middleware ────────────────────────────────────────────────────────────
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

# ── Request model ──────────────────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5
    user_id: str = ""   # Supabase UUID — empty string for anonymous users

# ── Supabase helpers ───────────────────────────────────────────────────────────
async def get_user_profile(user_id: str) -> dict:
    """Fetch tier and daily_query_count from Supabase for a logged-in user."""
    async with httpx.AsyncClient() as hc:
        resp = await hc.get(
            f"{SUPABASE_URL}/rest/v1/user_profiles",
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            },
            params={"id": f"eq.{user_id}", "select": "tier,daily_query_count,last_query_date"}
        )
        rows = resp.json()
        if rows:
            return rows[0]
        return None

async def check_and_increment_query(user_id: str) -> dict:
    """
    Check if user can ask a question. If yes, increment counter.
    Returns: {"allowed": bool, "tier": str, "remaining": int}
    """
    profile = await get_user_profile(user_id)
    if not profile:
        # User profile doesn't exist yet — create it
        await create_user_profile(user_id)
        return {"allowed": True, "tier": "free", "remaining": 2}

    tier = profile.get("tier", "free")
    limit = TIER_LIMITS.get(tier, 3)
    today = str(date.today())
    last_date = profile.get("last_query_date", "")
    count = profile.get("daily_query_count", 0)

    # Reset count if it's a new day
    if last_date != today:
        count = 0

    if count >= limit:
        return {"allowed": False, "tier": tier, "remaining": 0}

    # Increment
    new_count = count + 1
    async with httpx.AsyncClient() as hc:
        await hc.patch(
            f"{SUPABASE_URL}/rest/v1/user_profiles",
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal"
            },
            params={"id": f"eq.{user_id}"},
            json={"daily_query_count": new_count, "last_query_date": today}
        )

    remaining = limit - new_count
    return {"allowed": True, "tier": tier, "remaining": remaining}

async def create_user_profile(user_id: str):
    """Create a new user profile with free tier defaults."""
    async with httpx.AsyncClient() as hc:
        await hc.post(
            f"{SUPABASE_URL}/rest/v1/user_profiles",
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal"
            },
            json={
                "id": user_id,
                "tier": "free",
                "daily_query_count": 1,
                "last_query_date": str(date.today())
            }
        )

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "Sanaatan API is running", "version": "2.0.0"}

@app.get("/debug-ip")
async def debug_ip(request: Request):
    forwarded_for = request.headers.get("X-Forwarded-For") or ""
    client_ip = forwarded_for.split(",")[0].strip() or "unknown"
    return {
        "CF-Connecting-IP": request.headers.get("CF-Connecting-IP"),
        "X-Forwarded-For": request.headers.get("X-Forwarded-For"),
        "client_ip_resolved": client_ip
    }

# ── Razorpay Webhook ───────────────────────────────────────────────────────────
@app.post("/webhook/razorpay")
async def razorpay_webhook(req: Request):
    body = await req.body()
    sig = req.headers.get("X-Razorpay-Signature", "")
    expected = hmac.new(
        RAZORPAY_WEBHOOK_SECRET.encode(),
        body,
        hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(expected, sig):
        return JSONResponse(status_code=400, content={"error": "Invalid signature"})

    payload = json.loads(body)
    event   = payload.get("event", "")

    if event == "payment.captured":
        payment = payload.get("payload", {}).get("payment", {}).get("entity", {})
        email   = payment.get("email", "")
        plan_id = payment.get("description", "")
        tier    = PLAN_TIER_MAP.get(plan_id, "daily")
        if email:
            async with httpx.AsyncClient() as hc:
                await hc.patch(
                    f"{SUPABASE_URL}/rest/v1/user_profiles",
                    headers={
                        "apikey": SUPABASE_SERVICE_KEY,
                        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                        "Content-Type": "application/json",
                        "Prefer": "return=minimal"
                    },
                    params={"email": f"eq.{email}"},
                    json={"tier": tier}
                )

    return JSONResponse(status_code=200, content={"status": "ok"})

# ── Main ask endpoint ──────────────────────────────────────────────────────────
@app.post("/ask")
async def ask(question_request: QuestionRequest, request: Request):
    # ── IP-based rate limit (belt + suspenders with Cloudflare WAF) ──
    forwarded_for = request.headers.get("X-Forwarded-For") or ""
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

    # ── User query limit check ──
    user_id = question_request.user_id.strip()

    if user_id:
        # Logged-in user — enforce server-side daily limit
        result = await check_and_increment_query(user_id)
        if not result["allowed"]:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "You have reached your daily question limit. Upgrade for more questions.",
                    "status": "limit_reached",
                    "tier": result["tier"],
                    "remaining": 0
                },
                headers={"Access-Control-Allow-Origin": "*"}
            )
        tier = result["tier"]
        remaining = result["remaining"]
    else:
        # Anonymous user — frontend handles limit via localStorage
        # Backend trusts frontend for anon users (no auth = no server enforcement)
        tier = "free"
        remaining = None

    try:
        model = get_model(tier)

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
            messages=[{
                "role": "user",
                "content": f"Question: {question_request.question}\n\nVerses:\n{context}"
            }]
        )

        resp_body = {
            "question": question_request.question,
            "answer": response.content[0].text,
            "status": "success",
            "model_used": model,
            "tier": tier,
        }
        if remaining is not None:
            resp_body["remaining"] = remaining

        return resp_body

    except Exception as e:
        error_msg = str(e)
        if "overloaded" in error_msg.lower() or "529" in error_msg:
            user_message = "The scriptures are busy at this moment. Please try your question again in a few seconds. 🙏"
        else:
            user_message = "Something went wrong. Please try again. 🙏"
        return JSONResponse(
            status_code=500,
            content={"error": user_message, "status": "failed"},
            headers={"Access-Control-Allow-Origin": "*"}
        )