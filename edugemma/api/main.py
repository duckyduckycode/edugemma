"""
EduGemma — Offline AI STEM Tutor Backend
Powered by fine-tuned Gemma 4 via Ollama
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import httpx
import json
import sqlite3
import os

app = FastAPI(title="EduGemma", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("GEMMA_MODEL", "gemma4:e4b")
DB_PATH = os.getenv("DB_PATH", "data/progress.db")


# --- Database Setup ---
def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS students (
            id TEXT PRIMARY KEY,
            name TEXT,
            level INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            topic TEXT,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            difficulty INTEGER DEFAULT 3
        );
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            role TEXT,
            content TEXT,
            image_path TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS quiz_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            topic TEXT,
            question TEXT,
            student_answer TEXT,
            correct_answer TEXT,
            is_correct BOOLEAN,
            difficulty INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.close()

init_db()


# --- Models ---
class ChatRequest(BaseModel):
    message: str
    student_id: Optional[str] = "default"
    session_id: Optional[int] = None
    difficulty: Optional[int] = 3  # 1-5 scale

class QuizRequest(BaseModel):
    topic: str
    student_id: Optional[str] = "default"
    num_questions: Optional[int] = 3
    difficulty: Optional[int] = 3


# --- Ollama Integration ---
SYSTEM_PROMPT = """You are EduGemma, an adaptive STEM tutor. Your role:
- Explain concepts step-by-step at the student's level
- Adjust difficulty based on their understanding
- Encourage students when they struggle
- Use concrete examples and analogies
- When shown an image of a problem, identify it and help solve it
- Respond in the student's preferred language if specified
- Never just give the answer to a homework problem — guide them to find it

Difficulty levels: 1=beginner, 2=elementary, 3=intermediate, 4=advanced, 5=expert"""


async def query_gemma(messages: list, stream: bool = False) -> str:
    """Query Gemma 4 via Ollama API"""
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL,
                "messages": messages,
                "stream": stream,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 2048,
                }
            }
        )
        response.raise_for_status()
        result = response.json()
        return result.get("message", {}).get("content", "")


# --- Endpoints ---
@app.get("/")
async def root():
    return {"name": "EduGemma", "version": "0.1.0", "model": MODEL}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Main tutoring chat endpoint"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + f"\nCurrent difficulty: {request.difficulty}/5"},
        {"role": "user", "content": request.message}
    ]
    
    # Query Gemma
    try:
        reply = await query_gemma(messages)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {e}")
    
    # Log interaction
    conn = sqlite3.connect(DB_PATH)
    if request.session_id:
        conn.execute(
            "INSERT INTO interactions (session_id, role, content) VALUES (?, ?, ?)",
            (request.session_id, "user", request.message)
        )
        conn.execute(
            "INSERT INTO interactions (session_id, role, content) VALUES (?, ?, ?)",
            (request.session_id, "assistant", reply)
        )
    conn.close()
    
    return {"reply": reply, "difficulty": request.difficulty}


@app.post("/api/chat/image")
async def chat_with_image(
    message: str = "",
    image: UploadFile = File(...),
    student_id: str = "default",
    difficulty: int = 3
):
    """Chat with image upload (textbook problem, handwritten work, etc.)"""
    # Save image
    img_path = f"data/uploads/{image.filename}"
    os.makedirs("data/uploads", exist_ok=True)
    with open(img_path, "wb") as f:
        f.write(await image.read())
    
    # For now, use text-only with image description prompt
    # Full multimodal support requires Ollama vision model
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + f"\nCurrent difficulty: {difficulty}/5"},
        {"role": "user", "content": f"[Student uploaded an image of a problem]\n{message}\n\nPlease help me understand and solve this problem step by step."}
    ]
    
    try:
        reply = await query_gemma(messages)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {e}")
    
    return {"reply": reply, "difficulty": difficulty}


@app.post("/api/quiz")
async def generate_quiz(request: QuizRequest):
    """Generate adaptive quiz questions"""
    prompt = f"""Generate {request.num_questions} quiz questions about {request.topic} at difficulty level {request.difficulty}/5.
    
Format as JSON array:
[{{"question": "...", "options": ["A) ...", "B) ...", "C) ...", "D) ..."], "answer": "A", "explanation": "..."}}]

Make questions progressively challenging. Difficulty {request.difficulty} means {"basic recall" if request.difficulty <= 2 else "application and analysis" if request.difficulty <= 3 else "synthesis and evaluation"}."""

    messages = [
        {"role": "system", "content": "You are a quiz generator. Output valid JSON only."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        reply = await query_gemma(messages)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {e}")
    
    # Try to parse JSON from response
    try:
        # Extract JSON from potential markdown code blocks
        import re
        json_match = re.search(r'\[.*\]', reply, re.DOTALL)
        if json_match:
            questions = json.loads(json_match.group())
        else:
            questions = [{"raw": reply}]
    except json.JSONDecodeError:
        questions = [{"raw": reply}]
    
    return {"questions": questions, "topic": request.topic, "difficulty": request.difficulty}


@app.get("/api/progress/{student_id}")
async def get_progress(student_id: str):
    """Get student progress and stats"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    # Quiz stats
    quizzes = conn.execute(
        "SELECT topic, COUNT(*) as total, SUM(is_correct) as correct, AVG(difficulty) as avg_diff FROM quiz_results WHERE student_id = ? GROUP BY topic",
        (student_id,)
    ).fetchall()
    
    # Recent sessions
    sessions = conn.execute(
        "SELECT * FROM sessions WHERE student_id = ? ORDER BY started_at DESC LIMIT 10",
        (student_id,)
    ).fetchall()
    
    conn.close()
    
    return {
        "student_id": student_id,
        "quiz_stats": [dict(q) for q in quizzes],
        "recent_sessions": [dict(s) for s in sessions]
    }


@app.get("/health")
async def health():
    """Check if Ollama is running"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags")
            models = resp.json().get("models", [])
            has_gemma = any("gemma" in m.get("name", "").lower() for m in models)
            return {"ollama": "running", "gemma_available": has_gemma, "models": [m["name"] for m in models]}
    except:
        return {"ollama": "not running", "gemma_available": False}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
