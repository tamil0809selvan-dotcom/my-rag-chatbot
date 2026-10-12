
import os
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

client = genai.Client(api_key=API_KEY) if API_KEY else None

app = FastAPI(title="My RAG Chatbot API")

chunks = []
MAX_PDF_SIZE = 10 * 1024 * 1024
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 200


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"status": "success", "message": "RAG API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global chunks

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="PDF file mattum upload pannunga"
        )

    pdf_bytes = await file.read(MAX_PDF_SIZE + 1)

    if len(pdf_bytes) > MAX_PDF_SIZE:
        raise HTTPException(
            status_code=413,
            detail="PDF size 10 MB-ku keezha irukkanum"
        )

    try:
        reader = PdfReader(BytesIO(pdf_bytes))

        extracted_text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="PDF read panna mudiyala"
        ) from exc

    if not extracted_text.strip():
        raise HTTPException(
            status_code=400,
            detail="PDF-la readable text illa"
        )

    new_chunks = [
        extracted_text[i:i + CHUNK_SIZE]
        for i in range(
            0,
            len(extracted_text),
            CHUNK_SIZE - CHUNK_OVERLAP
        )
        if extracted_text[i:i + CHUNK_SIZE].strip()
    ]

    chunks = new_chunks

    return {
        "status": "success",
        "filename": file.filename,
        "chunks_created": len(chunks),
        "message": "PDF upload and processing successful"
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="First PDF upload pannunga"
        )

    if client is None:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY Render environment-la set pannunga"
        )

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question empty-ah irukku"
        )

    # Lightweight keyword-based retrieval; no embedding model or FAISS.
    stop_words = {
        "the", "is", "are", "was", "were", "what", "who",
        "when", "where", "why", "how", "a", "an", "and",
        "or", "of", "to", "in", "on", "for", "with", "please"
    }

    question_words = {
        word.lower().strip(".,?!:;()[]{}\"'")
        for word in question.split()
    } - stop_words

    scored_chunks = []

    for chunk in chunks:
        chunk_words = {
            word.lower().strip(".,?!:;()[]{}\"'")
            for word in chunk.split()
        }
        score = len(question_words & chunk_words)
        scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)

    selected = [
        chunk for score, chunk in scored_chunks[:4]
        if score > 0
    ]

    if not selected:
        selected = chunks[:2]

    context = "\n\n".join(selected)[:7000]

    prompt = f"""
Answer the user's question using only the PDF context below.
If the answer is not available in the context, say that clearly.
Do not invent facts.

PDF context:
{context}

Question:
{question}
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        answer = response.text or "Answer generate panna mudiyala."
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Gemini response fail aachu. API key, model name, "
                "quota check pannunga."
            )
        ) from exc

    return {
        "question": question,
        "answer": answer
    }
