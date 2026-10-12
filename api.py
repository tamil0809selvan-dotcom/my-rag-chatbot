from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
from io import BytesIO
import numpy as np
import faiss

from app import (
    load_embedding_model,
    ask_gemini_with_retry,
)

app = FastAPI(title="My RAG Chatbot API")

model = load_embedding_model()
chunks = []
index = None


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
    global chunks, index

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF file mattum upload pannunga")

    pdf_bytes = await file.read()

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="PDF read panna mudiyala") from exc

    if not text.strip():
        raise HTTPException(status_code=400, detail="PDF-la readable text illa")

    chunks = [
        text[i:i + 1000]
        for i in range(0, len(text), 850)
        if text[i:i + 1000].strip()
    ]

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return {
        "status": "success",
        "filename": file.filename,
        "chunks_created": len(chunks),
        "message": "PDF upload and processing successful",
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    if index is None or not chunks:
        raise HTTPException(
            status_code=400,
            detail="First PDF upload pannunga",
        )

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question empty-ah irukku")

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_embedding = np.asarray(query_embedding, dtype=np.float32)

    scores, indices = index.search(
        query_embedding,
        min(4, len(chunks)),
    )

    context = "\n\n".join(
        chunks[i] for i in indices[0] if i >= 0
    )

    prompt = f"""
Answer the question using the PDF context below.
If the answer is not in the context, say so clearly.

PDF context:
{context}

Question:
{question}
"""

    try:
        answer = ask_gemini_with_retry(prompt)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Gemini answer generate panna mudiyala. API key/config check pannunga.",
        ) from exc

    return {"question": question, "answer": str(answer)}