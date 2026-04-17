"""
INSIGHIFY AI - Smart Study Assistant Backend
FastAPI + PyPDF + sentence-transformers + FAISS
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import io
import json
import random
import numpy as np

# PDF parsing
import pypdf

# Sentence embeddings + FAISS
from sentence_transformers import SentenceTransformer
import faiss

app = FastAPI(title="Insighify AI Backend", version="1.0.0")

# Allow all origins for local dev / hackathon
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Global State (in-memory for simplicity) -----
pdf_chunks: list[str] = []          # Text chunks from uploaded PDF
embeddings_index = None             # FAISS index
embed_model = None                  # Sentence transformer model
chunk_embeddings: np.ndarray = None # numpy array of embeddings

# ----- Load embedding model once at startup -----
@app.on_event("startup")
async def load_model():
    global embed_model
    print("Loading sentence-transformer model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded!")


# ----- Helper: split text into chunks -----
def split_into_chunks(text: str, chunk_size: int = 500) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks


# ----- Helper: search relevant chunks via FAISS -----
def search_chunks(query: str, top_k: int = 3) -> list[str]:
    if embeddings_index is None or not pdf_chunks:
        return []
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    distances, indices = embeddings_index.search(q_emb, min(top_k, len(pdf_chunks)))
    results = [pdf_chunks[i] for i in indices[0] if i < len(pdf_chunks)]
    return results


# ----- Helper: naive extractive summarizer -----
def extractive_summary(chunks: list[str], n: int = 8) -> list[str]:
    """Pick the most 'content-rich' sentences (by length heuristic)."""
    sentences = []
    for chunk in chunks:
        for sent in chunk.split(". "):
            sent = sent.strip()
            if len(sent) > 40:
                sentences.append(sent)
    # Sort by length (proxy for information density) and pick top n
    sentences.sort(key=len, reverse=True)
    return list(dict.fromkeys(sentences[:n]))  # deduplicate while preserving order


# ===== ENDPOINTS =====

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF, extract text, build FAISS index."""
    global pdf_chunks, embeddings_index, chunk_embeddings

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Read PDF bytes
    contents = await file.read()
    reader = pypdf.PdfReader(io.BytesIO(contents))

    # Extract all text
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() or ""

    if not full_text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from this PDF.")

    # Split into chunks
    pdf_chunks = split_into_chunks(full_text, chunk_size=150)

    # Build FAISS index
    embeddings = embed_model.encode(pdf_chunks, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    chunk_embeddings = embeddings

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner-product = cosine on normalized vecs
    index.add(embeddings)
    embeddings_index = index

    return {
        "message": "PDF uploaded and indexed successfully!",
        "filename": file.filename,
        "pages": len(reader.pages),
        "chunks": len(pdf_chunks),
    }


@app.get("/ask")
async def ask_question(q: str):
    """Answer a question using relevant PDF chunks."""
    if not pdf_chunks:
        raise HTTPException(status_code=400, detail="Please upload a PDF first.")

    relevant = search_chunks(q, top_k=3)
    context = " ".join(relevant)

    # Simple extractive answer: return the most relevant sentence
    sentences = [s.strip() for s in context.split(". ") if len(s.strip()) > 30]
    answer = sentences[0] if sentences else "I couldn't find a specific answer in the document."

    return {
        "question": q,
        "answer": answer,
        "sources": relevant,
    }


@app.get("/summary")
async def get_summary():
    """Return a bullet-point summary of the PDF."""
    if not pdf_chunks:
        raise HTTPException(status_code=400, detail="Please upload a PDF first.")

    bullets = extractive_summary(pdf_chunks, n=8)
    return {"summary": bullets}


@app.get("/questions")
async def get_important_questions():
    """Generate 5–10 important questions from the PDF content."""
    if not pdf_chunks:
        raise HTTPException(status_code=400, detail="Please upload a PDF first.")

    # Pick content-rich chunks and form questions
    sample_chunks = random.sample(pdf_chunks, min(10, len(pdf_chunks)))
    questions = []
    starters = [
        "What is", "Explain", "How does", "Why is", "Define",
        "What are the key aspects of", "Describe", "What role does",
        "Compare and contrast", "What are the implications of"
    ]
    seen = set()
    for chunk in sample_chunks:
        words = chunk.split()
        if len(words) < 5:
            continue
        # Grab a noun-phrase-like snippet (words 2-6)
        phrase = " ".join(words[2:6]).strip(".,;:")
        if phrase in seen or len(phrase) < 5:
            continue
        seen.add(phrase)
        starter = random.choice(starters)
        questions.append(f"{starter} {phrase}?")
        if len(questions) >= 8:
            break

    return {"questions": questions}


@app.get("/quiz")
async def get_quiz():
    """Generate 5 MCQs from the PDF content."""
    if not pdf_chunks:
        raise HTTPException(status_code=400, detail="Please upload a PDF first.")

    sample = random.sample(pdf_chunks, min(5, len(pdf_chunks)))
    mcqs = []
    for chunk in sample:
        sentences = [s.strip() for s in chunk.split(".") if len(s.strip()) > 30]
        if not sentences:
            continue
        sent = sentences[0]
        words = sent.split()
        if len(words) < 6:
            continue
        # Make a fill-in-the-blank style question
        blank_idx = random.randint(3, min(7, len(words) - 1))
        answer_word = words[blank_idx].strip(".,;:()")
        question_text = " ".join(words[:blank_idx]) + " _____ " + " ".join(words[blank_idx + 1:])

        # Generate fake options (other words from chunk as distractors)
        other_words = [w.strip(".,;:()") for w in words if len(w) > 4 and w != words[blank_idx]]
        distractors = random.sample(other_words, min(3, len(other_words)))
        options = distractors[:3] + [answer_word]
        random.shuffle(options)

        mcqs.append({
            "question": question_text + "?",
            "options": options,
            "answer": answer_word,
        })

    return {"quiz": mcqs}


@app.get("/flashcards")
async def get_flashcards():
    """Generate Q/A flashcards from PDF chunks."""
    if not pdf_chunks:
        raise HTTPException(status_code=400, detail="Please upload a PDF first.")

    sample = random.sample(pdf_chunks, min(6, len(pdf_chunks)))
    cards = []
    for chunk in sample:
        sentences = [s.strip() for s in chunk.split(".") if len(s.strip()) > 40]
        if not sentences:
            continue
        answer = sentences[0]
        words = answer.split()
        phrase = " ".join(words[:5]).strip(".,;:")
        question = f"What do you know about: {phrase}?"
        cards.append({"question": question, "answer": answer})

    return {"flashcards": cards}


@app.get("/topics")
async def get_topics():
    """Extract key topics and insights from the PDF."""
    if not pdf_chunks:
        raise HTTPException(status_code=400, detail="Please upload a PDF first.")

    # Frequency-based keyword extraction (simple but effective)
    from collections import Counter
    import re

    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "this", "that", "these", "those",
        "it", "its", "they", "them", "their", "we", "our", "you", "your",
        "as", "if", "not", "also", "which", "who", "what", "when", "where",
    }

    all_text = " ".join(pdf_chunks).lower()
    words = re.findall(r'\b[a-z]{4,}\b', all_text)
    filtered = [w for w in words if w not in stop_words]
    freq = Counter(filtered)
    top_topics = [word.title() for word, _ in freq.most_common(10)]

    # Simple study suggestions
    suggestions = [
        f"Focus on understanding '{top_topics[0]}' in depth" if top_topics else "Review core concepts",
        "Create a mind map connecting the key topics",
        "Practice with the generated quiz questions",
        "Use flashcards for terminology review",
        "Summarize each major section in your own words",
    ]

    # Weak areas (based on infrequent but important terms)
    weak = [word.title() for word, _ in freq.most_common()[20:25]] or ["No specific weak areas detected"]

    return {
        "topics": top_topics,
        "weak_areas": weak,
        "study_suggestions": suggestions,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "pdf_loaded": len(pdf_chunks) > 0}
