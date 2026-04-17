# 🌈 INSIGHIFY AI – Smart Study Assistant

A hackathon-ready AI study assistant that transforms PDFs into summaries, quizzes, flashcards, and more.

---

## 📁 Folder Structure

```
insighify/
├── backend/
│   ├── main.py           # FastAPI application
│   └── requirements.txt  # Python dependencies
├── frontend/
│   └── index.html        # Full website (single file)
└── README.md
```

---

## ⚙️ Backend Setup

### 1. Create a virtual environment
```bash
cd backend
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ First run downloads the `all-MiniLM-L6-v2` model (~90MB). This happens automatically.

### 3. Start the backend
```bash
uvicorn main:app --reload --port 8000
```

The API will be available at: **http://localhost:8000**  
API docs (Swagger): **http://localhost:8000/docs**

---

## 🌐 Frontend Setup

No build step needed — it's a single HTML file!

Open `frontend/index.html` directly in your browser:
```bash
# Mac
open frontend/index.html

# Windows
start frontend/index.html

# Or use VS Code Live Server extension
```

> Make sure the backend is running on port 8000 before using the site.

---

## 🔗 API Endpoints

| Method | Endpoint       | Description                  |
|--------|---------------|------------------------------|
| POST   | `/upload`      | Upload PDF file              |
| GET    | `/ask?q=...`   | Ask a question               |
| GET    | `/summary`     | Get bullet-point summary     |
| GET    | `/questions`   | Get important questions      |
| GET    | `/quiz`        | Get MCQ quiz                 |
| GET    | `/flashcards`  | Get flashcards               |
| GET    | `/topics`      | Get key topics & insights    |
| GET    | `/health`      | Health check                 |

---

## 🚀 Deployment (Quick Options)

### Option A: Render (Free)
1. Push code to GitHub
2. Go to https://render.com → New Web Service
3. Set Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Update `API` variable in `index.html` to your Render URL

### Option B: Railway
1. `railway init` → `railway up`
2. Set environment: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Option C: Vercel (Frontend only)
- Deploy `index.html` to Vercel for the frontend
- Host backend separately on Render/Railway

---

## 🛠 Tech Stack

| Layer     | Tech                           |
|-----------|-------------------------------|
| Frontend  | HTML5, CSS3, Vanilla JS       |
| Backend   | Python, FastAPI               |
| PDF       | PyPDF                         |
| Embeddings| sentence-transformers (MiniLM)|
| Search    | FAISS (vector similarity)     |

---

## 💡 Notes

- PDF state is **in-memory** — uploading a new PDF replaces the old one
- Works best with text-based PDFs (not scanned images)
- For production: add persistent storage (SQLite / Postgres) and authentication
