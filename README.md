# NLP Katas

Learn NLP as a representation and modeling discipline — from raw text to transformers.

## Running the Python Backend

```bash
cd backend/python-nlp-katas
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Key endpoints:

- `GET /health` — health check
- `GET /api/tracks` — list available tracks
- `GET /api/tracks/python-nlp/katas` — list all katas
- `GET /api/tracks/python-nlp/katas/{phase}/{kata_id}/content` — get kata markdown
- `POST /api/execute` — run Python code (`{"code": "...", "kata_id": "..."}`)

## Running the Frontend

```bash
cd frontend
npm install
npm run dev
```

The app will be available at `http://localhost:3000`. API requests are proxied to the backend on port 8000.

## Project Structure

```
nlp-katas/
├── frontend/                    (SolidJS + Tailwind CSS v4)
├── backend/
│   └── python-nlp-katas/        (FastAPI + kata content)
│       ├── main.py
│       ├── app/
│       └── phase-0/
├── todo.md                      (global checklist)
└── CLAUDE.md                    (project instructions)
```
