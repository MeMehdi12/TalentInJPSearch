# TalentIn JP Search

AI-powered candidate search platform for the Japanese job market. Combines semantic vector search (Qdrant) with structured filtering (DuckDB) and GPT-powered JD parsing.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI (Python) + Uvicorn |
| AI/Embeddings | sentence-transformers (`all-mpnet-base-v2`), OpenAI GPT |
| Vector DB | Qdrant Cloud |
| Structured DB | DuckDB |
| Frontend | React + Vite |
| Web Server | Nginx (reverse proxy + HTTPS) |
| Hosting | AWS EC2 + Let's Encrypt SSL |

## Project Structure

```
├── backend/                # FastAPI application
│   ├── search_api_v2.py    # Main API (entry point)
│   ├── search_service.py   # Search orchestration
│   ├── hybrid_search.py    # Qdrant hybrid search
│   ├── filter_service.py   # DuckDB filter queries
│   ├── ranking_service.py  # Candidate ranking engine
│   ├── openai_parser.py    # GPT JD parsing
│   ├── config.py           # Environment config
│   └── requirements.txt    # Python dependencies
├── frontend/               # React + Vite SPA
│   ├── src/
│   └── package.json
└── deployment/             # Server configuration
    ├── talentin-backend.service  # Systemd unit
    └── nginx-talentin.conf       # Nginx config
```

## Local Development

### Prerequisites
- Python 3.10+
- Node.js 18+
- A `.env` file (see `backend/.env.example`)

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn search_api_v2:app --reload --port 8001
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## Deployment

### Deploy backend changes
```bash
# On your server
cd /var/www/talentin
git pull origin main
sudo systemctl restart talentin-backend
```

### Deploy frontend changes
```bash
# Windows: rebuild and push
cd frontend
npm run build
git add .
git commit -m "Rebuild frontend"
git push

# On server: pull and done (Nginx serves static files directly)
cd /var/www/talentin && git pull origin main
```

## Environment Variables

Copy `backend/.env.example` and fill in your values. **Never commit `.env` to Git.**

Key variables:
- `QDRANT_URL` / `QDRANT_API_KEY` — Qdrant Cloud connection
- `DUCKDB_PATH` — Absolute path to the `.duckdb` file
- `OPENAI_API_KEY` — For JD parsing
- `SEARCH_ENV` — `local` or `production`
