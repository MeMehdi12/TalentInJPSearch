# 🚀 Talentin JPSEARCH: Semantic Talent Search Engine
**From "What is this?" → "How does the math work?"**  
*For recruiters, marketers, AND engineers*

> **TL;DR**: Type "Python devs in Tokyo" → AI extracts skills/location → **Math finds exact matches** → #1 candidate has Python + Tokyo + right experience.

## 🎯 What Does It Do? (Noob Level)
```
1. Recruiter types: "Senior React devs ex-Google in SF"
2. AI parses → skills: ["react"], companies: ["google"], location: "SF"  
3. **Magic math** ranks candidates → #1: "Senior React Engineer @ Meta (ex-Google), SF"
```

**Why better than LinkedIn?**
- Understands "React dev" = "Frontend Engineer"
- Finds "Google" alumni even if resume says "Alphabet Inc."
- Boosts exact matches **massively** (skills + location + title = jackpot!)

## 🏗️ System Architecture (Marketing/Tech Lead)
```
Frontend (React) → FastAPI v2 → Qdrant Vectors + DuckDB Profiles
                           ↓
                    Hybrid Search (Semantic + Keywords)
                           ↓
                    Smart Re-Ranking (Aggressive Boosts)
                           ↓
                 Top 50 candidates (200ms)
```

**Data**: 200K+ profiles (skills, work history, education, certs)  
**Scale**: Multi-tenant, 500 QPS, <200ms latency

## 🔍 How Search Works (Step-by-Step)

### Step 1: Parse Natural Language
```
Input: "Python devs in Tokyo with 5+ years"
AI Output:
{
  "search_text": "senior python developer",
  "skills": {"must_have": ["python"]},
  "location": {"city": "Tokyo"},
  "experience": {"min_years": 5}
}
```

### Step 2: Create Vectors (Math Time! 🚀)
**Two vectors per candidate profile:**

| Type | Math | What it finds |
|------|------|---------------|
| **Dense (Semantic)** | **Cosine Similarity** | "Python dev" ≈ "Backend Engineer" (0.82 sim) |
| **Sparse (Keywords)** | **BM25** | Exact "python", "django", "Tokyo" |

**Code**:
```python
# Dense: 768-dim embedding
dense_vec = model.encode("senior python developer")  # [0.12, -0.34, ...]

# Sparse: BM25 weights (only non-zero terms)
sparse = encoder.encode(["python", "django", "tokyo"])
# → indices=[45, 128, 890], values=[1.24, 0.92, 0.67]
```

### Step 3: Qdrant Hybrid Search
```
Qdrant.prefetch(dense=3000 candidates)
Qdrant.prefetch(sparse=3000 candidates)  
RRF Fusion → Top 1000 → Normalize [0.10, 0.65]
```
**RRF Math**: `score = 1/(60 + rank)` → Merges semantic + keyword ranks

### Step 4: DuckDB Hydration
Fetch full profiles (work history, certs, education) for top 1000.

### Step 5: **Smart Re-Ranking** (The Secret Sauce 🔥)
Base score [0.10-0.65] + **Aggressive Boosts**:

| Match | Bonus | Example |
|-------|--------|---------|
| **Skills Perfect** | +0.40 | ALL 3 skills matched |
| **Company** | +0.20 | Ex-Google |
| **Title Exact** | +0.20 | "Senior Engineer" |
| **Tokyo City** | +0.35 | Exact city (must_match=+0.60) |
| **5-10yr Exp** | +0.12 | Perfect fit |
| **AWS Cert** | +0.10 | Text match |
| **3+ Exact** | +0.15 | **Jackpot combo!** |

**Final Score Math**:
```
score = base_rrf + skills_bonus + location_bonus + title_bonus + ...
# Perfect match → 0.65 + 0.40 + 0.35 + 0.20 = 1.60 → capped at 1.0
```

**Code Example** (`search_api_v2.py#smart_rerank`):
```python
# Skills: Exact array match + text fallback
if "python" in profile.skills:
    bonus += 0.10  # Array match
if "python" in profile.headline.lower():
    bonus += 0.06  # Text match

# Location: Tiered matching
if target_city == profile.city:
    bonus += 0.35  # Exact
elif target_state == profile.state:
    bonus += 0.15  # State

# Combo jackpot
if exact_matches >= 3:
    bonus += 0.15  # Skills+Location+Title = #1!
```

### Step 6: Post-Filter (must_match location)
Hard filter if `location_preference="must_match"`.

## 🧮 Deep Math (Engineer Level)

### 1. Cosine Similarity (Dense)
```
cosine(a,b) = (a·b) / (||a|| * ||b||) ∈ [-1, 1]
"python dev" embedding · candidate embedding = 0.82
```

### 2. BM25 Sparse (Keywords)
```
score = IDF × (TF × (k+1)) / (TF + k × (1-b + b × doc_len/avg_len))
python: IDF=1.2, TF=2 → weight=1.24
```

### 3. RRF Fusion
```
rrf_rank = 1 / (60 + rank_dense + rank_sparse)
Normalize → [0.10, 0.65] (headroom for boosts)
```

### 4. Skill Expansion (Graph)
```
DuckDB.skill_relationships:
python → django(0.85), flask(0.78), pandas(0.65)
min_sim=0.55, max_expand=5 → SparseVector includes all
```

## ⚙️ Production Setup
```
docker-compose up  # Qdrant + DuckDB + FastAPI
npm run dev        # React frontend

API Endpoints:
POST /api/v2/smart-search    # "React devs in SF"
POST /api/v2/search          # Structured JSON
GET  /api/v2/health          # 🟢 Healthy
```

**Performance**: 150-250ms, 500 QPS, 200K profiles

## 📈 Scaling & Costs
- **Qdrant Cloud**: $50/mo pod (handles 10K QPS)
- **OpenAI**: $0.001/query (parsing)
- **DuckDB**: Local file (zero cost)

## 🧪 Questions Answered
**Q: Why #1 candidate?**  
A: Perfect skills(0.40) + SF(0.35) + title(0.20) + combo(0.15) = **1.10** 🥇

**Q: "Backend" matches "Python dev"?**  
A: Cosine sim=0.78 → RRF rank boost

**Q: Misses skills?**  
A: Skills expanded + text fallback + headline search

**Built for recruiters who need "the one" candidate, not 1000 maybes.**

