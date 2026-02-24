"""
Smart Query Preprocessor - Handle ANY Human Input
==================================================

This module makes the search SMART. People will search with:
- Typos: "pythin developer in SF"
- Slang: "ML eng at FAANG"
- Abbreviations: "AWS k8s exp preferred"
- Japanese: "東京のPythonエンジニア"
- Messy text: "people who worked at google or meta, know react, in bay area"

Our job: Understand ALL of it and return exact matches.
"""

import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from normalizers import (
    COMPANY_ALIASES, CITY_ALIASES, STATE_ALIASES, COUNTRY_ALIASES, SKILL_ALIASES,
    normalize_text, fuzzy_match, levenshtein_distance
)


# =============================================================================
# EXPANDED SKILL CATEGORIES - Related skills that should match together
# =============================================================================

SKILL_CATEGORIES = {
    # Programming Languages
    "python_ecosystem": ["python", "django", "flask", "fastapi", "pandas", "numpy", "celery", "pytest"],
    "javascript_ecosystem": ["javascript", "typescript", "react", "vue", "angular", "node", "express", "nextjs"],
    "java_ecosystem": ["java", "spring", "spring boot", "maven", "gradle", "hibernate", "jvm"],
    "dotnet_ecosystem": ["c#", ".net", "asp.net", "entity framework", "blazor"],
    "go_ecosystem": ["golang", "go", "gin", "echo"],
    "ruby_ecosystem": ["ruby", "rails", "ruby on rails", "sinatra"],
    "rust_ecosystem": ["rust", "tokio", "actix"],
    
    # Frontend
    "frontend": ["frontend", "front end", "react", "vue", "angular", "css", "html", "javascript", "typescript", "ui", "ux"],
    "react_stack": ["react", "redux", "nextjs", "gatsby", "react native"],
    "vue_stack": ["vue", "vuex", "nuxt", "vuetify"],
    
    # Backend
    "backend": ["backend", "back end", "api", "rest", "graphql", "microservices", "server"],
    "api_development": ["rest", "restful", "graphql", "grpc", "api design", "swagger", "openapi"],
    
    # Data
    "data_engineering": ["data engineering", "etl", "data pipeline", "airflow", "spark", "kafka", "hadoop"],
    "data_science": ["data science", "machine learning", "statistics", "pandas", "numpy", "jupyter", "r"],
    "machine_learning": ["machine learning", "ml", "deep learning", "tensorflow", "pytorch", "scikit-learn", "xgboost"],
    "ai": ["ai", "artificial intelligence", "machine learning", "deep learning", "nlp", "computer vision", "llm"],
    "nlp": ["nlp", "natural language processing", "bert", "gpt", "transformers", "spacy", "huggingface"],
    "computer_vision": ["computer vision", "cv", "opencv", "yolo", "image processing", "cnn"],
    
    # Databases
    "sql_databases": ["sql", "postgresql", "postgres", "mysql", "sqlite", "oracle", "sql server"],
    "nosql_databases": ["nosql", "mongodb", "dynamodb", "cassandra", "couchbase", "redis"],
    "databases": ["database", "db", "sql", "nosql", "postgresql", "mysql", "mongodb", "redis"],
    
    # Cloud
    "aws": ["aws", "amazon web services", "ec2", "s3", "lambda", "rds", "dynamodb", "ecs", "eks"],
    "gcp": ["gcp", "google cloud", "bigquery", "cloud functions", "gke", "cloud run"],
    "azure": ["azure", "microsoft azure", "azure functions", "azure devops", "cosmos db"],
    "cloud": ["cloud", "aws", "gcp", "azure", "cloud computing", "serverless"],
    
    # DevOps
    "devops": ["devops", "cicd", "docker", "kubernetes", "terraform", "ansible", "jenkins", "gitlab"],
    "containers": ["docker", "kubernetes", "k8s", "container", "containerization", "ecs", "eks", "gke"],
    "infrastructure": ["infrastructure", "terraform", "ansible", "cloudformation", "iac", "infra"],
    "cicd": ["cicd", "ci/cd", "jenkins", "github actions", "gitlab ci", "circleci", "travis"],
    
    # Mobile
    "mobile": ["mobile", "ios", "android", "react native", "flutter", "swift", "kotlin"],
    "ios": ["ios", "swift", "objective-c", "xcode", "cocoapods"],
    "android": ["android", "kotlin", "java", "android studio", "gradle"],
    
    # Security
    "security": ["security", "cybersecurity", "infosec", "penetration testing", "encryption", "oauth"],
    
    # Marketing & Social
    "marketing": ["marketing", "digital marketing", "content marketing", "growth marketing", "performance marketing"],
    "social media": ["social media", "social media marketing", "smm", "instagram", "facebook", "twitter", "linkedin"],
    "seo": ["seo", "search engine optimization", "sem", "google analytics", "search marketing"],
    "content": ["content", "content marketing", "content strategy", "copywriting", "blogging"],
}

# =============================================================================
# TITLE PATTERNS - What people search for
# =============================================================================

TITLE_PATTERNS = {
    # Generic Engineer/Developer (MUST be first to catch broad searches)
    "software engineer": ["software engineer", "swe", "software dev", "software developer", "sw engineer", "programmer", "engineer", "developer", "coder"],
    
    # AI/ML specific
    "ai engineer": ["ai engineer", "ai developer", "artificial intelligence engineer", "artificial intelligence developer", "ai specialist", "ai ml engineer"],
    "ml engineer": ["ml engineer", "machine learning engineer", "machine learning developer", "ml developer", "deep learning engineer", "ml eng"],
    "data scientist": ["data scientist", "data science", "ds", "applied scientist", "research scientist", "ai researcher"],
    
    # Backend/Frontend/Full-stack
    "backend engineer": ["backend engineer", "backend developer", "back end", "backend dev", "server side", "backend", "backend software engineer"],
    "frontend engineer": ["frontend engineer", "frontend developer", "front end", "frontend dev", "ui developer", "ui engineer", "frontend", "frontend software engineer"],
    "fullstack engineer": ["fullstack", "full stack", "full-stack", "fullstack engineer", "fullstack developer", "full stack developer"],
    
    # Infrastructure & DevOps
    "devops engineer": ["devops", "devops engineer", "sre", "site reliability", "platform engineer", "infra engineer", "infrastructure engineer"],
    "data engineer": ["data engineer", "data engineering", "de", "etl engineer", "data pipeline", "data platform engineer"],
    
    # Mobile & QA
    "mobile engineer": ["mobile engineer", "mobile developer", "mobile dev", "ios developer", "android developer", "app developer", "mobile software engineer"],
    "qa engineer": ["qa", "qa engineer", "qe", "test engineer", "sdet", "quality assurance", "software test engineer"],
    
    # Security
    "security engineer": ["security engineer", "infosec", "cybersecurity", "security analyst", "penetration tester", "infosec engineer"],
    
    # Manager/Lead variations  
    "engineering manager": ["engineering manager", "em", "eng manager", "engineering lead"],
    "tech lead": ["tech lead", "tl", "technical lead", "team lead", "lead engineer"],
    "staff engineer": ["staff engineer", "senior staff", "principal engineer", "distinguished engineer"],
    "architect": ["architect", "software architect", "solutions architect", "system architect"],
    
    # Non-engineer
    "product manager": ["product manager", "pm", "product owner", "po", "apm"],
    "designer": ["designer", "ux designer", "ui designer", "product designer", "ux/ui"],
    "analyst": ["analyst", "business analyst", "data analyst", "systems analyst"],
    
    # Marketing & Sales
    "marketing manager": ["marketing manager", "marketing lead", "marketing director", "head of marketing"],
    "marketing": ["marketing", "marketer", "marketing specialist", "marketing coordinator"],
    "digital marketing": ["digital marketing", "digital marketer", "online marketing", "performance marketing"],
    "social media manager": ["social media manager", "social media", "community manager", "social media coordinator"],
    "content marketing": ["content marketing", "content manager", "content strategist"],
    "sales manager": ["sales manager", "sales lead", "sales director", "head of sales"],
    "account manager": ["account manager", "account executive", "client success", "customer success"],
    
    # HR & Operations
    "recruiter": ["recruiter", "talent acquisition", "hiring manager", "recruitment"],
    "hr manager": ["hr manager", "human resources", "hr", "people ops", "people operations"],
    "operations manager": ["operations manager", "ops manager", "operations", "ops"],
    
    # Finance & Legal
    "accountant": ["accountant", "accounting", "cpa", "finance"],
    "financial analyst": ["financial analyst", "finance analyst", "fp&a"],
    "legal": ["legal", "lawyer", "attorney", "counsel", "paralegal"],
}

# =============================================================================
# EXPERIENCE LEVEL KEYWORDS
# =============================================================================

EXPERIENCE_LEVELS = {
    "junior": {"keywords": ["junior", "jr", "entry", "new grad", "fresh", "0-2 years"], "years": (0, 2)},
    "mid": {"keywords": ["mid", "mid-level", "intermediate", "2-5 years"], "years": (2, 5)},
    "senior": {"keywords": ["senior", "sr", "experienced", "5+ years", "5-10 years"], "years": (5, 15)},
    "staff": {"keywords": ["staff", "principal", "distinguished", "10+ years"], "years": (10, 30)},
    "lead": {"keywords": ["lead", "manager", "director"], "years": (7, 30)},
}

# =============================================================================
# SLANG / ABBREVIATION EXPANSION
# =============================================================================

SLANG_EXPANSIONS = {
    # Skills
    "ml": "machine learning",
    "ai": "artificial intelligence", 
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "fe": "frontend",
    "be": "backend",
    "fs": "fullstack",
    "k8s": "kubernetes",
    "tf": "terraform",
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "rb": "ruby",
    "db": "database",
    "sql": "sql",
    
    # Companies
    "faang": ["meta", "amazon", "apple", "netflix", "google"],
    "maang": ["meta", "amazon", "apple", "netflix", "google"],
    "big tech": ["google", "meta", "amazon", "apple", "microsoft"],
    
    # Titles
    "swe": "software engineer",
    "sde": "software development engineer",
    "sre": "site reliability engineer",
    "em": "engineering manager",
    "tl": "tech lead",
    "pm": "product manager",
    "de": "data engineer",
    "ds": "data scientist",
    "eng": "engineer",
    "dev": "developer",
    
    # Experience
    "yoe": "years of experience",
    "yr": "year",
    "yrs": "years",
    "exp": "experience",
    
    # Location
    "sf": "san francisco",
    "nyc": "new york",
    "la": "los angeles",
    "sea": "seattle",
    "bay area": "san francisco",
    "silicon valley": "san francisco",
}

# =============================================================================
# JAPANESE-ENGLISH MAPPINGS
# =============================================================================

JP_EN_MAPPINGS = {
    # Skills
    "エンジニア": "engineer",
    "開発者": "developer",
    "プログラマー": "programmer",
    "機械学習": "machine learning",
    "深層学習": "deep learning",
    "データサイエンス": "data science",
    "フロントエンド": "frontend",
    "バックエンド": "backend",
    "フルスタック": "fullstack",
    "クラウド": "cloud",
    
    # Companies  
    "グーグル": "google",
    "アマゾン": "amazon",
    "アップル": "apple",
    "マイクロソフト": "microsoft",
    "メルカリ": "mercari",
    "楽天": "rakuten",
    "サイバーエージェント": "cyberagent",
    "リクルート": "recruit",
    "ヤフー": "yahoo japan",
    "ソフトバンク": "softbank",
    "トヨタ": "toyota",
    "ソニー": "sony",
    "任天堂": "nintendo",
    
    # Locations
    "東京": "tokyo",
    "大阪": "osaka",
    "京都": "kyoto",
    "名古屋": "nagoya",
    "福岡": "fukuoka",
    "札幌": "sapporo",
    "渋谷": "shibuya",
    "新宿": "shinjuku",
    "六本木": "roppongi",
    "品川": "shinagawa",
}


@dataclass
class ExtractedQuery:
    """Structured extraction from natural language query"""
    # Original
    original_query: str
    
    # Extracted entities
    skills: List[str] = field(default_factory=list)
    expanded_skills: List[str] = field(default_factory=list)  # After category expansion
    companies: List[str] = field(default_factory=list)
    expanded_companies: List[str] = field(default_factory=list)  # After alias expansion
    titles: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)  # Professional certifications
    
    # Location
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    
    # Experience
    min_years: Optional[int] = None
    max_years: Optional[int] = None
    experience_level: Optional[str] = None
    
    # Search text (cleaned, for embedding)
    search_text: str = ""
    
    # Keywords for sparse search
    keywords: List[str] = field(default_factory=list)
    
    # Debug info
    extractions: Dict[str, List[str]] = field(default_factory=dict)


class SmartQueryPreprocessor:
    """
    Intelligent query preprocessor that handles ANY human input.
    
    Features:
    - Extracts skills, companies, locations from messy text
    - Expands abbreviations and slang
    - Handles Japanese text
    - Fuzzy matches typos
    - Expands skill categories (python -> django, flask, etc.)
    """
    
    def __init__(self):
        self._build_lookup_tables()
    
    def _build_lookup_tables(self):
        """Build reverse lookup tables for fast matching"""
        # All known skills (from aliases)
        self.all_skills = set()
        for canonical, aliases in SKILL_ALIASES.items():
            self.all_skills.add(canonical)
            self.all_skills.update(aliases)
        
        # All known companies
        self.all_companies = set()
        for canonical, aliases in COMPANY_ALIASES.items():
            self.all_companies.add(canonical)
            self.all_companies.update(aliases)
        
        # All known cities
        self.all_cities = set()
        for canonical, aliases in CITY_ALIASES.items():
            self.all_cities.add(canonical)
            self.all_cities.update(aliases)
        
        # All known states
        self.all_states = set()
        for canonical, aliases in STATE_ALIASES.items():
            self.all_states.add(canonical)
            self.all_states.update(aliases)
        
        # All known countries
        self.all_countries = set()
        for canonical, aliases in COUNTRY_ALIASES.items():
            self.all_countries.add(canonical)
            self.all_countries.update(aliases)
    
    def preprocess(self, query: str) -> ExtractedQuery:
        """
        Main entry point - process any query into structured extraction.
        """
        result = ExtractedQuery(original_query=query)
        
        # 1. Normalize - lowercase first
        query_lower = query.lower().strip()
        
        # 2. Translate Japanese FIRST (before other processing)
        query_translated = self._translate_japanese(query_lower)
        
        # 3. Expand slang
        query_expanded = self._expand_slang(query_translated)
        
        # 4. Extract entities
        result.skills = self._extract_skills(query_expanded)
        result.companies = self._extract_companies(query_expanded)
        result.titles = self._extract_titles(query_expanded)
        result.certifications = self._extract_certifications(query_expanded)
        
        # 5. Extract location (only if explicitly mentioned)
        city, state, country = self._extract_location(query_expanded)
        result.city = city
        result.state = state
        result.country = country
        
        # 6. Extract experience
        min_years, max_years, level = self._extract_experience(query_expanded)
        result.min_years = min_years
        result.max_years = max_years
        result.experience_level = level
        
        # 7. Expand skills with categories
        result.expanded_skills = self._expand_skill_categories(result.skills)
        
        # 8. Expand companies with aliases
        result.expanded_companies = self._expand_company_aliases(result.companies)
        
        # 9. Handle typos - try to recover skills if none found
        if not result.skills:
            result.skills = self._fuzzy_extract_skills(query_expanded)
            result.expanded_skills = self._expand_skill_categories(result.skills)
        
        # 10. Build search text for embedding
        result.search_text = self._build_search_text(result, query_expanded)
        
        # 11. Build keywords for sparse search
        result.keywords = self._build_keywords(result)
        
        # 12. Debug info
        result.extractions = {
            "original": [query],
            "translated": [query_translated],
            "expanded": [query_expanded],
            "skills_found": result.skills,
            "companies_found": result.companies,
            "location": [f"{city}, {state}, {country}"],
            "experience": [f"{min_years}-{max_years} years ({level})"],
        }
        
        return result
    
    def _translate_japanese(self, text: str) -> str:
        """Translate Japanese terms to English"""
        for jp, en in JP_EN_MAPPINGS.items():
            text = text.replace(jp.lower(), en)
        return text
    
    def _expand_slang(self, text: str) -> str:
        """Expand slang and abbreviations"""
        words = text.split()
        expanded = []
        
        for word in words:
            if word in SLANG_EXPANSIONS:
                expansion = SLANG_EXPANSIONS[word]
                if isinstance(expansion, list):
                    expanded.extend(expansion)
                else:
                    expanded.append(expansion)
            else:
                expanded.append(word)
        
        return " ".join(expanded)
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from text - PRECISE matching only"""
        found_skills = set()
        text_lower = text.lower()
        text_words = set(text_lower.split())
        
        # Only match EXACT skills from aliases - no fuzzy for skills
        for canonical, aliases in SKILL_ALIASES.items():
            for alias in aliases:
                # Word boundary check - must be a complete word
                pattern = r'\b' + re.escape(alias) + r'\b'
                if re.search(pattern, text_lower):
                    found_skills.add(canonical)
                    break
        
        # Also check for multi-word skills
        for canonical, aliases in SKILL_ALIASES.items():
            for alias in aliases:
                if ' ' in alias and alias in text_lower:
                    found_skills.add(canonical)
                    break
        
        # Check skill categories directly (frontend, backend, etc.)
        for category, category_skills in SKILL_CATEGORIES.items():
            pattern = r'\b' + re.escape(category.replace('_', ' ')) + r'\b'
            if re.search(pattern, text_lower):
                # Add the category name as a "skill"
                found_skills.add(category.replace('_', ' '))
        
        return list(found_skills)
    
    def _fuzzy_extract_skills(self, text: str) -> List[str]:
        """Fuzzy extract skills when exact matching fails - for typo tolerance"""
        found_skills = set()
        words = text.lower().split()
        
        for word in words:
            if len(word) < 3:
                continue
            
            # Check against all known skills
            best_match = None
            best_score = 0.75  # Minimum threshold
            
            for canonical, aliases in SKILL_ALIASES.items():
                for alias in aliases:
                    if len(alias) < 3:
                        continue
                    ratio = SequenceMatcher(None, word, alias).ratio()
                    if ratio > best_score:
                        best_score = ratio
                        best_match = canonical
            
            if best_match:
                found_skills.add(best_match)
        
        return list(found_skills)
    
    def _fuzzy_find_skill(self, word: str, threshold: float = 0.8) -> Optional[str]:
        """Find skill with fuzzy matching for typos"""
        best_match = None
        best_score = 0
        
        for skill in self.all_skills:
            if len(skill) > 2:
                ratio = SequenceMatcher(None, word.lower(), skill.lower()).ratio()
                if ratio > best_score and ratio >= threshold:
                    best_score = ratio
                    best_match = skill
        
        return best_match
    
    def _extract_companies(self, text: str) -> List[str]:
        """Extract companies from text - precise matching"""
        found_companies = set()
        text_lower = text.lower()
        
        # Check for company names with word boundaries
        for canonical, aliases in COMPANY_ALIASES.items():
            for alias in aliases:
                # Use word boundary for single-word aliases
                if ' ' not in alias:
                    pattern = r'\b' + re.escape(alias) + r'\b'
                    if re.search(pattern, text_lower):
                        found_companies.add(canonical)
                        break
                else:
                    # Multi-word: just check containment
                    if alias in text_lower:
                        found_companies.add(canonical)
                        break
        
        # Look for "at [company]" or "from [company]" patterns
        patterns = [
            r'\bat\s+(\w+)',  # "at Google"
            r'\bfrom\s+(\w+)',  # "from Meta"
            r'\bworked\s+(?:at|for)\s+(\w+)',  # "worked at Google"
            r'\bex[-\s]?(\w+)',  # "ex-Google"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                match_lower = match.lower().strip()
                # Check if this looks like a company
                for canonical, aliases in COMPANY_ALIASES.items():
                    if match_lower in aliases:
                        found_companies.add(canonical)
        
        return list(found_companies)
    
    def _extract_titles(self, text: str) -> List[str]:
        """Extract job titles from text - precise matching"""
        found_titles = set()
        text_lower = text.lower()
        
        for canonical, variations in TITLE_PATTERNS.items():
            for variation in variations:
                # Use word boundary matching
                pattern = r'\b' + re.escape(variation) + r'\b'
                if re.search(pattern, text_lower):
                    found_titles.add(canonical)
                    break
        
        return list(found_titles)
    
    def _extract_certifications(self, text: str) -> List[str]:
        """Extract professional certifications from text"""
        found_certs = set()
        text_lower = text.lower()
        
        # Define certification patterns with their canonical names
        cert_patterns = {
            "CPA": [r'\b(cpa|uscpa|us cpa|u\.s\. cpa|certified public accountant)\b'],
            "AWS Certified Solutions Architect": [r'\b(aws certified|aws cert|aws solutions architect)\b'],
            "AWS Certified Developer": [r'\b(aws developer|aws dev cert)\b'],
            "PMP": [r'\b(pmp|project management professional)\b'],
            "CFA": [r'\b(cfa|chartered financial analyst)\b'],
            "CISSP": [r'\b(cissp|certified information systems security professional)\b'],
            "Six Sigma": [r'\b(six sigma|6 sigma|six ?sigma (green|black) belt)\b'],
            "CMA": [r'\b(cma|certified management accountant)\b'],
            "CIA": [r'\b(cia|certified internal auditor)\b'],
            "CFP": [r'\b(cfp|certified financial planner)\b'],
        }
        
        for cert_name, patterns in cert_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    found_certs.add(cert_name)
                    break
        
        return list(found_certs)
    
    def _extract_location(self, text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract location from text - precise word boundary matching"""
        city = None
        state = None
        country = None
        text_lower = text.lower()
        
        # Skip common false positives
        false_positive_states = {"or", "in", "at", "to", "of", "for", "go"}
        
        # Check cities - require word boundaries
        for canonical, aliases in CITY_ALIASES.items():
            for alias in aliases:
                # Skip very short aliases that could be false positives
                if len(alias) < 3 and alias not in {"sf", "la", "ny"}:
                    continue
                pattern = r'\b' + re.escape(alias) + r'\b'
                if re.search(pattern, text_lower):
                    city = canonical
                    break
            if city:
                break
        
        # Check states - require word boundaries, skip false positives
        for canonical, aliases in STATE_ALIASES.items():
            for alias in aliases:
                # Skip common words that happen to be state abbreviations
                if alias in false_positive_states:
                    continue
                if len(alias) < 3:
                    # For 2-letter state codes, require them to be preceded by location context
                    location_pattern = r'(?:in|at|from|near|around)\s+' + re.escape(alias) + r'\b'
                    if re.search(location_pattern, text_lower):
                        state = canonical
                        break
                else:
                    pattern = r'\b' + re.escape(alias) + r'\b'
                    if re.search(pattern, text_lower):
                        state = canonical
                        break
            if state:
                break
        
        # Check countries - require word boundaries
        for canonical, aliases in COUNTRY_ALIASES.items():
            for alias in aliases:
                # Skip very short aliases
                if len(alias) < 3 and alias not in {"us", "uk", "jp"}:
                    continue
                # For short codes, require location context
                if len(alias) <= 2:
                    location_pattern = r'(?:in|at|from)\s+' + re.escape(alias) + r'\b'
                    if re.search(location_pattern, text_lower):
                        country = canonical
                        break
                else:
                    pattern = r'\b' + re.escape(alias) + r'\b'
                    if re.search(pattern, text_lower):
                        country = canonical
                        break
            if country:
                break
        
        # Infer state from city
        if city and not state:
            city_state_map = {
                "san francisco": "california",
                "los angeles": "california",
                "san jose": "california",
                "san diego": "california",
                "new york": "new york",
                "seattle": "washington",
                "austin": "texas",
                "boston": "massachusetts",
                "chicago": "illinois",
                "denver": "colorado",
                "atlanta": "georgia",
                "miami": "florida",
                "portland": "oregon",
                "tokyo": "tokyo",
                "osaka": "osaka",
            }
            state = city_state_map.get(city)
        
        return city, state, country
    
    def _extract_experience(self, text: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
        """Extract experience requirements from text"""
        min_years = None
        max_years = None
        level = None
        
        # Look for explicit year ranges
        year_patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)',  # "5+ years"
            r'(\d+)\s*-\s*(\d+)\s*(?:years?|yrs?)',  # "3-5 years"
            r'at\s*least\s*(\d+)\s*(?:years?|yrs?)',  # "at least 3 years"
        ]
        
        for pattern in year_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) == 2 and groups[1]:
                    min_years = int(groups[0])
                    max_years = int(groups[1])
                else:
                    min_years = int(groups[0])
                break
        
        # Look for experience level keywords
        for level_name, level_info in EXPERIENCE_LEVELS.items():
            for keyword in level_info["keywords"]:
                if keyword in text:
                    level = level_name
                    if min_years is None:
                        min_years = level_info["years"][0]
                    if max_years is None:
                        max_years = level_info["years"][1]
                    break
            if level:
                break
        
        return min_years, max_years, level
    
    def _expand_skill_categories(self, skills: List[str]) -> List[str]:
        """Expand skills to include related skills from categories"""
        expanded = set(skills)
        
        for skill in skills:
            skill_lower = skill.lower()
            
            # Check each category
            for category, category_skills in SKILL_CATEGORIES.items():
                if skill_lower in category_skills:
                    # Add some related skills from the category
                    expanded.update(category_skills[:5])
        
        return list(expanded)
    
    def _expand_company_aliases(self, companies: List[str]) -> List[str]:
        """Expand companies with all their aliases"""
        expanded = set()
        
        for company in companies:
            company_lower = company.lower()
            if company_lower in COMPANY_ALIASES:
                expanded.update(COMPANY_ALIASES[company_lower])
            else:
                expanded.add(company_lower)
        
        return list(expanded)
    
    def _build_search_text(self, result: ExtractedQuery, query_expanded: str) -> str:
        """Build clean search text for embedding"""
        parts = []
        
        # Add titles
        for title in result.titles:
            parts.append(title)
        
        # Add skills
        for skill in result.skills[:5]:  # Limit to top skills
            parts.append(skill)
        
        # Add location
        if result.city:
            parts.append(result.city)
        
        # Add experience level
        if result.experience_level:
            parts.append(f"{result.experience_level} level")
        
        if parts:
            return " ".join(parts)
        else:
            # Fallback: use the cleaned/expanded query
            return query_expanded if query_expanded else result.original_query
    
    def _build_keywords(self, result: ExtractedQuery) -> List[str]:
        """Build keyword list for sparse search"""
        keywords = []
        
        # Add all skills (expanded)
        keywords.extend(result.expanded_skills)
        
        # Add companies (expanded)
        keywords.extend(result.expanded_companies)
        
        # Add titles
        keywords.extend(result.titles)
        
        # Add location
        if result.city:
            keywords.append(result.city)
        if result.state:
            keywords.append(result.state)
        
        return list(set(keywords))


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_preprocessor: Optional[SmartQueryPreprocessor] = None


def get_preprocessor() -> SmartQueryPreprocessor:
    """Get singleton preprocessor instance"""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = SmartQueryPreprocessor()
    return _preprocessor


def smart_preprocess(query: str) -> ExtractedQuery:
    """Convenience function to preprocess a query"""
    return get_preprocessor().preprocess(query)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    test_queries = [
        "python developers in SF with 5+ years",
        "ML eng at FAANG companies",
        "東京のPythonエンジニア",
        "ex google engineers who know react",
        "senior backend dev with aws k8s exp in bay area",
        "pythin devloper in sanfracisco",  # typos
        "full stack js devs nyc",
        "data scientist ml deep learning",
        "rust or go developers for infrastructure",
        "people who worked at meta or google, frontend, 3-5 years experience",
    ]
    
    preprocessor = SmartQueryPreprocessor()
    
    print("=" * 80)
    print("SMART QUERY PREPROCESSOR TEST")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 60)
        
        result = preprocessor.preprocess(query)
        
        print(f"   Skills:     {result.skills}")
        print(f"   Expanded:   {result.expanded_skills[:8]}...")
        print(f"   Companies:  {result.companies}")
        print(f"   Location:   {result.city}, {result.state}, {result.country}")
        print(f"   Experience: {result.min_years}-{result.max_years} years ({result.experience_level})")
        print(f"   Search:     {result.search_text}")
        print(f"   Keywords:   {result.keywords[:10]}...")
