"""
Input Normalizers for Robust Search
====================================
Handle crazy human inputs: aliases, abbreviations, typos, variations.
"""

import re
from typing import List, Optional, Dict, Set


# =============================================================================
# COMPANY ALIASES - Map variations to canonical names
# =============================================================================

COMPANY_ALIASES: Dict[str, Set[str]] = {
    # FAANG / Big Tech
    "google": {"google", "alphabet", "google inc", "google llc", "youtube", "deepmind", "waymo"},
    "meta": {"meta", "facebook", "fb", "instagram", "whatsapp", "meta platforms"},
    "amazon": {"amazon", "aws", "amazon web services", "amazon.com", "twitch", "whole foods"},
    "apple": {"apple", "apple inc", "apple computer"},
    "netflix": {"netflix", "netflix inc"},
    "microsoft": {"microsoft", "msft", "ms", "linkedin", "github", "azure"},
    
    # Other Tech Giants
    "nvidia": {"nvidia", "nvda"},
    "tesla": {"tesla", "tesla motors", "tesla inc"},
    "twitter": {"twitter", "x", "x corp"},
    "uber": {"uber", "uber technologies"},
    "lyft": {"lyft", "lyft inc"},
    "airbnb": {"airbnb", "air bnb"},
    "salesforce": {"salesforce", "sfdc", "salesforce.com"},
    "oracle": {"oracle", "oracle corporation"},
    "ibm": {"ibm", "international business machines"},
    "intel": {"intel", "intel corporation"},
    "cisco": {"cisco", "cisco systems"},
    "adobe": {"adobe", "adobe systems"},
    
    # Startups / Unicorns
    "stripe": {"stripe", "stripe inc"},
    "databricks": {"databricks"},
    "snowflake": {"snowflake", "snowflake computing"},
    "coinbase": {"coinbase", "coinbase global"},
    "robinhood": {"robinhood", "robinhood markets"},
    
    # Finance
    "goldman": {"goldman", "goldman sachs", "gs"},
    "jpmorgan": {"jpmorgan", "jp morgan", "jpm", "chase", "jpmorgan chase"},
    "morgan stanley": {"morgan stanley", "ms"},
    "citadel": {"citadel", "citadel securities", "citadel llc"},
    
    # Consulting
    "mckinsey": {"mckinsey", "mckinsey & company", "mckinsey and company"},
    "bcg": {"bcg", "boston consulting group", "boston consulting"},
    "bain": {"bain", "bain & company", "bain and company"},
    "deloitte": {"deloitte", "deloitte consulting"},
    "accenture": {"accenture"},
    
    # Japanese Companies
    "rakuten": {"rakuten", "楽天"},
    "sony": {"sony", "ソニー", "sony corporation"},
    "toyota": {"toyota", "トヨタ", "toyota motor"},
    "softbank": {"softbank", "ソフトバンク", "soft bank"},
    "mercari": {"mercari", "メルカリ"},
    "line": {"line", "line corporation", "ライン"},
    "yahoo japan": {"yahoo japan", "ヤフー", "yahoo jp", "yahoo! japan"},
    "cyberagent": {"cyberagent", "サイバーエージェント", "cyber agent"},
}


# =============================================================================
# LOCATION ALIASES - Map abbreviations/variations to canonical names
# =============================================================================

CITY_ALIASES: Dict[str, Set[str]] = {
    # US Cities
    "san francisco": {"san francisco", "sf", "san fran", "s.f.", "san fransisco"},
    "new york": {"new york", "nyc", "ny", "new york city", "manhattan"},
    "los angeles": {"los angeles", "la", "l.a.", "los angles"},
    "seattle": {"seattle", "sea"},
    "austin": {"austin", "atx"},
    "chicago": {"chicago", "chi"},
    "boston": {"boston", "bos"},
    "denver": {"denver", "den"},
    "san jose": {"san jose", "sj"},
    "palo alto": {"palo alto", "pa"},
    "mountain view": {"mountain view", "mv", "mtv"},
    "menlo park": {"menlo park", "mp"},
    "sunnyvale": {"sunnyvale"},
    "cupertino": {"cupertino"},
    
    # Japan
    "tokyo": {"tokyo", "東京", "tky"},
    "osaka": {"osaka", "大阪"},
    "kyoto": {"kyoto", "京都"},
    "nagoya": {"nagoya", "名古屋"},
    "fukuoka": {"fukuoka", "福岡"},
    "yokohama": {"yokohama", "横浜"},
    "sapporo": {"sapporo", "札幌"},
    "kobe": {"kobe", "神戸"},
    "shibuya": {"shibuya", "渋谷"},
    "shinjuku": {"shinjuku", "新宿"},
    "minato": {"minato", "港区"},
    "roppongi": {"roppongi", "六本木"},
    
    # International
    "london": {"london", "ldn"},
    "singapore": {"singapore", "sg"},
    "berlin": {"berlin"},
    "toronto": {"toronto", "to"},
    "vancouver": {"vancouver", "van"},
    "sydney": {"sydney", "syd"},
    "bangalore": {"bangalore", "bengaluru"},
    "mumbai": {"mumbai", "bombay"},
    "hong kong": {"hong kong", "hk"},
    "shanghai": {"shanghai", "上海"},
    "beijing": {"beijing", "peking", "北京"},
    "shenzhen": {"shenzhen", "深圳"},
}

STATE_ALIASES: Dict[str, Set[str]] = {
    # US States
    "california": {"california", "ca", "cali"},
    "new york": {"new york", "ny"},
    "texas": {"texas", "tx"},
    "washington": {"washington", "wa"},
    "massachusetts": {"massachusetts", "ma", "mass"},
    "illinois": {"illinois", "il"},
    "colorado": {"colorado", "co"},
    "georgia": {"georgia", "ga"},
    "florida": {"florida", "fl"},
    "oregon": {"oregon", "or"},
    "virginia": {"virginia", "va"},
    
    # Japan Prefectures
    "tokyo": {"tokyo", "東京都"},
    "osaka": {"osaka", "大阪府"},
    "kanagawa": {"kanagawa", "神奈川県"},
    "aichi": {"aichi", "愛知県"},
    "fukuoka": {"fukuoka", "福岡県"},
    "hokkaido": {"hokkaido", "北海道"},
    "kyoto": {"kyoto", "京都府"},
}

COUNTRY_ALIASES: Dict[str, Set[str]] = {
    "usa": {"usa", "us", "united states", "america", "u.s.", "u.s.a."},
    "japan": {"japan", "jp", "日本", "jpn"},
    "uk": {"uk", "united kingdom", "britain", "england", "gb"},
    "canada": {"canada", "ca"},
    "germany": {"germany", "de", "deutschland"},
    "france": {"france", "fr"},
    "australia": {"australia", "au"},
    "india": {"india", "in"},
    "china": {"china", "cn", "中国"},
    "singapore": {"singapore", "sg"},
}


# =============================================================================
# SKILL ALIASES - Common variations and abbreviations
# =============================================================================

SKILL_ALIASES: Dict[str, Set[str]] = {
    # Languages
    "python": {"python", "py", "python3", "python 3"},
    "javascript": {"javascript", "js", "ecmascript", "es6", "es2015"},
    "typescript": {"typescript", "ts"},
    "golang": {"golang", "go"},
    "c++": {"c++", "cpp", "c plus plus"},
    "c#": {"c#", "csharp", "c sharp"},
    "ruby": {"ruby", "rb"},
    "rust": {"rust", "rs"},
    "kotlin": {"kotlin", "kt"},
    "swift": {"swift"},
    
    # Frameworks
    "react": {"react", "reactjs", "react.js", "react js"},
    "angular": {"angular", "angularjs", "angular.js"},
    "vue": {"vue", "vuejs", "vue.js", "vue js"},
    "django": {"django"},
    "flask": {"flask"},
    "fastapi": {"fastapi", "fast api"},
    "spring": {"spring", "spring boot", "springboot"},
    "rails": {"rails", "ruby on rails", "ror"},
    "nextjs": {"nextjs", "next.js", "next js"},
    "express": {"express", "expressjs", "express.js"},
    "node": {"node", "nodejs", "node.js", "node js"},
    
    # Databases
    "postgresql": {"postgresql", "postgres", "psql", "pg"},
    "mysql": {"mysql", "my sql"},
    "mongodb": {"mongodb", "mongo"},
    "redis": {"redis"},
    "elasticsearch": {"elasticsearch", "es", "elastic"},
    "dynamodb": {"dynamodb", "dynamo", "dynamo db"},
    "cassandra": {"cassandra"},
    
    # Cloud / DevOps
    "aws": {"aws", "amazon web services", "amazon aws"},
    "gcp": {"gcp", "google cloud", "google cloud platform"},
    "azure": {"azure", "microsoft azure"},
    "docker": {"docker", "docker container"},
    "kubernetes": {"kubernetes", "k8s", "kube"},
    "terraform": {"terraform", "tf"},
    "ci/cd": {"ci/cd", "cicd", "ci cd", "continuous integration"},
    "jenkins": {"jenkins"},
    "github actions": {"github actions", "gh actions", "gha"},
    
    # ML / Data
    "machine learning": {"machine learning", "ml"},
    "deep learning": {"deep learning", "dl"},
    "tensorflow": {"tensorflow", "tf"},
    "pytorch": {"pytorch", "torch"},
    "pandas": {"pandas", "pd"},
    "numpy": {"numpy", "np"},
    "scikit-learn": {"scikit-learn", "sklearn", "scikit learn"},
    "spark": {"spark", "apache spark", "pyspark"},
    "llm": {"llm", "large language model", "large language models"},
    "nlp": {"nlp", "natural language processing"},
    "computer vision": {"computer vision", "cv"},
    
    # Other
    "graphql": {"graphql", "graph ql"},
    "rest": {"rest", "restful", "rest api"},
    "agile": {"agile", "scrum"},
    "sql": {"sql", "structured query language"},
    "git": {"git", "github", "gitlab"},
}


# =============================================================================
# NORMALIZER FUNCTIONS
# =============================================================================

def normalize_text(text: Optional[str]) -> str:
    """Basic text normalization: lowercase, strip, clean"""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text.lower().strip())


def find_canonical_name(text: str, aliases: Dict[str, Set[str]]) -> str:
    """Find canonical name from aliases dictionary"""
    text = normalize_text(text)
    if not text:
        return ""
    
    for canonical, variations in aliases.items():
        if text in variations:
            return canonical
    
    return text  # Return original if no match


def normalize_company(company: Optional[str]) -> str:
    """Normalize company name with alias expansion"""
    return find_canonical_name(company, COMPANY_ALIASES)


def normalize_city(city: Optional[str]) -> str:
    """Normalize city name"""
    return find_canonical_name(city, CITY_ALIASES)


def normalize_state(state: Optional[str]) -> str:
    """Normalize state/prefecture name"""
    return find_canonical_name(state, STATE_ALIASES)


def normalize_country(country: Optional[str]) -> str:
    """Normalize country name"""
    return find_canonical_name(country, COUNTRY_ALIASES)


def normalize_skill(skill: Optional[str]) -> str:
    """Normalize skill name"""
    return find_canonical_name(skill, SKILL_ALIASES)


def expand_company_search(companies: List[str]) -> List[str]:
    """
    Expand company list to include all aliases.
    E.g., ["Google"] -> ["google", "alphabet", "youtube", "deepmind", ...]
    """
    expanded = set()
    for company in companies:
        canonical = normalize_company(company)
        if canonical in COMPANY_ALIASES:
            expanded.update(COMPANY_ALIASES[canonical])
        else:
            expanded.add(normalize_text(company))
    return list(expanded)


def expand_skill_search(skills: List[str]) -> List[str]:
    """
    Expand skills to include all aliases.
    E.g., ["python"] -> ["python", "py", "python3", "python 3"]
    """
    expanded = set()
    for skill in skills:
        canonical = normalize_skill(skill)
        if canonical in SKILL_ALIASES:
            expanded.update(SKILL_ALIASES[canonical])
        else:
            expanded.add(normalize_text(skill))
    return list(expanded)


def normalize_location(city: Optional[str] = None, 
                       state: Optional[str] = None, 
                       country: Optional[str] = None) -> Dict[str, str]:
    """Normalize full location"""
    return {
        "city": normalize_city(city),
        "state": normalize_state(state),
        "country": normalize_country(country),
    }


def clean_list(items: Optional[List[str]]) -> List[str]:
    """Clean and deduplicate a list of strings"""
    if not items:
        return []
    return list(set(normalize_text(item) for item in items if item))


# =============================================================================
# FUZZY MATCHING (for typo tolerance)
# =============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein edit distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def fuzzy_match(text: str, candidates: List[str], max_distance: int = 2) -> Optional[str]:
    """Find best fuzzy match from candidates"""
    text = normalize_text(text)
    if not text:
        return None
    
    best_match = None
    best_distance = max_distance + 1
    
    for candidate in candidates:
        distance = levenshtein_distance(text, candidate.lower())
        if distance < best_distance:
            best_distance = distance
            best_match = candidate
    
    return best_match if best_distance <= max_distance else None


def fuzzy_match_company(company: str) -> Optional[str]:
    """Fuzzy match company name"""
    # First try exact alias match
    canonical = normalize_company(company)
    if canonical != normalize_text(company):
        return canonical
    
    # Try fuzzy match against all known companies
    all_companies = []
    for canonical, aliases in COMPANY_ALIASES.items():
        all_companies.extend(aliases)
    
    return fuzzy_match(company, all_companies, max_distance=2)


if __name__ == "__main__":
    # Test the normalizers
    print("=== Testing Normalizers ===\n")
    
    # Companies
    print("Company normalization:")
    for test in ["Google", "FB", "amazon web services", "メルカリ", "msft"]:
        print(f"  {test} -> {normalize_company(test)}")
    
    print("\nCompany expansion:")
    print(f"  ['Google', 'Meta'] -> {expand_company_search(['Google', 'Meta'])[:8]}...")
    
    # Locations
    print("\nLocation normalization:")
    for test in ["SF", "NYC", "東京", "ca", "uk"]:
        print(f"  {test} -> city={normalize_city(test)}, state={normalize_state(test)}, country={normalize_country(test)}")
    
    # Skills
    print("\nSkill normalization:")
    for test in ["js", "k8s", "py", "react.js", "postgres"]:
        print(f"  {test} -> {normalize_skill(test)}")
