"""
Text Parsing Service Module
============================

Production-ready text parsing using OpenAI for:
- Job description parsing
- Skill extraction from unstructured text
- Profile summarization

Features:
- Structured output with validation
- Cost tracking
- Caching for repeated queries
- Error handling with fallbacks

Author: Talentin AI Search Team
Version: 1.0.0
"""

import json
import logging
import hashlib
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from functools import lru_cache

from openai_config import (
    get_openai_client,
    get_openai_config,
    call_openai_with_retry,
    get_cost_tracker,
    estimate_cost
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_jd_parser_prompt() -> str:
    """Load JD parser system prompt from file"""
    prompt_path = Path(__file__).parent / "prompts" / "jd_parser.txt"
    
    if not prompt_path.exists():
        logger.warning(f"JD parser prompt not found at {prompt_path}, using default")
        return "You are a job description parser. Extract requirements as JSON."
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


@lru_cache(maxsize=100)
def parse_job_description(text: str) -> Tuple[Optional[Dict], float]:
    """
    Parse job description and extract structured requirements.
    
    Args:
        text: Job description text
    
    Returns:
        Tuple of (parsed_data dict or None, cost_usd)
    
    Example:
        >>> jd = "We need a senior Python developer with 5+ years..."
        >>> result, cost = parse_job_description(jd)
        >>> print(result['required_skills'])
        ['python']
        >>> print(result['min_years_experience'])
        5
    """
    try:
        client = get_openai_client()
        config = get_openai_config()
        system_prompt = load_jd_parser_prompt()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Parse this job description:\n\n{text}"}
        ]
        
        logger.info(f"Parsing job description ({len(text)} chars)")
        
        # Call OpenAI
        response = call_openai_with_retry(
            client=client,
            model=config.model,
            messages=messages,
            temperature=0.1,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        # Extract and parse response
        content = response.choices[0].message.content
        usage = response.usage
        
        # Track cost
        cost_tracker = get_cost_tracker()
        cost_tracker.add_call(
            model=config.model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens
        )
        
        cost = estimate_cost(
            config.model,
            usage.prompt_tokens,
            usage.completion_tokens
        )
        
        # Parse JSON
        parsed_data = json.loads(content)
        logger.info(f"Successfully parsed JD: {len(parsed_data.get('required_skills', []))} required skills")
        
        return parsed_data, cost
        
    except Exception as e:
        logger.error(f"JD parsing failed: {e}")
        return None, 0.0


@lru_cache(maxsize=200)
def extract_skills_from_text(text: str) -> Tuple[Optional[List[Dict]], float]:
    """
    Extract skills from unstructured text.
    
    Args:
        text: Unstructured text (e.g., resume, profile description)
    
    Returns:
        Tuple of (list of skill dicts or None, cost_usd)
        Each skill dict: {"skill": str, "confidence": float, "category": str}
    
    Example:
        >>> text = "Experienced in Node.js, React, and PostgreSQL..."
        >>> skills, cost = extract_skills_from_text(text)
        >>> print(skills[0])
        {'skill': 'node.js', 'confidence': 0.95, 'category': 'backend'}
    """
    try:
        client = get_openai_client()
        config = get_openai_config()
        
        system_prompt = """You are an expert at extracting technical skills from text.

Extract all technical skills mentioned and categorize them.

Output JSON format:
{
  "skills": [
    {"skill": "python", "confidence": 0.95, "category": "programming"},
    {"skill": "aws", "confidence": 0.90, "category": "cloud"}
  ]
}

Categories: programming, frontend, backend, database, cloud, devops, ml_ai, design, other

Confidence: 0.0 to 1.0 (how certain the skill is mentioned)
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract skills from this text:\n\n{text}"}
        ]
        
        logger.info(f"Extracting skills from text ({len(text)} chars)")
        
        # Call OpenAI
        response = call_openai_with_retry(
            client=client,
            model=config.model,
            messages=messages,
            temperature=0.1,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        # Extract and parse
        content = response.choices[0].message.content
        usage = response.usage
        
        # Track cost
        cost_tracker = get_cost_tracker()
        cost_tracker.add_call(
            model=config.model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens
        )
        
        cost = estimate_cost(
            config.model,
            usage.prompt_tokens,
            usage.completion_tokens
        )
        
        # Parse JSON
        parsed_data = json.loads(content)
        skills = parsed_data.get("skills", [])
        
        logger.info(f"Extracted {len(skills)} skills")
        
        return skills, cost
        
    except Exception as e:
        logger.error(f"Skill extraction failed: {e}")
        return None, 0.0


def summarize_profile(profile_data: Dict) -> Tuple[Optional[Dict], float]:
    """
    Generate a concise summary of a candidate profile.
    
    Args:
        profile_data: Profile dict with keys: full_name, current_title, 
                     current_company, years_experience, skills, location
    
    Returns:
        Tuple of (summary dict or None, cost_usd)
        Summary dict: {"summary": str, "key_highlights": List[str]}
    
    Example:
        >>> profile = {
        ...     "full_name": "Tanaka Yuki",
        ...     "current_title": "Senior ML Engineer",
        ...     "current_company": "Google",
        ...     "years_experience": 8,
        ...     "skills": ["python", "tensorflow", "pytorch"],
        ...     "location": "Tokyo, Japan"
        ... }
        >>> result, cost = summarize_profile(profile)
        >>> print(result['summary'])
        'Senior ML Engineer with 8 years of experience at Google...'
    """
    try:
        client = get_openai_client()
        config = get_openai_config()
        
        system_prompt = """You are an expert at summarizing professional profiles.

Create a concise 2-sentence summary and extract 3-5 key highlights.

Output JSON format:
{
  "summary": "Two sentence summary of the candidate...",
  "key_highlights": [
    "8 years of experience in AI/ML",
    "Worked at Google and Rakuten",
    "Expert in Python, TensorFlow, PyTorch"
  ]
}
"""
        
        # Build profile description
        profile_text = f"""
Name: {profile_data.get('full_name', 'Unknown')}
Current Title: {profile_data.get('current_title', 'Unknown')}
Current Company: {profile_data.get('current_company', 'Unknown')}
Years of Experience: {profile_data.get('years_experience', 'Unknown')}
Top Skills: {', '.join(profile_data.get('skills', [])[:10])}
Location: {profile_data.get('location', 'Unknown')}
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Summarize this profile:\n{profile_text}"}
        ]
        
        logger.info(f"Summarizing profile: {profile_data.get('full_name', 'Unknown')}")
        
        # Call OpenAI
        response = call_openai_with_retry(
            client=client,
            model=config.model,
            messages=messages,
            temperature=0.3,
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        
        # Extract and parse
        content = response.choices[0].message.content
        usage = response.usage
        
        # Track cost
        cost_tracker = get_cost_tracker()
        cost_tracker.add_call(
            model=config.model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens
        )
        
        cost = estimate_cost(
            config.model,
            usage.prompt_tokens,
            usage.completion_tokens
        )
        
        # Parse JSON
        summary_data = json.loads(content)
        logger.info(f"Profile summarized successfully")
        
        return summary_data, cost
        
    except Exception as e:
        logger.error(f"Profile summarization failed: {e}")
        return None, 0.0


def jd_to_search_query(jd_text: str) -> Tuple[Optional[Dict], float]:
    """
    Convert job description to search query filters.
    
    This combines JD parsing with query structure conversion.
    
    Args:
        jd_text: Job description text
    
    Returns:
        Tuple of (search_query dict or None, cost_usd)
        Search query compatible with ParsedQueryV2 format
    """
    # Parse JD
    jd_parsed, cost = parse_job_description(jd_text)
    
    if not jd_parsed:
        return None, cost
    
    # Convert to search query format
    search_query = {
        "search_text": " ".join(jd_parsed.get("job_titles", [])),
        "filters": {
            "skills": {
                "must_have": jd_parsed.get("required_skills", []),
                "nice_to_have": jd_parsed.get("nice_to_have_skills", []),
                "exclude": []
            },
            "location": {
                "city": jd_parsed.get("location_preferences", {}).get("cities", [None])[0],
                "state": None,
                "country": None
            },
            "experience": {
                "min_years": jd_parsed.get("min_years_experience"),
                "max_years": jd_parsed.get("max_years_experience")
            },
            "companies": {
                "worked_at": [],
                "current_only": False,
                "exclude": []
            },
            "job_titles": jd_parsed.get("job_titles", []),
            "domain": jd_parsed.get("industry")
        },
        "options": {
            "limit": 50,
            "expand_skills": True
        }
    }
    
    logger.info(f"Converted JD to search query: {len(search_query['filters']['skills']['must_have'])} required skills")
    
    return search_query, cost


# Example usage
if __name__ == "__main__":
    # Test JD parsing
    jd = """
    We are looking for a Senior Backend Engineer with 5+ years of experience.
    
    Requirements:
    - Strong proficiency in Python and Go
    - Experience with AWS, Docker, and Kubernetes
    - Database experience (PostgreSQL, Redis)
    
    Nice to have:
    - GraphQL experience
    - Microservices architecture
    """
    
    print("="*60)
    print("Testing JD Parser")
    print("="*60)
    result, cost = parse_job_description(jd)
    if result:
        print(f"Cost: ${cost:.6f}")
        print(f"Required skills: {result['required_skills']}")
        print(f"Nice-to-have: {result['nice_to_have_skills']}")
        print(f"Min experience: {result['min_years_experience']} years")
    
    # Test skill extraction
    text = "Experienced in building scalable microservices with Node.js, React, and PostgreSQL."
    
    print("\n" + "="*60)
    print("Testing Skill Extraction")
    print("="*60)
    skills, cost = extract_skills_from_text(text)
    if skills:
        print(f"Cost: ${cost:.6f}")
        for skill in skills:
            print(f"- {skill['skill']} ({skill['category']}, confidence: {skill['confidence']})")
