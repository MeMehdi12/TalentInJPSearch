"""
OpenAI Query Parser Module
===========================

Production-ready natural language query parser using OpenAI GPT-4o-mini.

Features:
- Converts natural language to structured JSON filters
- Fallback to regex parser on failures
- Cost tracking and logging
- Retry logic with exponential backoff
- Comprehensive error handling

Author: Talentin AI Search Team
Version: 1.0.0
"""

import json
import logging
import re
from typing import Dict, Optional, Tuple
from pathlib import Path

from openai_config import (
    get_openai_client,
    get_openai_config,
    call_openai_with_retry,
    get_cost_tracker
)
from search_schema import ParsedQueryV2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_system_prompt() -> str:
    """Load query parser system prompt from file"""
    prompt_path = Path(__file__).parent / "prompts" / "query_parser.txt"
    
    if not prompt_path.exists():
        logger.warning(f"System prompt not found at {prompt_path}, using default")
        return "You are a talent search query parser. Convert natural language to JSON filters."
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def parse_query_with_openai(
    query: str,
    fallback_to_regex: bool = True
) -> Tuple[Optional[ParsedQueryV2], str, float]:
    """
    Parse natural language query using OpenAI GPT-4o-mini.
    
    Args:
        query: Natural language search query
        fallback_to_regex: If True, fall back to regex parser on failure
    
    Returns:
        Tuple of (ParsedQueryV2 object or None, parsing_method, cost_usd)
        - parsing_method: "openai", "regex", or "failed"
        - cost_usd: Estimated cost in USD
    
    Examples:
        >>> result, method, cost = parse_query_with_openai("Find senior Python devs in Tokyo")
        >>> print(result.filters.skills.must_have)
        ['python']
        >>> print(method)
        'openai'
        >>> print(f"${cost:.6f}")
        $0.000123
    """
    try:
        client = get_openai_client()
        config = get_openai_config()
        system_prompt = load_system_prompt()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Parse this query: {query}"}
        ]
        
        logger.info("="*60)
        logger.info(f"🤖 OpenAI Parser: Sending query to GPT")
        logger.info(f"   Query: '{query}'")
        logger.info(f"   Query Length: {len(query)}")
        logger.info(f"   Model: {config.model}")
        logger.info("="*60)
        
        # Call OpenAI with retry logic
        response = call_openai_with_retry(
            client=client,
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            response_format={"type": "json_object"}
        )
        
        # Extract response
        content = response.choices[0].message.content
        usage = response.usage
        
        # Track cost
        cost_tracker = get_cost_tracker()
        cost_tracker.add_call(
            model=config.model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens
        )
        
        # Estimate cost
        from openai_config import estimate_cost
        cost = estimate_cost(
            config.model,
            usage.prompt_tokens,
            usage.completion_tokens
        )
        
        # Parse JSON response
        try:
            parsed_json = json.loads(content)
            logger.info(f"Successfully parsed query with OpenAI: {parsed_json}")
            
            # Convert to ParsedQueryV2 object
            parsed_query = _json_to_parsed_query(parsed_json)
            
            return parsed_query, "openai", cost
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI response as JSON: {e}")
            logger.error(f"Response content: {content}")
            
            if fallback_to_regex:
                logger.info("Falling back to regex parser")
                return _fallback_regex_parser(query), "regex", cost
            else:
                return None, "failed", cost
    
    except Exception as e:
        logger.error(f"OpenAI parsing failed: {e}")
        
        if fallback_to_regex:
            logger.info("Falling back to regex parser due to error")
            return _fallback_regex_parser(query), "regex", 0.0
        else:
            return None, "failed", 0.0


def _json_to_parsed_query(data: dict) -> ParsedQueryV2:
    """
    Convert JSON response to ParsedQueryV2 object.
    
    Args:
        data: Parsed JSON from OpenAI
    
    Returns:
        ParsedQueryV2: Structured query object
    """
    from search_schema import (
        SkillFiltersV2, LocationFilterV2, ExperienceFilterV2,
        CompanyFilterV2, FiltersV2, SearchOptionsV2, ParsedQueryV2
    )
    
    # Extract filters
    filters_data = data.get("filters", {})
    
    # Skills
    skills_data = filters_data.get("skills", {})
    skills = SkillFiltersV2(
        must_have=skills_data.get("must_have", []),
        nice_to_have=skills_data.get("nice_to_have", []),
        exclude=skills_data.get("exclude", [])
    )
    
    # Location
    location_data = filters_data.get("location", {})
    location = LocationFilterV2(
        city=location_data.get("city"),
        state=location_data.get("state"),
        country=location_data.get("country")
    )
    
    # Experience
    experience_data = filters_data.get("experience", {})
    experience = ExperienceFilterV2(
        min_years=experience_data.get("min_years"),
        max_years=experience_data.get("max_years")
    )
    
    # Companies
    companies_data = filters_data.get("companies", {})
    companies = CompanyFilterV2(
        worked_at=companies_data.get("worked_at", []),
        current_only=companies_data.get("current_only", False),
        exclude=companies_data.get("exclude", [])
    )
    
    # Job titles and domain
    job_titles = filters_data.get("job_titles", [])
    domain = filters_data.get("domain")
    certifications = filters_data.get("certifications", [])
    
    # Options
    options_data = data.get("options", {})
    options = SearchOptionsV2(
        limit=options_data.get("limit", 50),
        expand_skills=options_data.get("expand_skills", True)
    )
    
    # Build filters object
    filters = FiltersV2(
        skills=skills,
        location=location,
        experience=experience,
        companies=companies,
        job_titles=job_titles,
        certifications=certifications,
        domain=domain
    )
    
    # Build final query
    return ParsedQueryV2(
        search_text=data.get("search_text", ""),
        filters=filters,
        options=options
    )


def _fallback_regex_parser(query: str) -> Optional[ParsedQueryV2]:
    """
    Fallback regex-based parser for simple queries.
    
    This is a simplified parser that handles basic patterns.
    Not as sophisticated as OpenAI, but reliable for common queries.
    
    Args:
        query: Natural language query
    
    Returns:
        ParsedQueryV2 or None
    """
    from search_schema import (
        SkillFiltersV2, LocationFilterV2, ExperienceFilterV2,
        CompanyFilterV2, FiltersV2, SearchOptionsV2, ParsedQueryV2
    )
    
    logger.info(f"Using regex fallback parser for: '{query}'")
    
    query_lower = query.lower()
    
    # Extract skills (basic keyword matching)
    skills = []
    skill_keywords = [
        "python", "java", "javascript", "react", "node", "aws", "docker",
        "kubernetes", "sql", "mongodb", "machine learning", "ml", "ai",
        "tensorflow", "pytorch", "go", "rust", "c++", "ruby"
    ]
    for skill in skill_keywords:
        if skill in query_lower:
            skills.append(skill)
    
    # Extract location
    city = None
    country = None
    
    # Cities
    city_patterns = [
        (r'\bin\s+(\w+(?:\s+\w+)?)', 1),  # "in Tokyo"
        (r'(\w+(?:\s+\w+)?)\s+based', 1),  # "Tokyo based"
    ]
    for pattern, group in city_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            city = match.group(group)
            break
    
    # Countries
    if "japan" in query_lower:
        country = "Japan"
    elif "usa" in query_lower or "united states" in query_lower:
        country = "United States"
    
    # Extract experience
    min_years = None
    max_years = None
    
    # Patterns: "5+ years", "3-7 years", "at least 10 years"
    exp_patterns = [
        (r'(\d+)\+\s*years?', lambda m: (int(m.group(1)), None)),
        (r'(\d+)\s*-\s*(\d+)\s*years?', lambda m: (int(m.group(1)), int(m.group(2)))),
        (r'at least (\d+)\s*years?', lambda m: (int(m.group(1)), None)),
    ]
    
    for pattern, extractor in exp_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            min_years, max_years = extractor(match)
            break
    
    # Seniority mapping
    if "senior" in query_lower and min_years is None:
        min_years = 5
        max_years = 15
    elif "junior" in query_lower and min_years is None:
        min_years = 0
        max_years = 3
    
    # Extract companies
    companies = []
    company_keywords = ["google", "amazon", "microsoft", "apple", "meta", "facebook"]
    for company in company_keywords:
        if company in query_lower:
            companies.append(company)
    
    # FAANG expansion
    if "faang" in query_lower:
        companies = ["facebook", "apple", "amazon", "netflix", "google"]
    
    # Extract certifications
    certifications = []
    certification_patterns = {
        "CPA": r'\b(cpa|uscpa|us cpa|certified public accountant)\b',
        "AWS Certified": r'\b(aws certified|aws cert)\b',
        "PMP": r'\b(pmp|project management professional)\b',
        "CFA": r'\b(cfa|chartered financial analyst)\b',
        "CISSP": r'\b(cissp|certified information systems security professional)\b',
        "Six Sigma": r'\b(six sigma|6 sigma)\b',
    }
    
    for cert_name, pattern in certification_patterns.items():
        if re.search(pattern, query_lower):
            certifications.append(cert_name)
    
    # Build filters
    filters = FiltersV2(
        skills=SkillFiltersV2(must_have=skills, nice_to_have=[], exclude=[]),
        location=LocationFilterV2(city=city, state=None, country=country),
        experience=ExperienceFilterV2(min_years=min_years, max_years=max_years),
        companies=CompanyFilterV2(worked_at=companies, current_only=False, exclude=[]),
        job_titles=[],
        certifications=certifications,
        domain=None
    )
    
    return ParsedQueryV2(
        search_text=query,
        filters=filters,
        options=SearchOptionsV2(limit=50, expand_skills=True)
    )


def validate_parsed_query(parsed_query: ParsedQueryV2) -> Tuple[bool, Optional[str]]:
    """
    Validate parsed query for common issues.
    
    Args:
        parsed_query: Parsed query object
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if query is too restrictive
    filters = parsed_query.filters
    
    # Count hard filters
    hard_filter_count = 0
    if filters.skills.must_have:
        hard_filter_count += len(filters.skills.must_have)
    if filters.location.city:
        hard_filter_count += 1
    if filters.companies.worked_at:
        hard_filter_count += 1
    if filters.experience.min_years is not None:
        hard_filter_count += 1
    
    if hard_filter_count > 10:
        return False, "Query is too restrictive (>10 hard filters). Results may be empty."
    
    # Check experience range
    if filters.experience.min_years and filters.experience.max_years:
        if filters.experience.min_years > filters.experience.max_years:
            return False, "Invalid experience range: min > max"
        if filters.experience.min_years > 50:
            return False, "Minimum experience too high (>50 years)"
    
    # Check limit
    if parsed_query.options.limit > 100:
        return False, "Limit too high (max 100)"
    
    return True, None


# Example usage
if __name__ == "__main__":
    # Test queries
    test_queries = [
        "Find senior Python developers in Tokyo with 5+ years",
        "ML engineers from FAANG with 7+ years, not from Meta",
        "Backend engineer with AWS and Docker experience",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        result, method, cost = parse_query_with_openai(query)
        
        if result:
            print(f"Parsing method: {method}")
            print(f"Cost: ${cost:.6f}")
            print(f"Search text: {result.search_text}")
            print(f"Must-have skills: {result.filters.skills.must_have}")
            print(f"Location: {result.filters.location.city}, {result.filters.location.country}")
            print(f"Experience: {result.filters.experience.min_years}+ years")
            print(f"Companies: {result.filters.companies.worked_at}")
        else:
            print(f"Parsing failed (method: {method})")
