"""
Ranking Service - Bonus scoring after hybrid retrieval
=======================================================
Applies additional ranking bonuses based on:
- Skills coverage (MASSIVE boost for exact matches)
- Company match (strong boost for work history)
- Education match
- Location proximity
- Title match (STRONG boost for exact title)
- Experience fit
- Certification match

Philosophy: EXACT MATCHES get MASSIVE boosts. 
"The output should be dem good - exact match - fuck this is the candidate I want"
"""

from typing import List, Dict, Any, Optional, Set
from search_schema import HybridSearchQuery, CandidateResult, RankingFactors


class RankingService:
    """
    Applies bonus scores to candidates after hybrid retrieval.
    
    All bonuses are ADDITIVE to the base RRF score.
    This ensures candidates that match more criteria rank higher.
    
    PHILOSOPHY: Exact matches deserve MASSIVE boosts.
    If someone searches for "python developer in SF" and a candidate is
    literally a Python Developer in San Francisco, they should be #1.
    """
    
    # Bonus weights - AGGRESSIVE for exact matches
    BONUSES = {
        # Core match bonuses (additive)
        "skills_perfect_match": 0.40,     # ALL required skills matched = MASSIVE boost
        "skills_coverage": 0.25,          # Partial skill coverage
        "company_exact_match": 0.25,      # Worked at EXACT company searched
        "company_related_match": 0.10,    # Worked at related company (subsidiary)
        "education_match": 0.12,          # Went to target school
        "location_exact": 0.15,           # EXACT city match
        "location_state": 0.06,           # Same state
        "location_country": 0.03,         # Same country
        "title_exact_match": 0.20,        # Title contains EXACT keywords
        "title_partial_match": 0.08,      # Title partially matches
        "experience_perfect_fit": 0.12,   # Experience EXACTLY in range
        "experience_close_fit": 0.05,     # Experience close to range
        "cert_match": 0.06,               # Has target certification
        "profile_completeness": 0.02,     # Profile quality bonus
        
        # Super bonuses for multiple exact matches
        "triple_match_bonus": 0.15,       # Skills + Location + Title all exact
        "double_match_bonus": 0.08,       # Two of the above exact
    }
    
    def rank(
        self,
        candidates: Dict[int, Dict[str, Any]],
        base_scores: Dict[int, float],
        query: HybridSearchQuery
    ) -> List[CandidateResult]:
        """
        Rank candidates by applying bonus scores.
        
        AGGRESSIVE scoring for exact matches - the goal is that
        someone who perfectly matches the query is #1, always.
        """
        results = []
        
        # Prepare query data for matching
        boost_skills_lower = set(s.lower() for s in (query.boost_skills or []))
        boost_companies_lower = set(c.lower() for c in (query.boost_companies or []))
        boost_schools_lower = set(s.lower() for s in (query.boost_schools or []))
        boost_titles_lower = set(t.lower() for t in (query.boost_titles or []))
        boost_certs_lower = set(c.lower() for c in (query.boost_certifications or []))
        
        # Location
        target_city = query.boost_location.city.lower() if query.boost_location and query.boost_location.city else None
        target_state = query.boost_location.state.lower() if query.boost_location and query.boost_location.state else None
        target_country = query.boost_location.country.lower() if query.boost_location and query.boost_location.country else None
        
        # Experience
        exp_min = query.experience_range.min_years if query.experience_range else None
        exp_max = query.experience_range.max_years if query.experience_range else None
        
        for forager_id, metadata in candidates.items():
            base_score = base_scores.get(forager_id, 0.5)
            
            # Calculate bonuses - track exact matches
            factors = RankingFactors(
                vector_similarity=base_score,
                skills_coverage=0.0,
                company_match_bonus=0.0,
                education_match_bonus=0.0,
                location_bonus=0.0,
                title_match_bonus=0.0,
                experience_fit_bonus=0.0,
                cert_match_bonus=0.0,
                profile_completeness_bonus=0.0
            )
            
            total_bonus = 0.0
            exact_matches = 0  # Count for combo bonus
            
            # 1. Skills coverage - AGGRESSIVE scoring
            if boost_skills_lower:
                profile_skills = set(s.lower() for s in metadata.get('skills', []))
                matched = boost_skills_lower & profile_skills
                num_matched = len(matched)
                num_required = len(boost_skills_lower)
                coverage = num_matched / num_required
                factors.skills_coverage = coverage
                
                if coverage == 1.0:  # PERFECT MATCH - all required skills
                    total_bonus += self.BONUSES["skills_perfect_match"]
                    exact_matches += 1
                elif coverage >= 0.5:  # Good coverage
                    total_bonus += coverage * self.BONUSES["skills_coverage"]
                else:  # Partial
                    total_bonus += coverage * self.BONUSES["skills_coverage"] * 0.5
            
            # 2. Company match - check for exact vs related
            if boost_companies_lower:
                profile_companies = set(c.lower() for c in metadata.get('companies', []))
                exact_company = bool(boost_companies_lower & profile_companies)
                
                if exact_company:
                    factors.company_match_bonus = 1.0
                    total_bonus += self.BONUSES["company_exact_match"]
                    exact_matches += 1
                else:
                    # Check for partial match (substring)
                    related_match = False
                    for target in boost_companies_lower:
                        for company in profile_companies:
                            if target in company or company in target:
                                related_match = True
                                break
                        if related_match:
                            break
                    if related_match:
                        factors.company_match_bonus = 0.5
                        total_bonus += self.BONUSES["company_related_match"]
            
            # 3. Education match
            if boost_schools_lower:
                profile_schools = set(s.lower() for s in metadata.get('schools', []))
                matched = False
                for target in boost_schools_lower:
                    for school in profile_schools:
                        if target in school or school in target:
                            matched = True
                            break
                    if matched:
                        break
                if matched:
                    factors.education_match_bonus = 1.0
                    total_bonus += self.BONUSES["education_match"]
            
            # 4. Location bonus - AGGRESSIVE for exact match
            profile_city = (metadata.get('city') or '').lower()
            profile_state = (metadata.get('state') or '').lower()
            profile_country = (metadata.get('country') or '').lower()
            
            if target_city:
                if profile_city == target_city:
                    factors.location_bonus = 1.0
                    total_bonus += self.BONUSES["location_exact"]
                    exact_matches += 1
                elif target_state and profile_state == target_state:
                    factors.location_bonus = 0.4
                    total_bonus += self.BONUSES["location_state"]
                elif target_country and profile_country == target_country:
                    factors.location_bonus = 0.2
                    total_bonus += self.BONUSES["location_country"]
            elif target_state:
                if profile_state == target_state:
                    factors.location_bonus = 0.5
                    total_bonus += self.BONUSES["location_state"]
            
            # 5. Title match - STRONG for exact match
            if boost_titles_lower:
                current_title = (metadata.get('current_title') or '').lower()
                exact_title_match = False
                partial_match = False
                
                for target in boost_titles_lower:
                    if target in current_title:
                        # Check if it's a strong match (most of the title)
                        if len(target) > len(current_title) * 0.5:
                            exact_title_match = True
                        else:
                            partial_match = True
                        break
                
                if exact_title_match:
                    factors.title_match_bonus = 1.0
                    total_bonus += self.BONUSES["title_exact_match"]
                    exact_matches += 1
                elif partial_match:
                    factors.title_match_bonus = 0.5
                    total_bonus += self.BONUSES["title_partial_match"]
            
            # 6. Experience fit - bonus for being in range
            years_exp = metadata.get('years_experience')
            if years_exp is not None and (exp_min is not None or exp_max is not None):
                in_range = True
                if exp_min is not None and years_exp < exp_min:
                    in_range = False
                if exp_max is not None and years_exp > exp_max:
                    in_range = False
                
                if in_range:
                    factors.experience_fit_bonus = 1.0
                    total_bonus += self.BONUSES["experience_perfect_fit"]
                else:
                    # Partial credit based on proximity
                    if exp_min is not None and exp_max is not None:
                        ideal = (exp_min + exp_max) / 2
                        diff = abs(years_exp - ideal)
                        proximity = max(0, 1 - diff / 8)  # Decay over 8 years
                        factors.experience_fit_bonus = proximity
                        total_bonus += proximity * self.BONUSES["experience_close_fit"]
            
            # 7. Certification match
            if boost_certs_lower:
                profile_certs = set(c.lower() for c in metadata.get('certifications', []))
                matched = False
                for target in boost_certs_lower:
                    for cert in profile_certs:
                        if target in cert or cert in target:
                            matched = True
                            break
                    if matched:
                        break
                if matched:
                    factors.cert_match_bonus = 1.0
                    total_bonus += self.BONUSES["cert_match"]
            
            # 8. Profile completeness
            completeness = metadata.get('profile_completeness', 0) or 0
            completeness_factor = completeness / 100.0  # Normalize to 0-1
            factors.profile_completeness_bonus = completeness_factor
            total_bonus += completeness_factor * self.BONUSES["profile_completeness"]
            
            # 9. COMBO BONUS - multiple exact matches = extra boost!
            # This is what makes "the exact candidate" rise to the top
            if exact_matches >= 3:
                total_bonus += self.BONUSES["triple_match_bonus"]
            elif exact_matches >= 2:
                total_bonus += self.BONUSES["double_match_bonus"]
            
            # Final score
            final_score = base_score + total_bonus
            
            # Build result
            result = CandidateResult(
                forager_id=forager_id,
                score=final_score,
                ranking_factors=factors,
                full_name=metadata.get('full_name', ''),
                current_title=metadata.get('current_title'),
                current_company=metadata.get('current_company'),
                city=metadata.get('city'),
                state=metadata.get('state'),
                country=metadata.get('country'),
                years_experience=metadata.get('years_experience'),
                domain=metadata.get('domain'),
                skills=metadata.get('skills', []),
                matched_skills=list(boost_skills_lower & set(s.lower() for s in metadata.get('skills', []))) if boost_skills_lower else []
            )
            
            results.append(result)
        
        # Sort by final score descending
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
