"""
Configuration for Talentin Hybrid Search System
================================================
Supports local development and cloud production deployment.

Environment Variables:
  SEARCH_ENV=local|production
  QDRANT_URL=https://xxx.qdrant.io:6333
  QDRANT_API_KEY=your-api-key
  DUCKDB_PATH=/path/to/talent_search.duckdb
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables immediately
load_dotenv()


@dataclass
class Config:
    """Application configuration loaded from environment variables"""
    
    # === REQUIRED ===
    qdrant_url: str
    duckdb_path: str
    
    # === QDRANT ===
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "profiles_hybrid"  # Hybrid collection (dense + sparse)
    qdrant_collection_dense: str = "talent_profiles"  # Dense-only collection (legacy)
    
    # === EMBEDDING ===
    # Recommended models for talent search:
    # - "jinaai/jina-embeddings-v2-base-en" (768d) - Best for resumes/profiles
    # - "BAAI/bge-large-en-v1.5" (1024d) - Highest accuracy
    # - "sentence-transformers/all-MiniLM-L6-v2" (384d) - Fastest
    # - "sentence-transformers/all-mpnet-base-v2" (768d) - Current (general purpose)
    # CRITICAL: MUST match the model used to index vectors in Qdrant!
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_dim: int = 768  # Update to 1024 for bge-large, 384 for MiniLM
    
    # === SEARCH LIMITS ===
    default_limit: int = 50
    max_limit: int = 100
    prefetch_limit: int = 300  # How many to get from each vector type
    fusion_limit: int = 200   # How many after RRF fusion
    
    # === RANKING BONUSES (additive to base RRF score) ===
    # AGGRESSIVE BONUSES for smart search - exact matches dominate
    bonus_skills_coverage: float = 0.35    # Full coverage of required skills (was 0.20)
    bonus_company_match: float = 0.20      # Worked at target company (was 0.15)
    bonus_education_match: float = 0.15    # Went to target school (was 0.10)
    bonus_location_exact: float = 0.15     # Exact city match (was 0.08)
    bonus_location_nearby: float = 0.08    # Same state/country (was 0.04)
    bonus_title_match: float = 0.10        # Current title matches (was 0.05)
    bonus_experience_fit: float = 0.12     # Experience in ideal range (was 0.08)
    bonus_cert_match: float = 0.08         # Has target certification (was 0.05)
    bonus_profile_completeness: float = 0.03  # Profile quality (was 0.02)
    
    # === SKILL EXPANSION ===
    skill_expansion_min_similarity: float = 0.7  # Min similarity for skill expansion
    skill_expansion_max_related: int = 5         # Max related skills to expand
    
    @property
    def is_cloud(self) -> bool:
        """Check if running in cloud mode (Qdrant Cloud)"""
        return self.qdrant_api_key is not None
    
    @classmethod
    def load(cls) -> "Config":
        """Load configuration from environment variables"""
        
        env = os.getenv("SEARCH_ENV", "local").lower()
        base_dir = Path(__file__).parent.parent
        
        if env == "local":
            return cls(
                qdrant_url=os.getenv("QDRANT_URL", str(base_dir / "Database" / "qdrant_data")),
                qdrant_api_key=os.getenv("QDRANT_API_KEY"),  # Optional for local
                duckdb_path=os.getenv("DUCKDB_PATH", str(base_dir / "Database" / "talent_search.duckdb")),
                embedding_model=os.getenv("EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-base-en"),
                embedding_dim=int(os.getenv("EMBEDDING_DIM", "768")),
            )
        else:
            # Production - require all env vars
            qdrant_url = os.getenv("QDRANT_URL")
            if not qdrant_url:
                raise ValueError("QDRANT_URL environment variable is required in production")
            
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            if not qdrant_api_key:
                raise ValueError("QDRANT_API_KEY environment variable is required in production")
            
            # Strip whitespace to be safe
            qdrant_api_key = qdrant_api_key.strip()
            
            duckdb_path = os.getenv("DUCKDB_PATH")
            if not duckdb_path:
                raise ValueError("DUCKDB_PATH environment variable is required in production")
            
            return cls(
                qdrant_url=qdrant_url,
                qdrant_api_key=qdrant_api_key,
                duckdb_path=duckdb_path,
                embedding_model=os.getenv("EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-base-en"),
                embedding_dim=int(os.getenv("EMBEDDING_DIM", "768")),
            )


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        # Load .env without overriding existing environment variables
        load_dotenv(override=False)
        _config = Config.load()
    return _config


def reload_config() -> Config:
    """Force reload configuration"""
    global _config
    _config = Config.load()
    return _config
