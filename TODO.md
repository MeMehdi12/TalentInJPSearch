# Integration Endpoint - Production Readiness

## Plan
1. [x] Read and understand all relevant files
2. [x] Fix `title_matched` bug in `smart_rerank()` (search_api_v2.py)
3. [x] Add location grouping to integration endpoint (integration_api.py)
4. [x] Verify changes — syntax check passed, flow order confirmed

## Details

### Fix 1: `title_matched` bug (search_api_v2.py) ✅
- **Problem**: In `smart_rerank()`, section 1f references `title_matched` before it's assigned in section 3 → `NameError` at runtime
- **Fix**: Initialize `title_matched = False` at top of per-candidate loop (alongside `location_matched` and `exp_matched`)
- **Verified**: Variable now initialized before any reference

### Fix 2: Location grouping (integration_api.py) ✅
- **Problem**: Integration endpoint had no location tiering — results sorted purely by score after `smart_rerank()`
- **Fix**: Added `_integration_location_tier()` function + `reranked.sort(key=lambda r: (tier, -score))` after smart_rerank step
- **Tiers**:
  - Tier 0: exact city match (checks city, location, area, linkedin_area fields)
  - Tier 1: state match
  - Tier 2: country match
  - Tier 3: no location match
- **Scope**: Applies in `preferred` and `must_match` modes (not `remote`)
- **Flow position**: After smart_rerank (4a), before experience post-filter (4b)
- **Verified**: Correct insertion point, `original_city/state/country` variables available, syntax compiles clean

## Final Production Overview

### Integration Endpoint Flow (`POST /api/integration/search`)
1. **Auth** → API key validation (constant-time comparison)
2. **Parse** → Build `ParsedQueryV2` from request body
3. **Search** → `search_v2()` (Qdrant hybrid dense+sparse + DuckDB hydration)
4. **Rerank** → `smart_rerank()` with bonuses/penalties
5. **Location Group** → ✅ NEW: Tiered sort (city → state → country → other)
6. **Experience Filter** → Post-filter by min/max years
7. **Explicit Match Filter** → Filter by explicit_match fields
8. **Must-Match Location Filter** → Hard filter for must_match mode
9. **Paginate** → Slice by page/limit
10. **Hydrate** → Full profile data from DuckDB (no truncation)
11. **Response** → sampleresponse.json-shaped output

### Files Modified
| File | Change | Status |
|------|--------|--------|
| `backend/search_api_v2.py` | Initialize `title_matched = False` in `smart_rerank()` | ✅ Done |
| `backend/integration_api.py` | Add `_integration_location_tier()` + tiered sort | ✅ Done |

### Syntax Verification
- `backend/search_api_v2.py` — `py_compile`: ✅ PASS
- `backend/integration_api.py` — `py_compile`: ✅ PASS

### Risk Assessment
- **Low risk**: `title_matched` fix is a simple initialization — no behavioral change for existing logic
- **Low risk**: Location grouping only reorders results (no filtering/removal) — worst case is same order as before if no location specified
- **No breaking changes**: Integration endpoint is isolated from frontend API; frontend smart-search already has its own location grouping via `_location_match_tier()`
