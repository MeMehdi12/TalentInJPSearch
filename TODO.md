# JPSEARCH - Fix No-Skills Profiles in Top Results (Location Dominance)
Status: ✅ Plan Approved | ⏳ In Progress | ✅ Done

## Approved Plan Summary
- **Issue**: Profiles with empty skills rank high due to strong location boosts (+0.60) > weak skills penalty (-0.10).
- **Fix**: Tiered skills penalty + skills-from-text boost + location cap for low-skill profiles.
- **Target**: Top profiles must have 2+ skills OR strong text evidence.
- **Files**: search_api_v2.py (main), config.py (thresholds).

## Step-by-Step Implementation (Current: Step 1/6)

### ✅ Step 1: Add Config Thresholds [DONE]
- Edit `backend/config.py`: Added `MIN_SKILLS_REQUIRED=2`, `MIN_SKILLS_SCORE=0.20`, `skills_text_bonus=0.12`, `no_skills_penalty=-0.35`.
```
Status: ✅
```

### ✅ Step 2: Create Skills Check Util [DONE]
- Add `backend/search_service.py`: `has_minimum_skills(profile, min_count=2) → bool`.
```
Status: ✅
```

### ⏳ Step 3: Update smart_rerank Penalties [PENDING]
- `backend/search_api_v2.py`: 
  - No skills: -0.35 (was -0.10).
  - 1 skill: -0.20.
  - Skills-from-text bonus: +0.12 (was +0.06).
  - Location cap: max +0.40 if <2 skills.
```
Status: [ ]
```

### ⏳ Step 4: Add Hard Pre-Filter Option [PENDING]
- `build_qdrant_filter`: Add `skills: MatchAny(min_count=2)` if `query.min_skills_required >=2`.
```
Status: [ ]
```

### ✅ Step 5: Update Schema for min_skills [DONE]  
- `backend/search_schema.py`: Added `min_skills_required: int=2` to SearchOptionsV2 (default 2).  
```  
Status: ✅  
```  


### ⏳ Step 6: Test & Deploy [PENDING]
- Run `tests_integration.py`.
- Test query: 'react developer' + LA → verify top have skills.
- Deploy: `_deploy_final.ps1`.
```
Status: [ ]
```

**Next Action**: Complete Step 1 → Edit config.py → Mark ✅ → Proceed to Step 2.
**ETA**: 20 mins → Production deploy.

