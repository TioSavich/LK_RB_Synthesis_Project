# Update Notes - File Corrections

## Issue Found
The original extraction included an empty `metaphor.py` file with 0 lines of content.

## Corrections Made

### Files Added
1. **eple/core/logic_terms.py** (132 lines)
   - Foundational logical term representations
   - Defines `Term`, `Var`, `Atom`, `Predicate` classes
   - Essential for all other L&N implementations

2. **eple/domains/arithmetic/base.py** (51 lines)
   - Base arithmetic vocabulary (non-metaphorical)
   - Defines `P_Arithmetic` and `V_Arithmetic`
   - Foundation for metaphorical elaborations

3. **eple/core/incompatibility_engine.py** (162 lines)
   - Core logic engine with deontic scorekeeping
   - Material inference implementation
   - Brandomian commitment/entitlement tracking

### Files Removed
1. **eple/core/metaphor.py** - Empty file (0 lines)
   - **Note:** The metaphor functionality (`PragmaticProjection` class) is actually defined inside `mua.py`, not in a separate file

## Current File Count

### Python Files: 12 total

**Core Framework (4 files)**
1. `eple/core/logic_terms.py` - 132 lines
2. `eple/core/mua.py` - 303 lines (includes `PragmaticProjection`)
3. `eple/core/deontic_scorekeeper.py` - 107 lines
4. `eple/core/incompatibility_engine.py` - 162 lines

**Embodied Foundations (2 files)**
5. `eple/domains/embodiment/object_manipulation.py` - 113 lines
6. `eple/domains/embodiment/schemas.py` - 88 lines

**Arithmetic Domain (5 files)**
7. `eple/domains/arithmetic/base.py` - 51 lines
8. `eple/domains/arithmetic/core.py` - 86 lines
9. `eple/domains/arithmetic/strategies.py` - 334 lines
10. `eple/domains/arithmetic/strategy_as_elaboration.py` - 59 lines
11. `eple/domains/arithmetic/parsing.py` - 57 lines

**Geometry (1 file)**
12. `eple/domains/geometry.py`

### Documentation Files: 5 markdown files
1. `docs/synthesis_lk_rb.md`
2. `docs/Project_Overview.md`
3. `docs/Metaphor_Knowledge_Base.md`
4. `docs/brandomian_analysis.md`
5. `docs/lakoff_medium.md`

## Total Lines of Code
**~1,492 lines** of L&N theoretical content (up from ~1,147)

## Key Clarification

### Where is the Metaphor Implementation?

The conceptual metaphor mechanism is **NOT** in a separate `metaphor.py` file. Instead:

- **`PragmaticProjection` class** is defined in `eple/core/mua.py` (lines 206+)
- This class implements metaphorical mappings from source to target domains
- It handles inference propagation through metaphorical projection
- Used by `ArithmeticIsObjectCollection` in `arithmetic/core.py`

Example usage from `arithmetic/core.py`:
```python
from eple.core.mua import PragmaticProjection

ArithmeticIsObjectCollection = PragmaticProjection(
    source_practice=P_ObjectManipulation,
    target_practice=P_ArithmeticAsObjectCollection,
    mappings={
        "Combine": "Add",
        "Remainder": "Subtract",
        # ... more mappings
    }
)
```

## Complete and Functional

The extraction now includes all necessary dependencies:
- ✅ Logical term infrastructure
- ✅ Base vocabularies and practices
- ✅ Metaphor mechanism (in mua.py)
- ✅ Embodied foundations
- ✅ Arithmetic elaborations
- ✅ Complete documentation

All files should now be usable in a new project without missing dependencies.
