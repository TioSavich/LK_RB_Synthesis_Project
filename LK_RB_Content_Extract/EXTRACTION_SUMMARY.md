# L&N Content Extraction Summary

**Date:** 2025-11-03
**Source:** LK_RB_Synthesis_Project
**Destination:** LK_RB_Content_Extract

## What Was Extracted

### Python Files (10 files)

#### Core Theory (5 files)
1. **eple/domains/embodiment/object_manipulation.py** (113 lines)
   - Physical object manipulation as embodied foundation
   - Grounds "Arithmetic is Object Collection" metaphor

2. **eple/domains/embodiment/schemas.py** (88 lines)
   - Container schema and image schemas
   - Foundational pre-conceptual structures

3. **eple/domains/arithmetic/core.py** (86 lines)
   - "Arithmetic is Object Collection" metaphor definition
   - Demonstrates mathematical necessity from embodied constraints

4. **eple/core/metaphor.py**
   - Conceptual metaphor framework and mechanisms
   - Source-to-target domain mapping infrastructure

5. **eple/core/mua.py** (303 lines)
   - Meaning-Use Analysis framework
   - L&N + Brandom integration

#### Rich Implementations (5 files)
6. **eple/domains/arithmetic/strategies.py** (334 lines)
   - CountingOn and RMB strategies
   - Arithmetic procedures as metaphorical elaborations

7. **eple/domains/arithmetic/strategy_as_elaboration.py** (59 lines)
   - Links procedures to metaphors
   - Demonstrates cognitive grounding

8. **eple/domains/arithmetic/parsing.py** (57 lines)
   - Maps real child strategies to conceptual metaphors
   - Empirical grounding of theory

9. **eple/core/deontic_scorekeeper.py** (107 lines)
   - Incompatibility projection engine
   - Projects embodied constraints to abstract domains

10. **eple/domains/geometry.py**
    - Geometric image schemas
    - Spatial reasoning foundations

### Documentation (5 markdown files)

1. **docs/synthesis_lk_rb.md**
   - Core theoretical synthesis document
   - L&N + Brandom integration explained

2. **docs/Project_Overview.md**
   - EPLE architecture overview
   - Implementation roadmap

3. **docs/Metaphor_Knowledge_Base.md**
   - Four Grounding Metaphors taxonomy
   - Strategy-to-metaphor mappings

4. **docs/brandomian_analysis.md**
   - Brandom's inferentialism details
   - Pragmatic elaboration explained

5. **docs/lakoff_medium.md**
   - L&N theoretical summary
   - Key concepts from "Where Mathematics Comes From"

## What Was NOT Extracted

### Infrastructure (Excluded)
- `mud_generator.py` - MUD diagram generation
- `truncated_mud_generator.py` - Diagram generation
- `src/analysis/` - Analysis tools
- `scripts/` - Utility scripts
- `Python_Tests/` - Test scripts
- `src/automata/` - Specific algorithm implementations
- All `__pycache__/` directories
- Build and deployment files

### Rationale
The excluded files are implementation infrastructure, testing utilities, and diagram generation tools. The extraction focuses solely on files that encode Lakoff & Núñez theoretical concepts.

## Key L&N Concepts Captured

### The Four Grounding Metaphors
- ✅ Arithmetic is Object Collection
- ✅ Arithmetic is Object Construction
- ✅ Measuring Stick
- ✅ Arithmetic is Motion Along a Path

### Image Schemas
- ✅ Container Schema
- ✅ Part-Whole Schema
- ✅ Source-Path-Goal Schema

### Theoretical Mechanisms
- ✅ Conceptual metaphor as inference propagation
- ✅ Embodied constraints generating mathematical necessity
- ✅ Pragmatic elaboration (algorithmic vs. projective)
- ✅ Domain extension through metaphor
- ✅ Practice-Vocabulary relations (MUA)

### Cognitive Content
- ✅ Children's arithmetic strategies mapped to metaphors
- ✅ Progressive elaboration in mathematical development
- ✅ Embodied grounding of abstract mathematics

## File Statistics

```
Total Python Files: 10
Total Lines (Python): ~1,147 lines of L&N theory
Total Documentation: 5 markdown files
Total Size: ~10KB README + docs
```

## Directory Tree

```
LK_RB_Content_Extract/
├── README.md (10.5KB - comprehensive guide)
├── EXTRACTION_SUMMARY.md (this file)
├── docs/
│   ├── synthesis_lk_rb.md
│   ├── Project_Overview.md
│   ├── Metaphor_Knowledge_Base.md
│   ├── brandomian_analysis.md
│   └── lakoff_medium.md
└── eple/
    ├── core/
    │   ├── metaphor.py
    │   ├── mua.py
    │   └── deontic_scorekeeper.py
    └── domains/
        ├── embodiment/
        │   ├── object_manipulation.py
        │   └── schemas.py
        ├── arithmetic/
        │   ├── core.py
        │   ├── strategies.py
        │   ├── strategy_as_elaboration.py
        │   └── parsing.py
        └── geometry.py
```

## Next Steps

This extracted content is ready for:
1. ✅ Integration into a new logical framework
2. ✅ Reuse in alternative inference engines
3. ✅ Extension with additional L&N metaphors
4. ✅ Connection to different theorem provers
5. ✅ Use as reference implementation of L&N theory

The files are self-contained theoretical components that encode the cognitive structure of mathematical concepts according to Lakoff & Núñez, without being tied to the specific MUD generation or analysis infrastructure of the original project.
