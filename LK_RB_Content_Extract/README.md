# Lakoff & Núñez Theoretical Content - Extracted Core

This directory contains the richest **Lakoff & Núñez (L&N) theoretical content** extracted from the LK_RB_Synthesis project. These files encode the core concepts from *Where Mathematics Comes From*, including conceptual metaphors, image schemas, and embodied mathematics foundations.

## Purpose

This extraction isolates the theoretical L&N content from the implementation infrastructure (MUD generators, analysis tools, etc.) to facilitate reuse in a different logical framework. The files here represent the computational implementation of L&N's cognitive theory of mathematics.

## Directory Structure

```
LK_RB_Content_Extract/
├── eple/                           # Embodied Pragmatic Logic Engine core
│   ├── core/                       # Core theoretical frameworks
│   │   ├── metaphor.py            # Conceptual metaphor mechanisms
│   │   ├── mua.py                 # Meaning-Use Analysis (L&N + Brandom)
│   │   └── deontic_scorekeeper.py # Incompatibility projection engine
│   └── domains/                    # Domain-specific implementations
│       ├── embodiment/             # Embodied foundations
│       │   ├── object_manipulation.py  # Physical object manipulation
│       │   └── schemas.py              # Image schemas (Container, etc.)
│       ├── arithmetic/             # Arithmetic as elaborated domain
│       │   ├── core.py                 # "Arithmetic is Object Collection"
│       │   ├── strategies.py           # Arithmetic strategies as elaborations
│       │   ├── strategy_as_elaboration.py  # Strategy-metaphor linking
│       │   └── parsing.py              # Real strategies to metaphors
│       └── geometry.py             # Geometric image schemas
├── docs/                           # Theoretical documentation
│   ├── synthesis_lk_rb.md         # Core synthesis document
│   ├── Project_Overview.md        # EPLE architecture overview
│   ├── Metaphor_Knowledge_Base.md # Four Grounding Metaphors taxonomy
│   ├── brandomian_analysis.md     # Brandom integration details
│   └── lakoff_medium.md           # L&N theoretical summary
└── README.md                       # This file
```

## File Descriptions

### Top Tier - Core L&N Theory

#### 1. [eple/domains/embodiment/object_manipulation.py](eple/domains/embodiment/object_manipulation.py)
**The Embodied Foundation**
- Implements physical object manipulation as the source domain
- Defines predicates: `IsCollection`, `PartOf`, `Combine`, `Remainder`
- Encodes embodied constraints (e.g., "cannot take a whole from its part")
- Grounds the "Arithmetic is Object Collection" metaphor

**Key L&N Concepts:**
- Embodied cognition as the foundation for abstract mathematics
- Material inferences from physical manipulation
- Incompatibilities arising from physical constraints

#### 2. [eple/domains/embodiment/schemas.py](eple/domains/embodiment/schemas.py)
**Image Schemas**
- Implements the **Container Schema** (inside/outside, boundaries)
- Defines in-out incompatibility (cannot be inside AND outside)
- Transitivity of containment
- Pre-conceptual structures grounding logical reasoning

**Key L&N Concepts:**
- Image schemas as foundational cognitive structures
- Container schema grounding logical negation
- Physical experience structuring abstract thought

#### 3. [eple/domains/arithmetic/core.py](eple/domains/arithmetic/core.py)
**The Core Grounding Metaphor**
- Explicitly defines "Arithmetic is Object Collection" (AOC)
- Maps: `Combine → Add`, `Remainder → Subtract`, `PartOf → LessThan`
- Shows how arithmetic constraints emerge from metaphorical projection
- Demonstrates why `3 - 5` is impossible in natural numbers

**Key L&N Concepts:**
- Conceptual metaphor as inference propagation
- Grounding metaphors linking embodied and abstract domains
- Mathematical necessity from embodied constraints

#### 4. [eple/core/metaphor.py](eple/core/metaphor.py)
**Metaphor Mechanism Framework**
- Core infrastructure for conceptual metaphor implementation
- Defines source-to-target domain mappings
- Handles inference propagation across domains
- Implements metaphorical projection mechanisms

**Key L&N Concepts:**
- Conceptual metaphor theory formalized
- Domain mapping and structure preservation
- Cross-domain inference

#### 5. [eple/core/mua.py](eple/core/mua.py)
**Meta-Theoretical Framework**
- Integrates L&N with Brandom's pragmatism
- Defines Practice-Vocabulary relations
- Implements PP-Sufficiency (pragmatic elaboration)
- Models how embodied practices ground mathematical vocabularies

**Key L&N Concepts:**
- How "doing" (practices) grounds "saying" (vocabularies)
- Algorithmic vs. pragmatic elaboration
- LX (Elaborated-Explicating) relations

### Second Tier - Rich Implementations

#### 6. [eple/domains/arithmetic/strategies.py](eple/domains/arithmetic/strategies.py)
**Arithmetic Strategies as Elaborations**
- Implements CountingOn and RMB (Rearranging-to-Make-Bases) strategies
- Shows how arithmetic procedures elaborate metaphorical understanding
- Models cognitive development through progressive elaboration
- Demonstrates metaphor-to-algorithm pipeline

**Key L&N Concepts:**
- Children's arithmetic strategies as metaphorical elaborations
- Cognitive grounding of mathematical procedures
- Development through increasingly complex elaborations

#### 7. [eple/domains/arithmetic/strategy_as_elaboration.py](eple/domains/arithmetic/strategy_as_elaboration.py)
**Strategy-Metaphor Connection**
- Links procedural knowledge to embodied metaphors
- Shows prerequisite abilities for strategies
- Demonstrates algorithmic elaboration of metaphors
- Connects CountingOn to "Arithmetic is Object Collection"

**Key L&N Concepts:**
- Procedures require mastery of underlying metaphors
- Cognitive grounding of computational strategies
- Progressive elaboration in learning

#### 8. [eple/domains/arithmetic/parsing.py](eple/domains/arithmetic/parsing.py)
**Empirical Grounding**
- Maps real child arithmetic strategies to conceptual metaphors
- Links "counting on" to "Arithmetic is Motion Along a Path"
- Connects "chunking" to "Arithmetic is Object Construction"
- Grounds theory in empirical cognitive data

**Key L&N Concepts:**
- Real-world strategies cluster by metaphorical structure
- Empirical validation of metaphor theory
- Strategy-metaphor correspondence

#### 9. [eple/core/deontic_scorekeeper.py](eple/core/deontic_scorekeeper.py)
**Incompatibility Projection Engine**
- Projects embodied constraints to abstract domains
- Manages rule inheritance through metaphors
- Enforces logical constraints from physical impossibilities
- Implements deontic scorekeeping for commitments

**Key L&N Concepts:**
- Embodied constraints as mathematical incompatibilities
- Metaphorical projection of logical structure
- How physical impossibilities become logical necessities

#### 10. [eple/domains/geometry.py](eple/domains/geometry.py)
**Geometric Image Schemas**
- Geometric foundations for spatial reasoning
- Links to spatial image schemas
- Grounding for geometric metaphors

**Key L&N Concepts:**
- Spatial cognition grounding geometry
- Image schemas in geometric reasoning

## Key Lakoff & Núñez Concepts Encoded

### The Four Grounding Metaphors (4Gs)

1. **Arithmetic is Object Collection** - [core.py](eple/domains/arithmetic/core.py)
2. **Arithmetic is Object Construction** - [strategies.py](eple/domains/arithmetic/strategies.py)
3. **Measuring Stick** - [geometry.py](eple/domains/geometry.py)
4. **Arithmetic is Motion Along a Path** - [parsing.py](eple/domains/arithmetic/parsing.py)

### Image Schemas

- **Container Schema** - [schemas.py](eple/domains/embodiment/schemas.py)
- **Part-Whole Schema** - [object_manipulation.py](eple/domains/embodiment/object_manipulation.py)
- **Source-Path-Goal Schema** - Referenced in parsing and strategies

### Core Theoretical Claims

1. **Mathematical necessity arises from embodied constraints** - The impossibility of certain operations (like `3 - 5` in naturals) comes from physical impossibilities in the source domain
2. **Conceptual metaphors propagate inference** - The structure of reasoning in abstract math inherits from embodied source domains
3. **Mathematics develops through metaphorical elaboration** - New mathematical domains emerge when existing metaphors encounter contradictions

## Documentation

### Core Theory
- [docs/synthesis_lk_rb.md](docs/synthesis_lk_rb.md) - Complete theoretical synthesis of L&N with Brandom's pragmatism
- [docs/Metaphor_Knowledge_Base.md](docs/Metaphor_Knowledge_Base.md) - Detailed taxonomy of the Four Grounding Metaphors with strategy mappings
- [docs/lakoff_medium.md](docs/lakoff_medium.md) - Summary of L&N theoretical framework

### Architecture
- [docs/Project_Overview.md](docs/Project_Overview.md) - EPLE system architecture and implementation plan
- [docs/brandomian_analysis.md](docs/brandomian_analysis.md) - Integration with Brandom's inferentialism

## What's NOT Included

This extraction **excludes** implementation infrastructure:
- MUD (Meaning-Use Diagram) generators
- Analysis and reporting tools
- Test scripts and automata implementations
- Visualization and LaTeX generation
- Build and deployment scripts

## Usage

These files are designed to be **reusable theoretical components**. They encode:
- The cognitive structure of mathematical concepts
- Metaphorical mappings between domains
- Embodied foundations for abstract reasoning
- Inference propagation mechanisms

You can integrate these into a different logical framework while preserving the L&N theoretical content.

## Dependencies

The core files have minimal dependencies:
- Standard Python libraries
- Basic data structures (dictionaries, lists, sets)
- No heavy framework dependencies

Some files may reference each other (e.g., `core.py` references `object_manipulation.py`), but these are conceptual dependencies that show the metaphorical relationships.

## Further Reading

For the complete L&N theory, see:
- Lakoff, G., & Núñez, R. (2000). *Where Mathematics Comes From: How the Embodied Mind Brings Mathematics into Being*

For Brandom's pragmatism, see:
- Brandom, R. (2008). *Between Saying and Doing: Towards an Analytic Pragmatism*

---

**Extracted:** 2025-11-03
**Source:** LK_RB_Synthesis_Project
**Purpose:** Isolate L&N theoretical content for reuse in alternative logical frameworks
