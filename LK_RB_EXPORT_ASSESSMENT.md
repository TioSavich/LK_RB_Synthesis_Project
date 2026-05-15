# LK_RB Export Assessment for umedcta-formalization

Date: 2026-05-12

## Bottom Line

`LK_RB_Synthesis_Project` is not worth porting wholesale. Its Python scripts are a mix
of real conceptual scaffolding, exploratory stubs, and generated outputs whose
confidence scores overstate the evidence. But the repo does contain several strong
Brandom/Lakoff-Nunez bridge pieces that would make `umedcta-formalization` more
coherent if ported selectively.

The core finding is:

- The Python automaton analyzer has already been superseded by the Prolog
  `strategies/meta/automaton_analyzer.pl` pipeline.
- The first-class Brandom MUA layer has not really been ported.
- The arithmetic grounding metaphors have not really been ported as executable or
  queryable Prolog, even though the geometry-side L&N metaphor inventory is now strong.
- The learner has commitment/stress/reorganization machinery, but not the deontic
  scorekeeping distinction between commitment propagation and entitlement propagation.

So the export should be: preserve concepts, tables, and proof obligations; rewrite
the living pieces in Prolog; do not revive the Python pipeline as production code.

## What This Repo Contains

### 1. Narrative Synthesis

Useful files:

- `LK_RB_Synthesis/synthesis_lk_rb.md`
- `LK_RB_Synthesis/brandomian_analysis.md`
- `LK_RB_Synthesis/lakoff_medium.md`
- `LK_RB_Synthesis/Metaphor_Knowledge_Base.md`
- `LK_RB_Synthesis/Project_Overview.md`

These articulate the central thesis cleanly:

> Conceptual metaphors function as mechanisms of pragmatic elaboration
> (PP-sufficiency) that allow embodied practices to confer content on abstract
> mathematical vocabularies. Mathematical necessity is the explicit expression
> (LX) of constraints inherent in those embodied practices.

This thesis is still better stated here than in the current Prolog code. It should
be exported as a formalization agenda, not as prose decoration.

### 2. First-Class MUA Model

Useful source:

- `LK_RB_Synthesis/eple/core/mua.py`

Salvageable concepts:

- `Vocabulary`
- `Practice`
- `PP_Sufficiency`
- `PV_Sufficiency`
- `AlgorithmicElaboration`
- `PragmaticProjection`
- `find_pragmatic_metavocabulary`
- `is_LX`

Assessment:

The implementation is shallow, but the data model is right. The current
`umedcta-formalization` code uses Brandomian language throughout, but it does not
have first-class Prolog relations for MUA. The result is that claims like "RMB is
LX for Counting On" live in comments, docs, generated JSON, or informal interface
prose rather than in a queryable logical layer.

Destination:

- New Prolog module: `pml/mua_relations.pl`
- Tests: `pml/tests/test_mua_relations.pl`

Minimal predicates to port:

```prolog
vocabulary(Id, Description).
practice(Id, Description).
pv_sufficient(Practice, Vocabulary).
vp_sufficient(Vocabulary, Practice).
pp_sufficient(BasePractice, ElaboratedPractice, Mechanism).
pv_necessary(Practice, Vocabulary).
pragmatic_metavocabulary(MetaVocabulary, TargetVocabulary).
lx_for(ElaboratedVocabulary, BaseVocabulary).
```

The Prolog version should not copy the Python heuristics. It should define MUA
relations as explicit facts and small derivation rules.

### 3. Arithmetic Grounding Metaphors

Useful source:

- `LK_RB_Synthesis/Metaphor_Knowledge_Base.md`
- `LK_RB_Synthesis/eple/domains/arithmetic/core.py`
- `LK_RB_Synthesis/eple/domains/embodiment/object_manipulation.py`
- `LK_RB_Synthesis/eple/domains/embodiment/schemas.py`
- `LK_RB_Synthesis/eple/domains/arithmetic/parsing.py`

Assessment:

This is one of the most important gaps in `umedcta-formalization`. Geometry now has
substantial L&N Prolog coverage in `geometry/metaphors/lakoff_nunez_inventory.pl`
and `geometry/metaphors/measuring_stick.pl`. Arithmetic does not have an equivalent
queryable grounding-metaphor layer. It has grounded arithmetic operations and
strategy automata, but not the metaphorical mappings that explain why those
operations carry their inferential constraints.

The cleanest item to port is Arithmetic Is Object Collection:

```text
Combine   -> Add
Remainder -> Subtract
PartOf    -> LessThan
```

That mapping can justify why "take larger from smaller" is incoherent in the
natural-number object-collection domain and why the same operation becomes
available only after a metaphor/domain shift.

Destination options:

- `formalization/grounding_metaphors.pl` if the module is treated as part of
  arithmetic grounding.
- `pml/grounding_metaphors_arithmetic.pl` if the module is treated as
  Brandomian-metavocabulary infrastructure.

Recommendation:

Use `formalization/grounding_metaphors.pl` for the rules and add an interface doc
showing how PML reads those rules as MUA/LX structure.

### 4. Strategy Metadata and Brandomian PP-Nec/PP-Suff Tables

Useful source:

- `LK_RB_Synthesis/data/strategy_metadata.json`
- `LK_RB_Synthesis/brandomian_analysis.md`
- `LK_RB_Synthesis/Python_Tests/GEMINI_Hermeneutic_Calculator.md`

Assessment:

The metadata is much more valuable than the generated Python `analysis_results.json`.
It names PP-necessities, PP-sufficiencies, and the one explicit LX relation:
Rearranging to Make Bases elaborates Counting On by making associativity,
decomposition, and base-bridging explicit.

Current Prolog status:

- `strategies/meta/automaton_analyzer.pl` is already stronger than the Python
  analyzer.
- `docs/analysis/elaborations.pl` has queryable elaboration facts.
- But those facts are pattern-overlap facts, not Brandomian MUA facts.

Export target:

- New module: `strategies/strategy_mua.pl`
- Optional generated facts: `docs/analysis/strategy_mua_facts.pl`

Recommended predicates:

```prolog
strategy_practice(Strategy, Practice).
strategy_vocabulary(Strategy, Vocabulary).
pp_necessity(Strategy, Practice).
pp_sufficiency_component(Strategy, Practice).
strategy_lx_for(ElaboratedStrategy, BaseStrategy, ExplicitPrinciple).
strategy_grounding_metaphor(Strategy, Metaphor).
```

This should sit beside, not replace, `docs/analysis/elaborations.pl`.

### 5. Deontic Scorekeeping

Useful source:

- `LK_RB_Synthesis/eple/core/incompatibility_engine.py`
- `LK_RB_Synthesis/eple/core/deontic_scorekeeper.py`
- `LK_RB_Synthesis/eple/tests/test_deontic_scorekeeper.py`

Assessment:

`umedcta-formalization` already has proof, erasure, stress, and reorganization
machinery. It does not yet preserve the Brandomian distinction between:

- undertaking a commitment,
- being entitled to that commitment,
- propagating commitments through material inferences,
- propagating entitlements through material inferences,
- diagnosing incoherence when a commitment lacks entitlement.

That distinction matters for the user's worry. Without it, the system can model
crisis and proof failure, but it cannot cleanly model the game of giving and asking
for reasons.

Destination:

- `learner/deontic_scorekeeper.pl` or `arche-trace/deontic_scorekeeper.pl`

Recommendation:

Put the first version under `learner/` because it should interact with the ORR
cycle. Let `arche-trace/` remain the proof/erasure layer.

Minimal predicates:

```prolog
commitment(Agent, Proposition).
entitlement(Agent, Proposition).
undertake_commitment(Agent, Proposition).
grant_entitlement(Agent, Proposition).
commitment_consequence(Agent, Consequence).
entitlement_consequence(Agent, Consequence).
deontic_incoherent(Agent, Reason).
```

### 6. Python Automaton Analyzer

Useful source:

- `LK_RB_Synthesis/mud_generator.py`
- `LK_RB_Synthesis/output/analysis_results.json`
- `LK_RB_Synthesis/output/mud_diagrams.json`

Assessment:

Do not port this as code. It detects only two patterns in the saved output
(`base_decomposition`, `incremental_counting`) and assigns many `1.00` confidence
relationships from thin AST overlap. It was a useful prototype, but the Prolog
replacement is better:

- `strategies/meta/pattern_taxonomy.pl` has 12 named patterns.
- `strategies/meta/pattern_detectors.pl` records evidence.
- `strategies/meta/elaboration_detector.pl` calculates max and Jaccard confidence.
- `docs/analysis/elaborations.pl` and `.json` are generated outputs.

The only thing worth preserving from the Python analyzer is its MUD visual idiom:
strategy practices as P-nodes, vocabularies as V-nodes, and algorithmic
elaborations as PP-sufficiency arrows.

## What Has Already Been Exported Successfully

The following LK_RB ideas already exist in stronger form in `umedcta-formalization`:

- Grounded strategy automata: `strategies/math/*.pl`
- A richer strategy analyzer: `strategies/meta/*.pl`
- Queryable elaboration outputs: `docs/analysis/elaborations.pl` and `.json`
- Geometry-side L&N metaphor inventory: `geometry/metaphors/*.pl`
- Embodied prover and sequent calculus: `arche-trace/embodied_prover.pl`,
  `arche-trace/incompatibility_semantics.pl`
- Cross-layer compression/expansion interface:
  `interfaces/compression-across-layers.md`

These should be treated as successors, not recipients of copied Python code.

## Main Gaps in umedcta-formalization

### Gap 1: No Queryable MUA Layer

The project uses MUA vocabulary but lacks a Prolog MUA relation layer. That makes
analytic pragmatism a framing language rather than a working object.

Fix:

Build `pml/mua_relations.pl` and a small test suite.

### Gap 2: Arithmetic Grounding Metaphors Are Not First-Class

Geometry has L&N metaphor facts. Arithmetic mostly has grounded operations and
strategy FSMs. The object-collection, object-construction, measuring-stick, and
motion-path metaphors are documented but not queryable as the source of arithmetic
inferences.

Fix:

Build `formalization/grounding_metaphors.pl`, starting with AOC.

### Gap 3: Strategy Elaborations Are Pattern Facts, Not Meaning-Use Facts

The current elaboration graph says which strategies share transition patterns. It
does not say what a strategy makes explicit relative to another strategy.

Fix:

Build `strategies/strategy_mua.pl` and seed it from `strategy_metadata.json`.

### Gap 4: Deontic Scorekeeping Is Blurred with Crisis/Reorganization

The learner can detect stress and retract commitments, but commitment and entitlement
are not separated as statuses.

Fix:

Build a small deontic scorekeeper module that is driven by `material_inference/3`
and plugs into the ORR cycle.

## Recommended Export Sequence

1. **Port MUA relations first.**
   This gives the rest of the export a target ontology.

2. **Port Arithmetic Is Object Collection.**
   Use it as the proof-of-concept for metaphor as pragmatic projection.

3. **Add strategy MUA facts for Counting On, RMB, COBO, Chunking, Rounding, and Sliding.**
   These are the clearest bridges between LK_RB and the current strategy graph.

4. **Add deontic scorekeeping after the first three pieces exist.**
   Otherwise it will become a generic belief-revision module rather than a
   Brandomian one.

5. **Update the interface docs.**
   `interfaces/compression-across-layers.md` should be amended to say that
   compression is not merely lower cost; it is higher inferential commitment
   density per step. That is the Brandomian bridge.

## First Concrete Patch to Make

Create `pml/mua_relations.pl` with just enough machinery to prove:

```prolog
lx_for(v_rmb, v_counting_on).
```

The proof should depend on:

- `pp_sufficient(p_counting_on, p_rmb, algorithmic_elaboration)`
- `vp_sufficient(v_rmb, p_base_bridging)`
- `pv_necessary(p_base_bridging, v_counting_on)`
- `makes_explicit(v_rmb, p_base_bridging, associativity_and_decomposition)`

That one test would turn the central claim from the Python repo into a living
Prolog fact. Once that exists, AOC and the other grounding metaphors have a clear
place to attach.

## Items Not Worth Exporting

- `LK_RB_Synthesis/output/analysis_results.json`: superseded and too thin.
- `LK_RB_Synthesis/output/mud_diagrams.json`: useful only as visual inspiration.
- `LK_RB_Synthesis/mud_generator.py`: superseded by Prolog analyzer.
- Most one-off `Python_Tests/*.py`: strategy behavior already exists in Prolog.
- Generated LaTeX/PDF outputs: archive/reference only.

## Recommendation

Archive `LK_RB_Synthesis_Project` after extracting:

1. the MUA relation model,
2. the arithmetic 4G metaphor table,
3. the strategy PP-necessity/PP-sufficiency metadata,
4. the deontic commitment/entitlement distinction.

The current Prolog project is stronger computationally. What it lacks is not more
strategy code; it lacks a queryable Brandom/Lakoff-Nunez explanatory layer that
connects strategy execution, metaphorical grounding, and normative entitlement.
