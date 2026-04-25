# Algorithmic Elaboration Analysis Report

Generated on: 2025-10-06 15:15:50

## Overview

- **Computational Patterns Detected**: 2
- **Algorithmic Elaborations Found**: 16
- **MUD Diagrams Generated**: 3

## Computational Patterns

| Pattern | Type | Usage Count | Strategies |
|---------|------|-------------|------------|
| base_decomposition | decomposition | 4 | SMR_DIV_DealingByOnes, ADD_Rounding, SMR_DIV_CGOB... |
| incremental_counting | counting | 5 | ADD_Chunking, SMR_MULT_C2C, ADD_Counting... |

## Key Algorithmic Elaborations

| Base Strategy | Elaborated Strategy | Shared Patterns | Confidence |
|---------------|---------------------|----------------|------------|
| ADD_Rounding | SUB_Rounding | base_decomposition | 1.00 |
| ADD_Rounding | SMR_DIV_CGOB | base_decomposition | 1.00 |
| ADD_Rounding | SMR_DIV_DealingByOnes | base_decomposition | 1.00 |
| ADD_Counting | ADD_Chunking | incremental_counting | 1.00 |
| ADD_Counting | ADD_COBO | incremental_counting | 1.00 |
| ADD_Counting | COBO | incremental_counting | 1.00 |
| ADD_Counting | SMR_MULT_C2C | incremental_counting | 1.00 |
| ADD_Chunking | ADD_COBO | incremental_counting | 1.00 |
| ADD_Chunking | COBO | incremental_counting | 1.00 |
| ADD_Chunking | SMR_MULT_C2C | incremental_counting | 1.00 |

## Meaning-Use Diagrams

The following diagrams illustrate the algorithmic elaborations detected in the analysis. Each diagram shows strategies connected by shared computational patterns.

### Addition

**Operation:** Addition
**Strategies Analyzed:** 4
**Elaborations Detected:** 6

#### Strategies:
- ADD_Chunking
- ADD_Counting
- COBO
- ADD_COBO

#### Key Elaborations:
- **ADD_Counting** → **ADD_Chunking**
  - Shared patterns: incremental_counting
  - Confidence: 1.00
- **ADD_Counting** → **ADD_COBO**
  - Shared patterns: incremental_counting
  - Confidence: 1.00
- **ADD_Counting** → **COBO**
  - Shared patterns: incremental_counting
  - Confidence: 1.00
- **ADD_Chunking** → **ADD_COBO**
  - Shared patterns: incremental_counting
  - Confidence: 1.00
- **ADD_Chunking** → **COBO**
  - Shared patterns: incremental_counting
  - Confidence: 1.00
- ... and 1 more elaborations

#### TikZ Diagram Code:

```latex
\begin{tikzpicture}[
  % Node Styles
  pnode/.style={rectangle, rounded corners=5pt, draw, fill=gray!70, text=black, minimum height=1.3cm, minimum width=3.5cm, align=center, inner xsep=0.3cm, inner ysep=0.2cm},
  graybox/.style={rectangle, fill=lightgray!50, inner sep=4pt, minimum height=0.8cm, anchor=center, align=center, text centered, font=\tiny},
  % Arrow Styles
  solidarrow/.style={-Stealth, thick},
]
\tikzset{font=\small}

% MUD Diagram for: Addition

\node[pnode] (P_ADD_Counting) at (0.00,0.00) {P\textsubscript{ADD_Counting}};
\node[pnode] (P_ADD_Chunking) at (0.00,-4.00) {P\textsubscript{ADD_Chunking}};
\node[pnode] (P_ADD_COBO) at (0.00,-8.00) {P\textsubscript{ADD_COBO}};
\node[pnode] (P_COBO) at (0.00,-12.00) {P\textsubscript{COBO}};

\draw[solidarrow] (P_ADD_Counting) -- node[graybox, midway, fill=white] {1. incremental\_counting} (P_ADD_Chunking);
\draw[solidarrow] (P_ADD_Counting) -- node[graybox, midway, fill=white] {2. incremental\_counting} (P_ADD_COBO);
\draw[solidarrow] (P_ADD_Counting) -- node[graybox, midway, fill=white] {3. incremental\_counting} (P_COBO);
\draw[solidarrow] (P_ADD_Chunking) -- node[graybox, midway, fill=white] {4. incremental\_counting} (P_ADD_COBO);
\draw[solidarrow] (P_ADD_Chunking) -- node[graybox, midway, fill=white] {5. incremental\_counting} (P_COBO);
\draw[solidarrow] (P_ADD_COBO) -- node[graybox, midway, fill=white] {6. incremental\_counting} (P_COBO);

\end{tikzpicture}
```

---
### Miscellaneous

**Operation:** Miscellaneous
**Strategies Analyzed:** 2
**Elaborations Detected:** 1

#### Strategies:
- SMR_MULT_C2C
- COBO

#### Key Elaborations:
- **COBO** → **SMR_MULT_C2C**
  - Shared patterns: incremental_counting
  - Confidence: 1.00

#### TikZ Diagram Code:

```latex
\begin{tikzpicture}[
  % Node Styles
  pnode/.style={rectangle, rounded corners=5pt, draw, fill=gray!70, text=black, minimum height=1.3cm, minimum width=3.5cm, align=center, inner xsep=0.3cm, inner ysep=0.2cm},
  graybox/.style={rectangle, fill=lightgray!50, inner sep=4pt, minimum height=0.8cm, anchor=center, align=center, text centered, font=\tiny},
  % Arrow Styles
  solidarrow/.style={-Stealth, thick},
]
\tikzset{font=\small}

% MUD Diagram for: Miscellaneous

\node[pnode] (P_COBO) at (0.00,0.00) {P\textsubscript{COBO}};
\node[pnode] (P_SMR_MULT_C2C) at (0.00,-4.00) {P\textsubscript{SMR_MULT_C2C}};

\draw[solidarrow] (P_COBO) -- node[graybox, midway, fill=white] {1. incremental\_counting} (P_SMR_MULT_C2C);

\end{tikzpicture}
```

---
### Division

**Operation:** Division
**Strategies Analyzed:** 2
**Elaborations Detected:** 1

#### Strategies:
- SMR_DIV_DealingByOnes
- SMR_DIV_CGOB

#### Key Elaborations:
- **SMR_DIV_CGOB** → **SMR_DIV_DealingByOnes**
  - Shared patterns: base_decomposition
  - Confidence: 1.00

#### TikZ Diagram Code:

```latex
\begin{tikzpicture}[
  % Node Styles
  pnode/.style={rectangle, rounded corners=5pt, draw, fill=gray!70, text=black, minimum height=1.3cm, minimum width=3.5cm, align=center, inner xsep=0.3cm, inner ysep=0.2cm},
  graybox/.style={rectangle, fill=lightgray!50, inner sep=4pt, minimum height=0.8cm, anchor=center, align=center, text centered, font=\tiny},
  % Arrow Styles
  solidarrow/.style={-Stealth, thick},
]
\tikzset{font=\small}

% MUD Diagram for: Division

\node[pnode] (P_SMR_DIV_CGOB) at (0.00,0.00) {P\textsubscript{SMR_DIV_CGOB}};
\node[pnode] (P_SMR_DIV_DealingByOnes) at (0.00,-4.00) {P\textsubscript{SMR_DIV_DealingByOnes}};

\draw[solidarrow] (P_SMR_DIV_CGOB) -- node[graybox, midway, fill=white] {1. base\_decomposition} (P_SMR_DIV_DealingByOnes);

\end{tikzpicture}
```

---
