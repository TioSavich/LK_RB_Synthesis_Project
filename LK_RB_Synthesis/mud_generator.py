#!/usr/bin/env python3
"""
mud_generator.py: Consolidated Meaning-Use Diagram (MUD) Generator

This script provides a unified interface for:
1. Automated discovery of algorithmic elaborations from automaton code
2. Generation of professional TikZ diagrams following Brandom's conventions
3. Multiple output formats (TikZ, JSON, reports)

Usage:
    python mud_generator.py analyze --automata-dir src/automata
    python mud_generator.py generate --input automated_mud_results.json --output diagrams/
    python mud_generator.py report --input automated_mud_results.json --format latex
"""

import os
import sys
import json
import ast
import inspect
from typing import Dict, List, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import argparse

@dataclass
class ComputationalPattern:
    """Represents a detected computational pattern/subroutine."""
    name: str
    operation_type: str  # 'counting', 'decomposition', 'adjustment', etc.
    register_operations: List[str]
    state_transitions: List[str]
    strategies_using: Set[str] = field(default_factory=set)

@dataclass
class AlgorithmicElaboration:
    """Represents an algorithmic elaboration relationship."""
    base_strategy: str
    elaborated_strategy: str
    shared_patterns: Set[str]
    elaboration_type: str  # 'intra_categorial', 'inter_categorial'
    confidence: float

class AutomatonAnalyzer:
    """Analyzes automaton implementations to detect patterns and relationships."""

    def __init__(self, automata_dir: str):
        self.automata_dir = automata_dir
        self.patterns: Dict[str, ComputationalPattern] = {}
        self.elaborations: List[AlgorithmicElaboration] = []
        self.strategy_patterns: Dict[str, Set[str]] = defaultdict(set)

    def analyze_all_automata(self) -> Dict[str, Any]:
        """Main analysis pipeline."""
        print("🔬 Starting Automated Automaton Analysis")
        print("=" * 50)

        # Step 1: Extract patterns from all automata
        self._extract_patterns_from_automata()

        # Step 2: Detect algorithmic elaborations
        self._detect_elaborations()

        # Step 3: Generate analysis report
        return self._generate_analysis_report()

    def _extract_patterns_from_automata(self):
        """Extract computational patterns from automaton source code."""
        print("\n📋 Extracting Computational Patterns...")

        for operation_dir in ['addition', 'subtraction', 'multiplication', 'division']:
            op_path = os.path.join(self.automata_dir, operation_dir)
            if not os.path.exists(op_path):
                continue

            for filename in os.listdir(op_path):
                if filename.endswith('.py') and not filename.startswith('__'):
                    strategy_id = filename.replace('.py', '').replace('SAR_', '')
                    filepath = os.path.join(op_path, filename)

                    try:
                        patterns = self._analyze_single_automaton(filepath, strategy_id, operation_dir)
                        self.strategy_patterns[strategy_id] = patterns
                        print(f"✅ Analyzed {strategy_id}: {len(patterns)} patterns found")
                    except Exception as e:
                        print(f"❌ Error analyzing {strategy_id}: {e}")

    def _analyze_single_automaton(self, filepath: str, strategy_id: str, operation: str) -> Set[str]:
        """Analyze a single automaton file to extract patterns."""
        with open(filepath, 'r') as f:
            source_code = f.read()

        # Parse the AST
        tree = ast.parse(source_code)

        patterns_found = set()
        register_ops = []
        state_methods = []

        # Extract class definition and methods
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Look for execute_ methods (state handlers)
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name.startswith('execute_'):
                        state_name = item.name.replace('execute_', '')
                        operations = self._extract_operations_from_method(item)
                        register_ops.extend(operations)

                        # Detect specific patterns
                        patterns = self._detect_patterns_in_method(item, state_name, operations)
                        patterns_found.update(patterns)

                        # Update pattern usage
                        for pattern_name in patterns:
                            if pattern_name not in self.patterns:
                                self.patterns[pattern_name] = ComputationalPattern(
                                    name=pattern_name,
                                    operation_type=self._classify_pattern(pattern_name),
                                    register_operations=[],
                                    state_transitions=[]
                                )
                            self.patterns[pattern_name].strategies_using.add(strategy_id)

        return patterns_found

    def _extract_operations_from_method(self, method_node: ast.FunctionDef) -> List[str]:
        """Extract register operations from a method, including conditionals."""
        operations = []

        for node in ast.walk(method_node):
            # Regular assignments
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        operations.append(f"{target.id} = {self._extract_value(node.value)}")

            # Augmented assignments (x += 1, etc.)
            elif isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name):
                    target = node.target.id
                    if isinstance(node.op, ast.Add):
                        operations.append(f"{target} += {self._extract_value(node.value)}")
                    elif isinstance(node.op, ast.Sub):
                        operations.append(f"{target} -= {self._extract_value(node.value)}")

            # Function calls that might be transitions or operations
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == 'transition':
                        operations.append(f"transition: {self._extract_call_args(node)}")
                    elif node.func.attr == '_record_history':
                        operations.append(f"record_history: {self._extract_call_args(node)}")

        return operations

    def _extract_value(self, node: ast.AST) -> str:
        """Extract value from AST node."""
        if isinstance(node, ast.Constant):  # Python 3.8+
            return str(node.value)
        elif isinstance(node, ast.Num):  # Legacy support
            return str(node.n)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{node.attr}"
        elif isinstance(node, ast.BinOp):
            left = self._extract_value(node.left)
            right = self._extract_value(node.right)
            if isinstance(node.op, ast.Add):
                return f"{left} + {right}"
            elif isinstance(node.op, ast.Sub):
                return f"{left} - {right}"
            elif isinstance(node.op, ast.Mult):
                return f"{left} * {right}"
            elif isinstance(node.op, ast.Div):
                return f"{left} // {right}"  # Integer division common in automata
            elif isinstance(node.op, ast.Mod):
                return f"{left} % {right}"
        return "complex_expr"

    def _extract_call_args(self, call_node: ast.Call) -> str:
        """Extract arguments from a function call."""
        args = []
        for arg in call_node.args:
            if isinstance(arg, ast.Constant):  # Python 3.8+
                args.append(f"'{arg.value}'")
            elif isinstance(arg, ast.Str):  # Legacy support
                args.append(f"'{arg.s}'")
            elif isinstance(arg, ast.Name):
                args.append(arg.id)
            elif isinstance(arg, ast.Num):  # Legacy support
                args.append(str(arg.n))
            else:
                args.append("expr")
        return ", ".join(args)

    def _detect_patterns_in_method(self, method_node: ast.FunctionDef, state_name: str, operations: List[str]) -> Set[str]:
        """Detect computational patterns in a method."""
        patterns = set()

        # Get the source code for more detailed analysis
        method_source = self._get_method_source(method_node)

        # Pattern 1: Counting loops (state-based iteration)
        if self._is_counting_loop_pattern(method_source, operations):
            patterns.add("counting_loop")

        # Pattern 2: Base decomposition
        if self._is_decomposition_pattern(operations) or '//' in method_source or '%' in method_source:
            patterns.add("base_decomposition")

        # Pattern 3: Adjustment calculations
        if self._is_adjustment_pattern(operations) or 'TargetBase' in method_source or 'K =' in method_source:
            patterns.add("value_adjustment")

        # Pattern 4: Iterative addition/subtraction
        if self._is_iterative_arithmetic(operations) or 'Sum += ' in method_source or 'Current += ' in method_source:
            patterns.add("iterative_arithmetic")

        # Pattern 5: State-based counting transitions
        if self._is_state_based_counting(state_name, method_source):
            patterns.add("incremental_counting")

        # Pattern 6: Decomposition and reconstruction
        if self._is_decomposition_reconstruction_pattern(method_source):
            patterns.add("decomposition_reconstruction")

        return patterns

    def _get_method_source(self, method_node: ast.FunctionDef) -> str:
        """Extract source code from method node."""
        # This is a simplified approach - in practice you'd need line numbers
        # For now, we'll reconstruct from operations and state name
        return " ".join([str(op) for op in self._extract_operations_from_method(method_node)])

    def _is_counting_loop_pattern(self, method_source: str, operations: List[str]) -> bool:
        """Detect state-based counting loops."""
        # Look for patterns that indicate iterative counting
        has_counter = any('Count' in op for op in operations)
        has_increment = any('+=' in op for op in operations)
        has_comparison = '<' in method_source or '>' in method_source
        has_conditional = 'if' in method_source or 'while' in method_source

        return has_counter and has_increment and (has_comparison or has_conditional)

    def _is_decomposition_pattern(self, operations: List[str]) -> bool:
        """Detect base decomposition patterns."""
        return any('//' in op or '%' in op for op in operations)

    def _is_adjustment_pattern(self, operations: List[str]) -> bool:
        """Detect value adjustment patterns."""
        return any('TargetBase' in op or 'K =' in op for op in operations)

    def _is_iterative_arithmetic(self, operations: List[str]) -> bool:
        """Detect iterative arithmetic patterns."""
        return any('Sum += ' in op or 'Current += ' in op for op in operations)

    def _is_state_based_counting(self, state_name: str, method_source: str) -> bool:
        """Detect state-based counting patterns."""
        counting_states = ['inc_tens', 'inc_hundreds', 'add_bases', 'add_ones', 'loop_K', 'count']
        return any(state in state_name.lower() for state in counting_states)

    def _is_decomposition_reconstruction_pattern(self, method_source: str) -> bool:
        """Detect patterns that decompose and reconstruct values."""
        return ('//' in method_source and '%' in method_source) or \
               ('BaseCounter' in method_source and 'OneCounter' in method_source)

    def _classify_pattern(self, pattern_name: str) -> str:
        """Classify a pattern by its computational type."""
        classifications = {
            "counting_loop": "counting",
            "base_decomposition": "decomposition",
            "value_adjustment": "adjustment",
            "iterative_arithmetic": "arithmetic",
            "incremental_counting": "counting"
        }
        return classifications.get(pattern_name, "general")

    def _detect_elaborations(self):
        """Detect algorithmic elaborations based on shared patterns."""
        print("\n🔗 Detecting Algorithmic Elaborations...")

        strategy_list = list(self.strategy_patterns.keys())

        for i, strategy_a in enumerate(strategy_list):
            for strategy_b in strategy_list[i+1:]:
                shared_patterns = self.strategy_patterns[strategy_a] & self.strategy_patterns[strategy_b]

                if shared_patterns:
                    # Determine operation types
                    op_a = self._get_operation_type(strategy_a)
                    op_b = self._get_operation_type(strategy_b)

                    elaboration_type = "intra_categorial" if op_a == op_b else "inter_categorial"
                    confidence = len(shared_patterns) / max(len(self.strategy_patterns[strategy_a]),
                                                          len(self.strategy_patterns[strategy_b]))

                    # Determine elaboration direction based on pattern complexity
                    base_strategy, elab_strategy = self._determine_elaboration_direction(
                        strategy_a, strategy_b, shared_patterns
                    )

                    elaboration = AlgorithmicElaboration(
                        base_strategy=base_strategy,
                        elaborated_strategy=elab_strategy,
                        shared_patterns=shared_patterns,
                        elaboration_type=elaboration_type,
                        confidence=confidence
                    )

                    self.elaborations.append(elaboration)

    def _get_operation_type(self, strategy_id: str) -> str:
        """Determine operation type from strategy ID."""
        if any(keyword in strategy_id.upper() for keyword in ['ADD', 'COUNTING']):
            return 'addition'
        elif any(keyword in strategy_id.upper() for keyword in ['SUB', 'SLIDING']):
            return 'subtraction'
        elif any(keyword in strategy_id.upper() for keyword in ['MULT', 'CBO']):
            return 'multiplication'
        elif any(keyword in strategy_id.upper() for keyword in ['DIV', 'DEALING']):
            return 'division'
        return 'unknown'

    def _determine_elaboration_direction(self, strategy_a: str, strategy_b: str, shared_patterns: Set[str]) -> Tuple[str, str]:
        """Determine which strategy elaborates which based on pattern analysis."""
        # Simple heuristic: strategy with fewer unique patterns is the base
        unique_a = len(self.strategy_patterns[strategy_a] - shared_patterns)
        unique_b = len(self.strategy_patterns[strategy_b] - shared_patterns)

        if unique_a <= unique_b:
            return strategy_a, strategy_b
        else:
            return strategy_b, strategy_a

    def _generate_analysis_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        print(f"\n📊 Analysis Complete:")
        print(f"   • {len(self.patterns)} computational patterns detected")
        print(f"   • {len(self.elaborations)} algorithmic elaborations identified")

        return {
            "patterns": {
                name: {
                    "type": pattern.operation_type,
                    "strategies_using": list(pattern.strategies_using),
                    "usage_count": len(pattern.strategies_using)
                }
                for name, pattern in self.patterns.items()
            },
            "elaborations": [
                {
                    "base_strategy": elab.base_strategy,
                    "elaborated_strategy": elab.elaborated_strategy,
                    "shared_patterns": list(elab.shared_patterns),
                    "type": elab.elaboration_type,
                    "confidence": elab.confidence
                }
                for elab in self.elaborations
            ],
            "strategy_patterns": {
                strategy: list(patterns)
                for strategy, patterns in self.strategy_patterns.items()
            }
        }

class MUDGenerator:
    """Generates MUD diagrams from algorithmic elaboration analysis."""

    def __init__(self, analysis_results: Dict[str, Any]):
        self.analysis_results = analysis_results
        self.mud_diagrams = {}

    def generate_mud_diagrams(self) -> Dict[str, Any]:
        """Generate MUD diagrams for all discovered elaborations."""
        print("🎨 Generating Meaning-Use Diagrams")
        print("=" * 50)

        # Group elaborations by operation type
        operation_groups = self._group_elaborations_by_operation()

        for operation, elaborations in operation_groups.items():
            print(f"\n📊 Generating MUD for {operation}")
            mud_diagram = self._generate_operation_mud(operation, elaborations)
            self.mud_diagrams[operation] = mud_diagram

        return self.mud_diagrams

    def _group_elaborations_by_operation(self) -> Dict[str, List[Dict]]:
        """Group elaborations by their primary operation type."""
        operation_groups = defaultdict(list)

        for elab in self.analysis_results.get('elaborations', []):
            # Determine operation from strategy names
            base_op = self._extract_operation_type(elab['base_strategy'])
            elab_op = self._extract_operation_type(elab['elaborated_strategy'])

            # Use the more specific operation if they differ
            if base_op != elab_op:
                operation = f"{base_op}_to_{elab_op}"
            else:
                operation = base_op

            operation_groups[operation].append(elab)

        return operation_groups

    def _extract_operation_type(self, strategy_id: str) -> str:
        """Extract operation type from strategy ID."""
        strategy_upper = strategy_id.upper()

        if any(op in strategy_upper for op in ['ADD', 'COUNTING']):
            return 'addition'
        elif any(op in strategy_upper for op in ['SUB', 'SLIDING']):
            return 'subtraction'
        elif any(op in strategy_upper for op in ['MULT', 'CBO']):
            return 'multiplication'
        elif any(op in strategy_upper for op in ['DIV', 'DEALING']):
            return 'division'
        else:
            return 'general'

    def _generate_operation_mud(self, operation: str, elaborations: List[Dict]) -> Dict[str, Any]:
        """Generate a MUD diagram for a specific operation."""
        # Collect all strategies involved
        strategies = set()
        pattern_relationships = defaultdict(list)

        for elab in elaborations:
            strategies.add(elab['base_strategy'])
            strategies.add(elab['elaborated_strategy'])

            # Group by shared patterns
            for pattern in elab['shared_patterns']:
                pattern_relationships[pattern].append(elab)

        # Generate TikZ code
        tikz_code = self._generate_tikz_diagram(operation, list(strategies), pattern_relationships)

        return {
            'operation': operation,
            'strategies': list(strategies),
            'elaborations': elaborations,
            'pattern_relationships': dict(pattern_relationships),
            'tikz_diagram': tikz_code,
            'summary': self._generate_mud_summary(operation, elaborations)
        }

    def _generate_tikz_diagram(self, operation: str, strategies: List[str], pattern_relationships: Dict) -> str:
        """Generate TikZ code for the MUD diagram."""
        tikz_lines = [
            "\\begin{tikzpicture}[node distance=2cm and 2cm, auto]",
            f"\\node[draw, fill=blue!10, rounded corners] (title) at (0,0) {{\\textbf{{{operation.replace('_', ' ').title()} MUD}}}};",
            ""
        ]

        # Position strategies in a circle
        angle_step = 360 / len(strategies) if strategies else 1
        strategy_positions = {}

        for i, strategy in enumerate(strategies):
            angle = i * angle_step
            x = 4 * (angle * 3.14159 / 180)  # Convert to radians for positioning
            y = 3 * (angle * 3.14159 / 180)

            # Clean strategy name for display and escape underscores for LaTeX
            display_name = strategy.replace('SAR_', '').replace('_', '\\_')
            strategy_positions[strategy] = f"strategy_{i}"

            tikz_lines.append(f"\\node[draw, fill=green!10, rounded corners] ({strategy_positions[strategy]}) at ({x:.2f},{y:.2f}) {{{display_name}}};")

        tikz_lines.append("")

        # Add elaboration arrows
        arrow_count = 0
        for pattern, relationships in pattern_relationships.items():
            color = self._get_pattern_color(pattern)
            # Escape underscores in pattern names for LaTeX
            escaped_pattern = pattern.replace('_', '\\_')

            for elab in relationships:
                base_pos = strategy_positions[elab['base_strategy']]
                elab_pos = strategy_positions[elab['elaborated_strategy']]

                tikz_lines.append(f"\\draw[{color}, ->, thick] ({base_pos}) -- ({elab_pos})")
                tikz_lines.append(f"    node[midway, above, font=\\footnotesize] {{{escaped_pattern}}};")

                arrow_count += 1

        tikz_lines.extend([
            "",
            "\\end{tikzpicture}"
        ])

        return "\n".join(tikz_lines)

    def _get_pattern_color(self, pattern: str) -> str:
        """Get color for pattern arrows."""
        color_map = {
            'base_decomposition': 'red',
            'incremental_counting': 'blue',
            'counting_loop': 'green',
            'iterative_arithmetic': 'purple',
            'value_adjustment': 'orange'
        }
        return color_map.get(pattern, 'black')

    def _generate_mud_summary(self, operation: str, elaborations: List[Dict]) -> str:
        """Generate a textual summary of the MUD."""
        if not elaborations:
            return f"No algorithmic elaborations detected for {operation}."

        summary_lines = [
            f"Automated MUD Analysis for {operation.replace('_', ' ').title()}",
            "=" * 60,
            f"Total strategies analyzed: {len(set(e['base_strategy'] for e in elaborations) | set(e['elaborated_strategy'] for e in elaborations))}",
            f"Total elaborations detected: {len(elaborations)}",
            "",
            "Key Patterns Identified:"
        ]

        # Count pattern usage
        pattern_counts = defaultdict(int)
        for elab in elaborations:
            for pattern in elab['shared_patterns']:
                pattern_counts[pattern] += 1

        for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
            summary_lines.append(f"  • {pattern}: {count} relationships")

        summary_lines.extend([
            "",
            "Notable Elaborations:"
        ])

        # Show high-confidence elaborations
        high_confidence = [e for e in elaborations if e['confidence'] > 0.5]
        for elab in high_confidence[:5]:  # Show top 5
            summary_lines.append(f"  • {elab['base_strategy']} → {elab['elaborated_strategy']}")
            summary_lines.append(f"    Shared: {', '.join(elab['shared_patterns'])} (confidence: {elab['confidence']:.2f})")

        if len(high_confidence) > 5:
            summary_lines.append(f"  ... and {len(high_confidence) - 5} more high-confidence relationships")

        return "\n".join(summary_lines)

class ReportGenerator:
    """Generates reports in multiple formats from analysis results."""

    def __init__(self, analysis_results: Dict[str, Any], mud_diagrams: Dict[str, Any] = None):
        self.analysis_results = analysis_results
        self.mud_diagrams = mud_diagrams or {}

    def generate_markdown_report(self, strategy_name: str = None) -> str:
        """Generate a Markdown report for a specific strategy or general overview."""
        lines = [
            "# Algorithmic Elaboration Analysis Report",
            "",
            f"Generated on: {self._get_timestamp()}",
            "",
            "## Overview",
            "",
            f"- **Computational Patterns Detected**: {len(self.analysis_results.get('patterns', {}))}",
            f"- **Algorithmic Elaborations Found**: {len(self.analysis_results.get('elaborations', []))}",
            f"- **MUD Diagrams Generated**: {len(self.mud_diagrams)}",
            ""
        ]

        if strategy_name:
            lines.extend(self._generate_strategy_report(strategy_name))
        else:
            lines.extend(self._generate_overview_report())

        # Add MUD diagrams section if diagrams are available
        if self.mud_diagrams:
            lines.extend(self._generate_mud_diagrams_markdown_section())

        return "\n".join(lines)

    def _generate_strategy_report(self, strategy_name: str) -> List[str]:
        """Generate a detailed report for a specific strategy."""
        lines = [
            f"## Strategy Analysis: {strategy_name}",
            "",
            "### Computational Patterns Used",
        ]

        patterns = self.analysis_results.get('strategy_patterns', {}).get(strategy_name, [])
        if patterns:
            for pattern in patterns:
                pattern_info = self.analysis_results.get('patterns', {}).get(pattern, {})
                lines.append(f"- **{pattern}** ({pattern_info.get('type', 'unknown')})")
                lines.append(f"  - Used by {pattern_info.get('usage_count', 0)} other strategies")
        else:
            lines.append("- No patterns detected")

        lines.extend([
            "",
            "### Algorithmic Elaborations",
            "",
            "#### As Base Strategy:"
        ])

        base_elabs = [e for e in self.analysis_results.get('elaborations', [])
                     if e['base_strategy'] == strategy_name]
        if base_elabs:
            for elab in base_elabs:
                lines.extend([
                    f"- **Elaborates** → {elab['elaborated_strategy']}",
                    f"  - Shared patterns: {', '.join(elab['shared_patterns'])}",
                    f"  - Confidence: {elab['confidence']:.2f}",
                    ""
                ])
        else:
            lines.append("- None found")

        lines.extend([
            "",
            "#### As Elaborated Strategy:"
        ])

        elab_elabs = [e for e in self.analysis_results.get('elaborations', [])
                     if e['elaborated_strategy'] == strategy_name]
        if elab_elabs:
            for elab in elab_elabs:
                lines.extend([
                    f"- **Elaborated from** ← {elab['base_strategy']}",
                    f"  - Shared patterns: {', '.join(elab['shared_patterns'])}",
                    f"  - Confidence: {elab['confidence']:.2f}",
                    ""
                ])
        else:
            lines.append("- None found")

        return lines

    def _generate_overview_report(self) -> List[str]:
        """Generate a general overview report."""
        lines = [
            "## Computational Patterns",
            "",
            "| Pattern | Type | Usage Count | Strategies |",
            "|---------|------|-------------|------------|"
        ]

        for pattern_name, pattern_data in self.analysis_results.get('patterns', {}).items():
            strategies = ", ".join(pattern_data.get('strategies_using', [])[:3])  # Show first 3
            if len(pattern_data.get('strategies_using', [])) > 3:
                strategies += "..."
            lines.append(f"| {pattern_name} | {pattern_data.get('type', 'unknown')} | {pattern_data.get('usage_count', 0)} | {strategies} |")

        lines.extend([
            "",
            "## Key Algorithmic Elaborations",
            "",
            "| Base Strategy | Elaborated Strategy | Shared Patterns | Confidence |",
            "|---------------|---------------------|----------------|------------|"
        ])

        # Show top 10 by confidence
        elaborations = sorted(self.analysis_results.get('elaborations', []),
                            key=lambda x: x['confidence'], reverse=True)[:10]

        for elab in elaborations:
            patterns = ", ".join(elab['shared_patterns'])
            lines.extend([
                f"| {elab['base_strategy']} | {elab['elaborated_strategy']} | {patterns} | {elab['confidence']:.2f} |"
            ])

        return lines

    def generate_latex_report(self, strategy_name: str = None) -> str:
        """Generate a LaTeX report for a specific strategy or general overview."""
        lines = [
            "\\documentclass{article}",
            "\\usepackage[utf8]{inputenc}",
            "\\usepackage{geometry}",
            "\\usepackage{hyperref}",
            "\\usepackage{booktabs}",
            "\\usepackage{xcolor}",
            "\\usepackage{tikz}",
            "\\usetikzlibrary{positioning,arrows.meta}",
            "\\geometry{margin=1in}",
            "",
            "\\title{EPLE Algorithmic Elaboration Analysis Report}",
            f"\\date{{{self._get_timestamp()}}}",
            "\\author{EPLE Automated Analysis System}",
            "",
            "\\begin{document}",
            "\\maketitle",
            "",
            "\\section{Overview}",
            "",
            f"\\textbf{{Computational Patterns Detected:}} {len(self.analysis_results.get('patterns', {}))}\\\\",
            f"\\textbf{{Algorithmic Elaborations Found:}} {len(self.analysis_results.get('elaborations', []))}\\\\",
            f"\\textbf{{MUD Diagrams Generated:}} {len(self.mud_diagrams)}\\\\",
            ""
        ]

        if strategy_name:
            lines.extend(self._generate_strategy_latex_report(strategy_name))
        else:
            lines.extend(self._generate_overview_latex_report())

        # Add MUD diagrams section if diagrams are available
        if self.mud_diagrams:
            lines.extend(self._generate_mud_diagrams_latex_section())

        lines.extend([
            "",
            "\\end{document}"
        ])

        return "\n".join(lines)

    def _generate_strategy_latex_report(self, strategy_name: str) -> List[str]:
        """Generate a detailed LaTeX report for a specific strategy."""
        # Escape underscores in strategy name for LaTeX
        escaped_strategy = strategy_name.replace('_', '\\_')
        
        lines = [
            f"\\section{{Strategy Analysis: {escaped_strategy}}}",
            "",
            "\\subsection{Computational Patterns Used}",
            ""
        ]

        patterns = self.analysis_results.get('strategy_patterns', {}).get(strategy_name, [])
        if patterns:
            lines.append("\\begin{itemize}")
            for pattern in patterns:
                # Escape underscores in pattern names for LaTeX
                escaped_pattern = pattern.replace('_', '\\_')
                pattern_info = self.analysis_results.get('patterns', {}).get(pattern, {})
                lines.append(f"\\item \\textbf{{{escaped_pattern}}} ({pattern_info.get('type', 'unknown')})")
                lines.append(f"  \\textit{{Used by {pattern_info.get('usage_count', 0)} other strategies}}")
            lines.append("\\end{itemize}")
        else:
            lines.append("No patterns detected.")

        lines.extend([
            "",
            "\\subsection{Algorithmic Elaborations}",
            "",
            "\\subsubsection{As Base Strategy:}",
            ""
        ])

        base_elabs = [e for e in self.analysis_results.get('elaborations', [])
                     if e['base_strategy'] == strategy_name]
        if base_elabs:
            lines.append("\\begin{itemize}")
            for elab in base_elabs:
                # Escape underscores in strategy names and patterns for LaTeX
                elab_strategy = elab['elaborated_strategy'].replace('_', '\\_')
                patterns = ", ".join(elab['shared_patterns']).replace('_', '\\_')
                lines.extend([
                    f"\\item \\textbf{{Elaborates}} $\\rightarrow$ {elab_strategy}",
                    f"  \\textit{{Shared patterns: {patterns}}}",
                    f"  \\textit{{Confidence: {elab['confidence']:.2f}}}",
                    ""
                ])
            lines.append("\\end{itemize}")
        else:
            lines.append("None found.")

        lines.extend([
            "",
            "\\subsubsection{As Elaborated Strategy:}",
            ""
        ])

        elab_elabs = [e for e in self.analysis_results.get('elaborations', [])
                     if e['elaborated_strategy'] == strategy_name]
        if elab_elabs:
            lines.append("\\begin{itemize}")
            for elab in elab_elabs:
                # Escape underscores in strategy names and patterns for LaTeX
                base_strategy = elab['base_strategy'].replace('_', '\\_')
                patterns = ", ".join(elab['shared_patterns']).replace('_', '\\_')
                lines.extend([
                    f"\\item \\textbf{{Elaborated from}} $\\leftarrow$ {base_strategy}",
                    f"  \\textit{{Shared patterns: {patterns}}}",
                    f"  \\textit{{Confidence: {elab['confidence']:.2f}}}",
                    ""
                ])
            lines.append("\\end{itemize}")
        else:
            lines.append("None found.")

        return lines

    def _generate_overview_latex_report(self) -> List[str]:
        """Generate a general overview LaTeX report."""
        lines = [
            "\\section{Computational Patterns}",
            "",
            "\\begin{tabular}{@{}lll@{}}",
            "\\toprule",
            "\\textbf{Pattern} & \\textbf{Type} & \\textbf{Usage Count} \\\\",
            "\\midrule"
        ]

        for pattern_name, pattern_data in self.analysis_results.get('patterns', {}).items():
            # Escape underscores in pattern names for LaTeX
            escaped_pattern = pattern_name.replace('_', '\\_')
            lines.append(f"{escaped_pattern} & {pattern_data.get('type', 'unknown')} & {pattern_data.get('usage_count', 0)} \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "",
            "\\section{Key Algorithmic Elaborations}",
            "",
            "\\begin{tabular}{@{}llll@{}}",
            "\\toprule",
            "\\textbf{Base Strategy} & \\textbf{Elaborated Strategy} & \\textbf{Shared Patterns} & \\textbf{Confidence} \\\\",
            "\\midrule"
        ])

        # Show top 10 by confidence
        elaborations = sorted(self.analysis_results.get('elaborations', []),
                            key=lambda x: x['confidence'], reverse=True)[:10]

        for elab in elaborations:
            # Escape underscores in strategy names and patterns for LaTeX
            base_strategy = elab['base_strategy'].replace('_', '\\_')
            elab_strategy = elab['elaborated_strategy'].replace('_', '\\_')
            patterns = ", ".join(elab['shared_patterns']).replace('_', '\\_')
            lines.append(f"{base_strategy} & {elab_strategy} & {patterns} & {elab['confidence']:.2f} \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}"
        ])

        return lines

    def _generate_mud_diagrams_latex_section(self) -> List[str]:
        """Generate a LaTeX section containing all MUD diagrams."""
        lines = [
            "",
            "\\section{Meaning-Use Diagrams}",
            "",
            "The following diagrams illustrate the algorithmic elaborations detected in the analysis.",
            "Each diagram shows strategies connected by shared computational patterns.",
            ""
        ]

        for operation, diagram_data in self.mud_diagrams.items():
            # Create a subsection for each operation
            operation_title = operation.replace('_', ' ').title()
            lines.extend([
                f"\\subsection{{{operation_title}}}",
                "",
                f"\\textbf{{Operation:}} {operation_title}\\\\",
                f"\\textbf{{Strategies Analyzed:}} {len(diagram_data.get('strategies', []))}\\\\",
                f"\\textbf{{Elaborations Detected:}} {len(diagram_data.get('elaborations', []))}\\\\",
                "",
                "\\begin{center}",
                diagram_data.get('tikz_diagram', ''),
                "\\end{center}",
                "",
                "\\textbf{Summary:}\\\\",
                "\\begin{verbatim}",
                diagram_data.get('summary', ''),
                "\\end{verbatim}",
                "",
                "\\newpage"
            ])

        return lines

    def _generate_mud_diagrams_markdown_section(self) -> List[str]:
        """Generate a Markdown section containing MUD diagram information."""
        lines = [
            "",
            "## Meaning-Use Diagrams",
            "",
            "The following diagrams illustrate the algorithmic elaborations detected in the analysis. Each diagram shows strategies connected by shared computational patterns.",
            ""
        ]

        for operation, diagram_data in self.mud_diagrams.items():
            # Create a section for each operation
            operation_title = operation.replace('_', ' ').title()
            strategies = diagram_data.get('strategies', [])
            elaborations = diagram_data.get('elaborations', [])
            
            lines.extend([
                f"### {operation_title}",
                "",
                f"**Operation:** {operation_title}",
                f"**Strategies Analyzed:** {len(strategies)}",
                f"**Elaborations Detected:** {len(elaborations)}",
                "",
                "#### Strategies:",
            ])
            
            # List strategies
            for strategy in strategies:
                lines.append(f"- {strategy}")
            
            lines.extend([
                "",
                "#### Key Elaborations:"
            ])
            
            # List elaborations
            for elab in elaborations[:5]:  # Show top 5
                lines.extend([
                    f"- **{elab['base_strategy']}** → **{elab['elaborated_strategy']}**",
                    f"  - Shared patterns: {', '.join(elab['shared_patterns'])}",
                    f"  - Confidence: {elab['confidence']:.2f}"
                ])
            
            if len(elaborations) > 5:
                lines.append(f"- ... and {len(elaborations) - 5} more elaborations")
            
            lines.extend([
                "",
                "#### TikZ Diagram Code:",
                "",
                "```latex",
                diagram_data.get('tikz_diagram', ''),
                "```",
                "",
                "---"
            ])

        return lines

    def generate_html_report(self, strategy_name: str = None) -> str:
        """Generate an HTML report for a specific strategy or general overview."""
        lines = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "    <meta charset='UTF-8'>",
            "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            "    <title>EPLE Algorithmic Elaboration Analysis Report</title>",
            "    <style>",
            "        body { font-family: Arial, sans-serif; margin: 40px; }",
            "        h1, h2, h3 { color: #2c3e50; }",
            "        table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
            "        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "        th { background-color: #f2f2f2; }",
            "        .pattern { background-color: #e8f4f8; }",
            "        .elaboration { background-color: #f8e8e8; }",
            "        .confidence-high { color: #27ae60; font-weight: bold; }",
            "        .confidence-medium { color: #f39c12; }",
            "        .confidence-low { color: #e74c3c; }",
            "    </style>",
            "</head>",
            "<body>",
            "    <h1>EPLE Algorithmic Elaboration Analysis Report</h1>",
            f"    <p><strong>Generated on:</strong> {self._get_timestamp()}</p>",
            "    <h2>Overview</h2>",
            "    <ul>",
            f"        <li><strong>Computational Patterns Detected:</strong> {len(self.analysis_results.get('patterns', {}))}</li>",
            f"        <li><strong>Algorithmic Elaborations Found:</strong> {len(self.analysis_results.get('elaborations', []))}</li>",
            f"        <li><strong>MUD Diagrams Generated:</strong> {len(self.mud_diagrams)}</li>",
            "    </ul>"
        ]

        if strategy_name:
            lines.extend(self._generate_strategy_html_report(strategy_name))
        else:
            lines.extend(self._generate_overview_html_report())

        # Add MUD diagrams section if diagrams are available
        if self.mud_diagrams:
            lines.extend(self._generate_mud_diagrams_html_section())

        lines.extend([
            "</body>",
            "</html>"
        ])

        return "\n".join(lines)

    def _generate_strategy_html_report(self, strategy_name: str) -> List[str]:
        """Generate a detailed HTML report for a specific strategy."""
        lines = [
            f"    <h2>Strategy Analysis: {strategy_name}</h2>",
            "    <h3>Computational Patterns Used</h3>",
            "    <ul>"
        ]

        patterns = self.analysis_results.get('strategy_patterns', {}).get(strategy_name, [])
        if patterns:
            for pattern in patterns:
                pattern_info = self.analysis_results.get('patterns', {}).get(pattern, {})
                lines.append(f"        <li class='pattern'><strong>{pattern}</strong> ({pattern_info.get('type', 'unknown')})")
                lines.append(f"            <br><em>Used by {pattern_info.get('usage_count', 0)} other strategies</em></li>")
        else:
            lines.append("        <li>No patterns detected.</li>")

        lines.extend([
            "    </ul>",
            "    <h3>Algorithmic Elaborations</h3>",
            "    <h4>As Base Strategy:</h4>",
            "    <ul>"
        ])

        base_elabs = [e for e in self.analysis_results.get('elaborations', [])
                     if e['base_strategy'] == strategy_name]
        if base_elabs:
            for elab in base_elabs:
                confidence_class = self._get_confidence_class(elab['confidence'])
                lines.extend([
                    f"        <li class='elaboration'><strong>Elaborates</strong> → {elab['elaborated_strategy']}",
                    f"            <br><em>Shared patterns: {', '.join(elab['shared_patterns'])}</em>",
                    f"            <br><span class='{confidence_class}'>Confidence: {elab['confidence']:.2f}</span></li>"
                ])
        else:
            lines.append("        <li>None found.</li>")

        lines.extend([
            "    </ul>",
            "    <h4>As Elaborated Strategy:</h4>",
            "    <ul>"
        ])

        elab_elabs = [e for e in self.analysis_results.get('elaborations', [])
                     if e['elaborated_strategy'] == strategy_name]
        if elab_elabs:
            for elab in elab_elabs:
                confidence_class = self._get_confidence_class(elab['confidence'])
                lines.extend([
                    f"        <li class='elaboration'><strong>Elaborated from</strong> ← {elab['base_strategy']}",
                    f"            <br><em>Shared patterns: {', '.join(elab['shared_patterns'])}</em>",
                    f"            <br><span class='{confidence_class}'>Confidence: {elab['confidence']:.2f}</span></li>"
                ])
        else:
            lines.append("        <li>None found.</li>")

        lines.append("    </ul>")
        return lines

    def _generate_overview_html_report(self) -> List[str]:
        """Generate a general overview HTML report."""
        lines = [
            "    <h2>Computational Patterns</h2>",
            "    <table>",
            "        <tr>",
            "            <th>Pattern</th>",
            "            <th>Type</th>",
            "            <th>Usage Count</th>",
            "            <th>Strategies</th>",
            "        </tr>"
        ]

        for pattern_name, pattern_data in self.analysis_results.get('patterns', {}).items():
            strategies = ", ".join(pattern_data.get('strategies_using', [])[:3])
            if len(pattern_data.get('strategies_using', [])) > 3:
                strategies += "..."
            lines.extend([
                "        <tr>",
                f"            <td>{pattern_name}</td>",
                f"            <td>{pattern_data.get('type', 'unknown')}</td>",
                f"            <td>{pattern_data.get('usage_count', 0)}</td>",
                f"            <td>{strategies}</td>",
                "        </tr>"
            ])

        lines.extend([
            "    </table>",
            "    <h2>Key Algorithmic Elaborations</h2>",
            "    <table>",
            "        <tr>",
            "            <th>Base Strategy</th>",
            "            <th>Elaborated Strategy</th>",
            "            <th>Shared Patterns</th>",
            "            <th>Confidence</th>",
            "        </tr>"
        ])

        # Show top 10 by confidence
        elaborations = sorted(self.analysis_results.get('elaborations', []),
                            key=lambda x: x['confidence'], reverse=True)[:10]

        for elab in elaborations:
            patterns = ", ".join(elab['shared_patterns'])
            confidence_class = self._get_confidence_class(elab['confidence'])
            lines.extend([
                "        <tr>",
                f"            <td>{elab['base_strategy']}</td>",
                f"            <td>{elab['elaborated_strategy']}</td>",
                f"            <td>{patterns}</td>",
                f"            <td><span class='{confidence_class}'>{elab['confidence']:.2f}</span></td>",
                "        </tr>"
            ])

        lines.append("    </table>")
        return lines

    def _generate_mud_diagrams_html_section(self) -> List[str]:
        """Generate an HTML section containing MUD diagram information."""
        lines = [
            "    <h2>Meaning-Use Diagrams</h2>",
            "    <p>The following diagrams illustrate the algorithmic elaborations detected in the analysis. Each diagram shows strategies connected by shared computational patterns.</p>"
        ]

        for operation, diagram_data in self.mud_diagrams.items():
            # Create a section for each operation
            operation_title = operation.replace('_', ' ').title()
            strategies = diagram_data.get('strategies', [])
            elaborations = diagram_data.get('elaborations', [])
            
            lines.extend([
                f"    <h3>{operation_title}</h3>",
                "    <div style='background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px;'>",
                f"        <p><strong>Operation:</strong> {operation_title}</p>",
                f"        <p><strong>Strategies Analyzed:</strong> {len(strategies)}</p>",
                f"        <p><strong>Elaborations Detected:</strong> {len(elaborations)}</p>",
                "        <h4>Strategies:</h4>",
                "        <ul>"
            ])
            
            # List strategies
            for strategy in strategies:
                lines.append(f"            <li>{strategy}</li>")
            
            lines.extend([
                "        </ul>",
                "        <h4>Key Elaborations:</h4>",
                "        <ul>"
            ])
            
            # List elaborations
            for elab in elaborations[:5]:  # Show top 5
                confidence_class = self._get_confidence_class(elab['confidence'])
                lines.extend([
                    f"            <li><strong>{elab['base_strategy']}</strong> → <strong>{elab['elaborated_strategy']}</strong>",
                    f"                <br><em>Shared patterns: {', '.join(elab['shared_patterns'])}</em>",
                    f"                <br><span class='{confidence_class}'>Confidence: {elab['confidence']:.2f}</span></li>"
                ])
            
            if len(elaborations) > 5:
                lines.append(f"            <li><em>... and {len(elaborations) - 5} more elaborations</em></li>")
            
            lines.extend([
                "        </ul>",
                "        <h4>Diagram Code (TikZ):</h4>",
                "        <details>",
                "            <summary>Click to view TikZ code</summary>",
                "            <pre style='background-color: #f4f4f4; padding: 10px; border-radius: 3px; font-family: monospace; white-space: pre-wrap;'>",
                diagram_data.get('tikz_diagram', ''),
                "            </pre>",
                "        </details>",
                "    </div>"
            ])

        return lines

    def _get_confidence_class(self, confidence: float) -> str:
        """Get CSS class for confidence level."""
        if confidence >= 0.8:
            return "confidence-high"
        elif confidence >= 0.5:
            return "confidence-medium"
        else:
            return "confidence-low"

    def _get_timestamp(self) -> str:
        """Get current timestamp for reports."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def main():
    """Main entry point for MUD generation."""
    parser = argparse.ArgumentParser(description="Consolidated MUD Generator")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze automata for patterns')
    analyze_parser.add_argument('--automata-dir', default='src/automata', help='Directory containing automata')

    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate MUD diagrams')
    generate_parser.add_argument('--input', required=True, help='Input analysis results JSON file')
    generate_parser.add_argument('--output', default='diagrams/', help='Output directory for diagrams')

    # Report command
    report_parser = subparsers.add_parser('report', help='Generate reports')
    report_parser.add_argument('--input', required=True, help='Input analysis results JSON file')
    report_parser.add_argument('--strategy', help='Specific strategy to analyze')
    report_parser.add_argument('--format', choices=['markdown', 'latex', 'html'], default='markdown', help='Output format')
    report_parser.add_argument('--output', help='Output file (default: stdout)')

    args = parser.parse_args()

    if args.command == 'analyze':
        # Run analysis
        analyzer = AutomatonAnalyzer(args.automata_dir)
        results = analyzer.analyze_all_automata()

        # Save results
        output_file = 'automated_mud_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✅ Analysis complete. Results saved to {output_file}")

    elif args.command == 'generate':
        # Load analysis results
        with open(args.input, 'r') as f:
            data = json.load(f)

        # Extract only analysis results, ignore existing mud diagrams
        analysis_results = data.get('analysis_results', data)

        # Generate MUD diagrams
        mud_generator = MUDGenerator(analysis_results)
        mud_diagrams = mud_generator.generate_mud_diagrams()

        # Save combined results
        combined_results = {
            'analysis_results': analysis_results,
            'mud_diagrams': mud_diagrams
        }

        output_file = os.path.join(args.output, 'mud_diagrams.json')
        os.makedirs(args.output, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(combined_results, f, indent=2)

        print(f"✅ MUD diagrams generated. Results saved to {output_file}")

    elif args.command == 'report':
        # Load analysis results
        with open(args.input, 'r') as f:
            data = json.load(f)

        analysis_results = data.get('analysis_results', data)
        mud_diagrams = data.get('mud_diagrams', {})

        # Generate report
        report_gen = ReportGenerator(analysis_results, mud_diagrams)

        if args.format == 'markdown':
            report = report_gen.generate_markdown_report(args.strategy)

            if args.output:
                with open(args.output, 'w') as f:
                    f.write(report)
                print(f"✅ Markdown report saved to {args.output}")
            else:
                print(report)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()