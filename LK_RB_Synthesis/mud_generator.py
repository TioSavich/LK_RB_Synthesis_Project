#!/usr/bin/env python3#!/usr/bin/env python3

""""""

mud_generator.py: Consolidated Meaning-Use Diagram (MUD) Generatormud_generator.py: Consolidated Meaning-Use Diagram (MUD) Generator

"""

This script provides a unified interface for:

import os1. Automated discovery of algorithmic elaborations from automaton code

import sys2. Generation of professional TikZ diagrams following Brandom's conventions

import json3. Multiple output formats (TikZ, JSON, reports)

import ast

import inspectUsage:

from typing import Dict, List, Set, Tuple, Any    python mud_generator.py analyze --automata-dir src/automata

from dataclasses import dataclass, field    python mud_generator.py generate --input automated_mud_results.json --output diagrams/

from collections import defaultdict    python mud_generator.py report --input automated_mud_results.json --format latex

import argparse"""

import math # Import math for layout calculations

import datetimeimport os

import sys

@dataclassimport json

class ComputationalPattern:import ast

    """Represents a detected computational pattern/subroutine."""import inspect

    name: strfrom typing import Dict, List, Set, Tuple, Any

    operation_type: str  # 'counting', 'decomposition', 'adjustment', etc.from dataclasses import dataclass, field

    register_operations: List[str]from collections import defaultdict

    state_transitions: List[str]import argparse

    strategies_using: Set[str] = field(default_factory=set)

@dataclass

@dataclassclass ComputationalPattern:

class AlgorithmicElaboration:    """Represents a detected computational pattern/subroutine."""

    """Represents an algorithmic elaboration relationship."""    name: str

    base_strategy: str    operation_type: str  # 'counting', 'decomposition', 'adjustment', etc.

    elaborated_strategy: str    register_operations: List[str]

    shared_patterns: Set[str]    state_transitions: List[str]

    elaboration_type: str  # 'intra_categorial', 'inter_categorial'    strategies_using: Set[str] = field(default_factory=set)

    confidence: float

@dataclass

class AutomatonAnalyzer:class AlgorithmicElaboration:

    """Analyzes automaton implementations to detect patterns and relationships."""    """Represents an algorithmic elaboration relationship."""

    base_strategy: str

    def __init__(self, automata_dir: str):    elaborated_strategy: str

        self.automata_dir = automata_dir    shared_patterns: Set[str]

        self.patterns: Dict[str, ComputationalPattern] = {}    elaboration_type: str  # 'intra_categorial', 'inter_categorial'

        self.elaborations: List[AlgorithmicElaboration] = []    confidence: float

        self.strategy_patterns: Dict[str, Set[str]] = defaultdict(set)

class AutomatonAnalyzer:

    def analyze_all_automata(self) -> Dict[str, Any]:    """Analyzes automaton implementations to detect patterns and relationships."""

        """Main analysis pipeline."""

        print("🔬 Starting Automated Automaton Analysis", file=sys.stderr)    def __init__(self, automata_dir: str):

        print("=" * 50, file=sys.stderr)        self.automata_dir = automata_dir

        self.patterns: Dict[str, ComputationalPattern] = {}

        # Step 1: Extract patterns from all automata        self.elaborations: List[AlgorithmicElaboration] = []

        self._extract_patterns_from_automata()        self.strategy_patterns: Dict[str, Set[str]] = defaultdict(set)



        # Step 2: Detect algorithmic elaborations    def analyze_all_automata(self) -> Dict[str, Any]:

        self._detect_elaborations()        """Main analysis pipeline."""

        print("🔬 Starting Automated Automaton Analysis")

        # Step 3: Generate analysis report        print("=" * 50)

        return self._generate_analysis_report()

        # Step 1: Extract patterns from all automata

    def _extract_patterns_from_automata(self):        self._extract_patterns_from_automata()

        """Extract computational patterns from automaton source code."""

        print("\n📋 Extracting Computational Patterns...", file=sys.stderr)        # Step 2: Detect algorithmic elaborations

        self._detect_elaborations()

        for operation_dir in ['addition', 'subtraction', 'multiplication', 'division']:

            op_path = os.path.join(self.automata_dir, operation_dir)        # Step 3: Generate analysis report

            if not os.path.exists(op_path):        return self._generate_analysis_report()

                continue

    def _extract_patterns_from_automata(self):

            for filename in os.listdir(op_path):        """Extract computational patterns from automaton source code."""

                if filename.endswith('.py') and not filename.startswith('__'):        print("\n📋 Extracting Computational Patterns...")

                    strategy_id = filename.replace('.py', '').replace('SAR_', '')

                    filepath = os.path.join(op_path, filename)        for operation_dir in ['addition', 'subtraction', 'multiplication', 'division']:

            op_path = os.path.join(self.automata_dir, operation_dir)

                    try:            if not os.path.exists(op_path):

                        patterns = self._analyze_single_automaton(filepath, strategy_id, operation_dir)                continue

                        self.strategy_patterns[strategy_id] = patterns

                        print(f"✅ Analyzed {strategy_id}: {len(patterns)} patterns found", file=sys.stderr)            for filename in os.listdir(op_path):

                    except Exception as e:                if filename.endswith('.py') and not filename.startswith('__'):

                        print(f"❌ Error analyzing {strategy_id}: {e}", file=sys.stderr)                    strategy_id = filename.replace('.py', '').replace('SAR_', '')

                    filepath = os.path.join(op_path, filename)

    def _analyze_single_automaton(self, filepath: str, strategy_id: str, operation: str) -> Set[str]:

        """Analyze a single automaton file to extract patterns."""                    try:

        with open(filepath, 'r') as f:                        patterns = self._analyze_single_automaton(filepath, strategy_id, operation_dir)

            source_code = f.read()                        self.strategy_patterns[strategy_id] = patterns

                        print(f"✅ Analyzed {strategy_id}: {len(patterns)} patterns found")

        # Parse the AST                    except Exception as e:

        tree = ast.parse(source_code)                        print(f"❌ Error analyzing {strategy_id}: {e}")



        patterns_found = set()    def _analyze_single_automaton(self, filepath: str, strategy_id: str, operation: str) -> Set[str]:

        register_ops = []        """Analyze a single automaton file to extract patterns."""

        state_methods = []        with open(filepath, 'r') as f:

            source_code = f.read()

        # Extract class definition and methods

        for node in ast.walk(tree):        # Parse the AST

            if isinstance(node, ast.ClassDef):        tree = ast.parse(source_code)

                # Look for execute_ methods (state handlers)

                for item in node.body:        patterns_found = set()

                    if isinstance(item, ast.FunctionDef) and item.name.startswith('execute_'):        register_ops = []

                        state_name = item.name.replace('execute_', '')        state_methods = []

                        operations = self._extract_operations_from_method(item)

                        register_ops.extend(operations)        # Extract class definition and methods

        for node in ast.walk(tree):

                        # Detect specific patterns            if isinstance(node, ast.ClassDef):

                        patterns = self._detect_patterns_in_method(item, state_name, operations)                # Look for execute_ methods (state handlers)

                        patterns_found.update(patterns)                for item in node.body:

                    if isinstance(item, ast.FunctionDef) and item.name.startswith('execute_'):

                        # Update pattern usage                        state_name = item.name.replace('execute_', '')

                        for pattern_name in patterns:                        operations = self._extract_operations_from_method(item)

                            if pattern_name not in self.patterns:                        register_ops.extend(operations)

                                self.patterns[pattern_name] = ComputationalPattern(

                                    name=pattern_name,                        # Detect specific patterns

                                    operation_type=self._classify_pattern(pattern_name),                        patterns = self._detect_patterns_in_method(item, state_name, operations)

                                    register_operations=[],                        patterns_found.update(patterns)

                                    state_transitions=[]

                                )                        # Update pattern usage

                            self.patterns[pattern_name].strategies_using.add(strategy_id)                        for pattern_name in patterns:

                            if pattern_name not in self.patterns:

        return patterns_found                                self.patterns[pattern_name] = ComputationalPattern(

                                    name=pattern_name,

    def _extract_operations_from_method(self, method_node: ast.FunctionDef) -> List[str]:                                    operation_type=self._classify_pattern(pattern_name),

        """Extract register operations from a method, including conditionals."""                                    register_operations=[],

        operations = []                                    state_transitions=[]

                                )

        for node in ast.walk(method_node):                            self.patterns[pattern_name].strategies_using.add(strategy_id)

            # Regular assignments

            if isinstance(node, ast.Assign):        return patterns_found

                for target in node.targets:

                    if isinstance(target, ast.Name):    def _extract_operations_from_method(self, method_node: ast.FunctionDef) -> List[str]:

                        operations.append(f"{target.id} = {self._extract_value(node.value)}")        """Extract register operations from a method, including conditionals."""

        operations = []

            # Augmented assignments (x += 1, etc.)

            elif isinstance(node, ast.AugAssign):        for node in ast.walk(method_node):

                if isinstance(node.target, ast.Name):            # Regular assignments

                    target = node.target.id            if isinstance(node, ast.Assign):

                    if isinstance(node.op, ast.Add):                for target in node.targets:

                        operations.append(f"{target} += {self._extract_value(node.value)}")                    if isinstance(target, ast.Name):

                    elif isinstance(node.op, ast.Sub):                        operations.append(f"{target.id} = {self._extract_value(node.value)}")

                        operations.append(f"{target} -= {self._extract_value(node.value)}")

            # Augmented assignments (x += 1, etc.)

            # Function calls that might be transitions or operations            elif isinstance(node, ast.AugAssign):

            elif isinstance(node, ast.Call):                if isinstance(node.target, ast.Name):

                if isinstance(node.func, ast.Attribute):                    target = node.target.id

                    if node.func.attr == 'transition':                    if isinstance(node.op, ast.Add):

                        operations.append(f"transition: {self._extract_call_args(node)}")                        operations.append(f"{target} += {self._extract_value(node.value)}")

                    elif node.func.attr == '_record_history':                    elif isinstance(node.op, ast.Sub):

                        operations.append(f"record_history: {self._extract_call_args(node)}")                        operations.append(f"{target} -= {self._extract_value(node.value)}")



        return operations            # Function calls that might be transitions or operations

            elif isinstance(node, ast.Call):

    def _extract_value(self, node: ast.AST) -> str:                if isinstance(node.func, ast.Attribute):

        """Extract value from AST node."""                    if node.func.attr == 'transition':

        if isinstance(node, ast.Constant):  # Python 3.8+                        operations.append(f"transition: {self._extract_call_args(node)}")

            return str(node.value)                    elif node.func.attr == '_record_history':

        elif isinstance(node, ast.Num):  # Legacy support                        operations.append(f"record_history: {self._extract_call_args(node)}")

            return str(node.n)

        elif isinstance(node, ast.Name):        return operations

            return node.id

        elif isinstance(node, ast.Attribute):    def _extract_value(self, node: ast.AST) -> str:

            return f"{node.attr}"        """Extract value from AST node."""

        elif isinstance(node, ast.BinOp):        if isinstance(node, ast.Constant):  # Python 3.8+

            left = self._extract_value(node.left)            return str(node.value)

            right = self._extract_value(node.right)        elif isinstance(node, ast.Num):  # Legacy support

            if isinstance(node.op, ast.Add):            return str(node.n)

                return f"{left} + {right}"        elif isinstance(node, ast.Name):

            elif isinstance(node.op, ast.Sub):            return node.id

                return f"{left} - {right}"        elif isinstance(node, ast.Attribute):

            elif isinstance(node.op, ast.Mult):            return f"{node.attr}"

                return f"{left} * {right}"        elif isinstance(node, ast.BinOp):

            elif isinstance(node.op, ast.Div):            left = self._extract_value(node.left)

                return f"{left} // {right}"  # Integer division common in automata            right = self._extract_value(node.right)

            elif isinstance(node.op, ast.Mod):            if isinstance(node.op, ast.Add):

                return f"{left} % {right}"                return f"{left} + {right}"

        return "complex_expr"            elif isinstance(node.op, ast.Sub):

                return f"{left} - {right}"

    def _extract_call_args(self, call_node: ast.Call) -> str:            elif isinstance(node.op, ast.Mult):

        """Extract arguments from a function call."""                return f"{left} * {right}"

        args = []            elif isinstance(node.op, ast.Div):

        for arg in call_node.args:                return f"{left} // {right}"  # Integer division common in automata

            if isinstance(arg, ast.Constant):  # Python 3.8+            elif isinstance(node.op, ast.Mod):

                args.append(f"'{arg.value}'")                return f"{left} % {right}"

            elif isinstance(arg, ast.Str):  # Legacy support        return "complex_expr"

                args.append(f"'{arg.s}'")

            elif isinstance(arg, ast.Name):    def _extract_call_args(self, call_node: ast.Call) -> str:

                args.append(arg.id)        """Extract arguments from a function call."""

            elif isinstance(arg, ast.Num):  # Legacy support        args = []

                args.append(str(arg.n))        for arg in call_node.args:

            else:            if isinstance(arg, ast.Constant):  # Python 3.8+

                args.append("expr")                args.append(f"'{arg.value}'")

        return ", ".join(args)            elif isinstance(arg, ast.Str):  # Legacy support

                args.append(f"'{arg.s}'")

    def _detect_patterns_in_method(self, method_node: ast.FunctionDef, state_name: str, operations: List[str]) -> Set[str]:            elif isinstance(arg, ast.Name):

        """Detect computational patterns in a method."""                args.append(arg.id)

        patterns = set()            elif isinstance(arg, ast.Num):  # Legacy support

                args.append(str(arg.n))

        # Get the source code for more detailed analysis            else:

        method_source = self._get_method_source(method_node)                args.append("expr")

        return ", ".join(args)

        # Pattern 1: Counting loops (state-based iteration)

        if self._is_counting_loop_pattern(method_source, operations):    def _detect_patterns_in_method(self, method_node: ast.FunctionDef, state_name: str, operations: List[str]) -> Set[str]:

            patterns.add("counting_loop")        """Detect computational patterns in a method."""

        patterns = set()

        # Pattern 2: Base decomposition

        if self._is_decomposition_pattern(operations) or '//' in method_source or '%' in method_source:        # Get the source code for more detailed analysis

            patterns.add("base_decomposition")        method_source = self._get_method_source(method_node)



        # Pattern 3: Adjustment calculations        # Pattern 1: Counting loops (state-based iteration)

        if self._is_adjustment_pattern(operations) or 'TargetBase' in method_source or 'K =' in method_source:        if self._is_counting_loop_pattern(method_source, operations):

            patterns.add("value_adjustment")            patterns.add("counting_loop")



        # Pattern 4: Iterative addition/subtraction        # Pattern 2: Base decomposition

        if self._is_iterative_arithmetic(operations) or 'Sum += ' in method_source or 'Current += ' in method_source:        if self._is_decomposition_pattern(operations) or '//' in method_source or '%' in method_source:

            patterns.add("iterative_arithmetic")            patterns.add("base_decomposition")



        # Pattern 5: State-based counting transitions        # Pattern 3: Adjustment calculations

        if self._is_state_based_counting(state_name, method_source):        if self._is_adjustment_pattern(operations) or 'TargetBase' in method_source or 'K =' in method_source:

            patterns.add("incremental_counting")            patterns.add("value_adjustment")



        # Pattern 6: Decomposition and reconstruction        # Pattern 4: Iterative addition/subtraction

        if self._is_decomposition_reconstruction_pattern(method_source):        if self._is_iterative_arithmetic(operations) or 'Sum += ' in method_source or 'Current += ' in method_source:

            patterns.add("decomposition_reconstruction")            patterns.add("iterative_arithmetic")



        return patterns        # Pattern 5: State-based counting transitions

        if self._is_state_based_counting(state_name, method_source):

    def _get_method_source(self, method_node: ast.FunctionDef) -> str:            patterns.add("incremental_counting")

        """Extract source code from method node."""

        # This is a simplified approach - in practice you'd need line numbers        # Pattern 6: Decomposition and reconstruction

        # For now, we'll reconstruct from operations and state name        if self._is_decomposition_reconstruction_pattern(method_source):

        return " ".join([str(op) for op in self._extract_operations_from_method(method_node)])            patterns.add("decomposition_reconstruction")



    def _is_counting_loop_pattern(self, method_source: str, operations: List[str]) -> bool:        return patterns

        """Detect state-based counting loops."""

        # Look for patterns that indicate iterative counting    def _get_method_source(self, method_node: ast.FunctionDef) -> str:

        has_counter = any('Count' in op for op in operations)        """Extract source code from method node."""

        has_increment = any('+=' in op for op in operations)        # This is a simplified approach - in practice you'd need line numbers

        has_comparison = '<' in method_source or '>' in method_source        # For now, we'll reconstruct from operations and state name

        has_conditional = 'if' in method_source or 'while' in method_source        return " ".join([str(op) for op in self._extract_operations_from_method(method_node)])



        return has_counter and has_increment and (has_comparison or has_conditional)    def _is_counting_loop_pattern(self, method_source: str, operations: List[str]) -> bool:

        """Detect state-based counting loops."""

    def _is_decomposition_pattern(self, operations: List[str]) -> bool:        # Look for patterns that indicate iterative counting

        """Detect base decomposition patterns."""        has_counter = any('Count' in op for op in operations)

        return any('//' in op or '%' in op for op in operations)        has_increment = any('+=' in op for op in operations)

        has_comparison = '<' in method_source or '>' in method_source

    def _is_adjustment_pattern(self, operations: List[str]) -> bool:        has_conditional = 'if' in method_source or 'while' in method_source

        """Detect value adjustment patterns."""

        return any('TargetBase' in op or 'K =' in op for op in operations)        return has_counter and has_increment and (has_comparison or has_conditional)



    def _is_iterative_arithmetic(self, operations: List[str]) -> bool:    def _is_decomposition_pattern(self, operations: List[str]) -> bool:

        """Detect iterative arithmetic patterns."""        """Detect base decomposition patterns."""

        return any('Sum += ' in op or 'Current += ' in op for op in operations)        return any('//' in op or '%' in op for op in operations)



    def _is_state_based_counting(self, state_name: str, method_source: str) -> bool:    def _is_adjustment_pattern(self, operations: List[str]) -> bool:

        """Detect state-based counting patterns."""        """Detect value adjustment patterns."""

        counting_states = ['inc_tens', 'inc_hundreds', 'add_bases', 'add_ones', 'loop_K', 'count']        return any('TargetBase' in op or 'K =' in op for op in operations)

        return any(state in state_name.lower() for state in counting_states)

    def _is_iterative_arithmetic(self, operations: List[str]) -> bool:

    def _is_decomposition_reconstruction_pattern(self, method_source: str) -> bool:        """Detect iterative arithmetic patterns."""

        """Detect patterns that decompose and reconstruct values."""        return any('Sum += ' in op or 'Current += ' in op for op in operations)

        return ('//' in method_source and '%' in method_source) or \

               ('BaseCounter' in method_source and 'OneCounter' in method_source)    def _is_state_based_counting(self, state_name: str, method_source: str) -> bool:

        """Detect state-based counting patterns."""

    def _classify_pattern(self, pattern_name: str) -> str:        counting_states = ['inc_tens', 'inc_hundreds', 'add_bases', 'add_ones', 'loop_K', 'count']

        """Classify a pattern by its computational type."""        return any(state in state_name.lower() for state in counting_states)

        classifications = {

            "counting_loop": "counting",    def _is_decomposition_reconstruction_pattern(self, method_source: str) -> bool:

            "base_decomposition": "decomposition",        """Detect patterns that decompose and reconstruct values."""

            "value_adjustment": "adjustment",        return ('//' in method_source and '%' in method_source) or \

            "iterative_arithmetic": "arithmetic",               ('BaseCounter' in method_source and 'OneCounter' in method_source)

            "incremental_counting": "counting"

        }    def _classify_pattern(self, pattern_name: str) -> str:

        return classifications.get(pattern_name, "general")        """Classify a pattern by its computational type."""

        classifications = {

    def _detect_elaborations(self):            "counting_loop": "counting",

        """Detect algorithmic elaborations based on shared patterns."""            "base_decomposition": "decomposition",

        print("\n🔗 Detecting Algorithmic Elaborations...", file=sys.stderr)            "value_adjustment": "adjustment",

            "iterative_arithmetic": "arithmetic",

        strategy_list = list(self.strategy_patterns.keys())            "incremental_counting": "counting"

        }

        for i, strategy_a in enumerate(strategy_list):        return classifications.get(pattern_name, "general")

            for strategy_b in strategy_list[i+1:]:

                shared_patterns = self.strategy_patterns[strategy_a] & self.strategy_patterns[strategy_b]    def _detect_elaborations(self):

        """Detect algorithmic elaborations based on shared patterns."""

                if shared_patterns:        print("\n🔗 Detecting Algorithmic Elaborations...")

                    # Determine operation types

                    op_a = self._get_operation_type(strategy_a)        strategy_list = list(self.strategy_patterns.keys())

                    op_b = self._get_operation_type(strategy_b)

        for i, strategy_a in enumerate(strategy_list):

                    elaboration_type = "intra_categorial" if op_a == op_b else "inter_categorial"            for strategy_b in strategy_list[i+1:]:

                    confidence = len(shared_patterns) / max(len(self.strategy_patterns[strategy_a]),                shared_patterns = self.strategy_patterns[strategy_a] & self.strategy_patterns[strategy_b]

                                                          len(self.strategy_patterns[strategy_b]))

                if shared_patterns:

                    # Determine elaboration direction based on pattern complexity                    # Determine operation types

                    base_strategy, elab_strategy = self._determine_elaboration_direction(                    op_a = self._get_operation_type(strategy_a)

                        strategy_a, strategy_b, shared_patterns                    op_b = self._get_operation_type(strategy_b)

                    )

                    elaboration_type = "intra_categorial" if op_a == op_b else "inter_categorial"

                    elaboration = AlgorithmicElaboration(                    confidence = len(shared_patterns) / max(len(self.strategy_patterns[strategy_a]),

                        base_strategy=base_strategy,                                                          len(self.strategy_patterns[strategy_b]))

                        elaborated_strategy=elab_strategy,

                        shared_patterns=shared_patterns,                    # Determine elaboration direction based on pattern complexity

                        elaboration_type=elaboration_type,                    base_strategy, elab_strategy = self._determine_elaboration_direction(

                        confidence=confidence                        strategy_a, strategy_b, shared_patterns

                    )                    )



                    self.elaborations.append(elaboration)                    elaboration = AlgorithmicElaboration(

                        base_strategy=base_strategy,

    def _get_operation_type(self, strategy_id: str) -> str:                        elaborated_strategy=elab_strategy,

        """Determine operation type from strategy ID."""                        shared_patterns=shared_patterns,

        if any(keyword in strategy_id.upper() for keyword in ['ADD', 'COUNTING']):                        elaboration_type=elaboration_type,

            return 'addition'                        confidence=confidence

        elif any(keyword in strategy_id.upper() for keyword in ['SUB', 'SLIDING']):                    )

            return 'subtraction'

        elif any(keyword in strategy_id.upper() for keyword in ['MULT', 'CBO']):                    self.elaborations.append(elaboration)

            return 'multiplication'

        elif any(keyword in strategy_id.upper() for keyword in ['DIV', 'DEALING']):    def _get_operation_type(self, strategy_id: str) -> str:

            return 'division'        """Determine operation type from strategy ID."""

        return 'unknown'        if any(keyword in strategy_id.upper() for keyword in ['ADD', 'COUNTING']):

            return 'addition'

    def _determine_elaboration_direction(self, strategy_a: str, strategy_b: str, shared_patterns: Set[str]) -> Tuple[str, str]:        elif any(keyword in strategy_id.upper() for keyword in ['SUB', 'SLIDING']):

        """Determine which strategy elaborates which based on pattern analysis."""            return 'subtraction'

        # Simple heuristic: strategy with fewer unique patterns is the base        elif any(keyword in strategy_id.upper() for keyword in ['MULT', 'CBO']):

        unique_a = len(self.strategy_patterns[strategy_a] - shared_patterns)            return 'multiplication'

        unique_b = len(self.strategy_patterns[strategy_b] - shared_patterns)        elif any(keyword in strategy_id.upper() for keyword in ['DIV', 'DEALING']):

            return 'division'

        if unique_a <= unique_b:        return 'unknown'

            return strategy_a, strategy_b

        else:    def _determine_elaboration_direction(self, strategy_a: str, strategy_b: str, shared_patterns: Set[str]) -> Tuple[str, str]:

            return strategy_b, strategy_a        """Determine which strategy elaborates which based on pattern analysis."""

        # Simple heuristic: strategy with fewer unique patterns is the base

    def _generate_analysis_report(self) -> Dict[str, Any]:        unique_a = len(self.strategy_patterns[strategy_a] - shared_patterns)

        """Generate comprehensive analysis report."""        unique_b = len(self.strategy_patterns[strategy_b] - shared_patterns)

        print(f"\n📊 Analysis Complete:", file=sys.stderr)

        print(f"   • {len(self.patterns)} computational patterns detected", file=sys.stderr)        if unique_a <= unique_b:

        print(f"   • {len(self.elaborations)} algorithmic elaborations identified", file=sys.stderr)            return strategy_a, strategy_b

        else:

        return {            return strategy_b, strategy_a

            "patterns": {

                name: {    def _generate_analysis_report(self) -> Dict[str, Any]:

                    "type": pattern.operation_type,        """Generate comprehensive analysis report."""

                    "strategies_using": list(pattern.strategies_using),        print(f"\n📊 Analysis Complete:")

                    "usage_count": len(pattern.strategies_using)        print(f"   • {len(self.patterns)} computational patterns detected")

                }        print(f"   • {len(self.elaborations)} algorithmic elaborations identified")

                for name, pattern in self.patterns.items()

            },        return {

            "elaborations": [            "patterns": {

                {                name: {

                    "base_strategy": elab.base_strategy,                    "type": pattern.operation_type,

                    "elaborated_strategy": elab.elaborated_strategy,                    "strategies_using": list(pattern.strategies_using),

                    "shared_patterns": list(elab.shared_patterns),                    "usage_count": len(pattern.strategies_using)

                    "type": elab.elaboration_type,                }

                    "confidence": elab.confidence                for name, pattern in self.patterns.items()

                }            },

                for elab in self.elaborations            "elaborations": [

            ],                {

            "strategy_patterns": {                    "base_strategy": elab.base_strategy,

                strategy: list(patterns)                    "elaborated_strategy": elab.elaborated_strategy,

                for strategy, patterns in self.strategy_patterns.items()                    "shared_patterns": list(elab.shared_patterns),

            }                    "type": elab.elaboration_type,

        }                    "confidence": elab.confidence

                }

# --- MUDGenerator Class (Updated for MUD conventions) ---                for elab in self.elaborations

            ],

class MUDGenerator:            "strategy_patterns": {

    """Generates MUD diagrams from algorithmic elaboration analysis."""                strategy: list(patterns)

                for strategy, patterns in self.strategy_patterns.items()

    def __init__(self, analysis_results: Dict[str, Any]):            }

        self.analysis_results = analysis_results        }

        self.mud_diagrams = {}

class MUDGenerator:

    def generate_mud_diagrams(self) -> Dict[str, Any]:    """Generates MUD diagrams from algorithmic elaboration analysis."""

        """Generate MUD diagrams for all discovered elaborations."""

        operation_groups = self._group_elaborations_by_operation()    def __init__(self, analysis_results: Dict[str, Any]):

        self.analysis_results = analysis_results

        for operation, elaborations in operation_groups.items():        self.mud_diagrams = {}

            mud_diagram = self._generate_operation_mud(operation, elaborations)

            self.mud_diagrams[operation] = mud_diagram    def generate_mud_diagrams(self) -> Dict[str, Any]:

        """Generate MUD diagrams for all discovered elaborations."""

        return self.mud_diagrams        print("🎨 Generating Meaning-Use Diagrams")

        print("=" * 50)

    def _group_elaborations_by_operation(self) -> Dict[str, List[Dict]]:

        operation_groups = defaultdict(list)        # Group elaborations by operation type

        for elab in self.analysis_results.get('elaborations', []):        operation_groups = self._group_elaborations_by_operation()

            base_op = self._extract_operation_type(elab.get('base_strategy', ''))

            if base_op == 'general':        for operation, elaborations in operation_groups.items():

                base_op = 'miscellaneous'            print(f"\n📊 Generating MUD for {operation}")

            operation_groups[base_op].append(elab)            mud_diagram = self._generate_operation_mud(operation, elaborations)

        return operation_groups            self.mud_diagrams[operation] = mud_diagram



    def _extract_operation_type(self, strategy_id: str) -> str:        return self.mud_diagrams

        strategy_upper = strategy_id.upper()

        if 'ADD' in strategy_upper or 'COUNTING' in strategy_upper:    def _group_elaborations_by_operation(self) -> Dict[str, List[Dict]]:

            return 'addition'        """Group elaborations by their primary operation type."""

        elif 'SUB' in strategy_upper:        operation_groups = defaultdict(list)

            return 'subtraction'

        elif 'MULT' in strategy_upper:        for elab in self.analysis_results.get('elaborations', []):

            return 'multiplication'            # Determine operation from strategy names

        elif 'DIV' in strategy_upper:            base_op = self._extract_operation_type(elab['base_strategy'])

            return 'division'            elab_op = self._extract_operation_type(elab['elaborated_strategy'])

        return 'general'

            # Use the more specific operation if they differ

    def _generate_operation_mud(self, operation: str, elaborations: List[Dict]) -> Dict[str, Any]:            if base_op != elab_op:

        """Generate a MUD diagram for a specific operation."""                operation = f"{base_op}_to_{elab_op}"

        strategies = set()            else:

        elaboration_relationships = defaultdict(list)                operation = base_op



        for elab in elaborations:            operation_groups[operation].append(elab)

            base = elab.get('base_strategy')

            elaborated = elab.get('elaborated_strategy')        return operation_groups

            patterns = elab.get('shared_patterns', [])

    def _extract_operation_type(self, strategy_id: str) -> str:

            if base and elaborated:        """Extract operation type from strategy ID."""

                strategies.add(base)        strategy_upper = strategy_id.upper()

                strategies.add(elaborated)

                directional_key = (base, elaborated)        if any(op in strategy_upper for op in ['ADD', 'COUNTING']):

                for pattern in patterns:            return 'addition'

                    if pattern not in elaboration_relationships[directional_key]:        elif any(op in strategy_upper for op in ['SUB', 'SLIDING']):

                        elaboration_relationships[directional_key].append(pattern)            return 'subtraction'

        elif any(op in strategy_upper for op in ['MULT', 'CBO']):

        tikz_code = self._generate_tikz_diagram(operation, list(strategies), elaboration_relationships)            return 'multiplication'

        elif any(op in strategy_upper for op in ['DIV', 'DEALING']):

        return {            return 'division'

            'operation': operation,        else:

            'strategies': list(strategies),            return 'general'

            'tikz_diagram': tikz_code,

            'summary': f"Summary for {operation}"    def _generate_operation_mud(self, operation: str, elaborations: List[Dict]) -> Dict[str, Any]:

        }        """Generate a MUD diagram for a specific operation."""

        # Collect all strategies involved

    def _format_strategy_label(self, strategy_name: str) -> str:        strategies = set()

        """Formats the strategy name according to MUD typesetting rules: P\textsubscript{Name}."""        pattern_relationships = defaultdict(list)

        display_name = strategy_name.replace('SAR_', '')

        for elab in elaborations:

        if len(display_name) > 25:            strategies.add(elab['base_strategy'])

            best_split_point = -1            strategies.add(elab['elaborated_strategy'])

            middle = len(display_name) / 2

            for i, char in enumerate(display_name):            # Group by shared patterns

                if char == '_':            for pattern in elab['shared_patterns']:

                    if best_split_point == -1 or abs(i - middle) < abs(best_split_point - middle):                pattern_relationships[pattern].append(elab)

                        best_split_point = i

        # Generate TikZ code

            if best_split_point != -1 and best_split_point > 0:        tikz_code = self._generate_tikz_diagram(operation, list(strategies), pattern_relationships)

                part1 = display_name[:best_split_point]

                part2 = display_name[best_split_point+1:]        return {

                return rf"P\textsubscript{{{part1}}} \\ \textsubscript{{{part2}}}"            'operation': operation,

            'strategies': list(strategies),

        return rf"P\textsubscript{{{display_name}}}"            'elaborations': elaborations,

            'pattern_relationships': dict(pattern_relationships),

    def _generate_tikz_diagram(self, operation: str, strategies: List[str], elaboration_relationships: Dict[Tuple[str, str], List[str]]) -> str:            'tikz_diagram': tikz_code,

        """Generate TikZ code for the MUD diagram following Brandom's conventions."""            'summary': self._generate_mud_summary(operation, elaborations)

        tikz_lines = [        }

            r"\begin{tikzpicture}[",

            "  % Node Styles",    def _generate_tikz_diagram(self, operation: str, strategies: List[str], pattern_relationships: Dict) -> str:

            r"  vnode/.style={ellipse, draw, fill=lightgray!50, text=black, minimum height=1.3cm, minimum width=2.8cm, align=center},",        """Generate TikZ code for the MUD diagram."""

            r"  pnode/.style={rectangle, rounded corners=5pt, draw, fill=gray!70, text=black, minimum height=1.3cm, minimum width=3.5cm, align=center, inner xsep=0.3cm, inner ysep=0.2cm},",        tikz_lines = [

            r"  graybox/.style={rectangle, fill=lightgray!50, inner sep=4pt, minimum height=1.1cm, anchor=center, align=center, text centered},",            "\\begin{tikzpicture}[node distance=2cm and 2cm, auto]",

            "  % Arrow Styles",            f"\\node[draw, fill=blue!10, rounded corners] (title) at (0,0) {{\\textbf{{{operation.replace('_', ' ').title()} MUD}}}};",

            r"  solidarrow/.style={-Stealth, thick},",            ""

            r"  dashedarrow/.style={dashed, -Stealth, thick, gray},",        ]

            r"  textarrow/.style={align=center, inner sep=1pt}",

            r"]",        # Position strategies in a circle

            r"\tikzset{font=\linespread{0.8}\selectfont}",        angle_step = 360 / len(strategies) if strategies else 1

            "",        strategy_positions = {}

            f"% Diagram for: {operation.replace('_', ' ')}",

            ""        for i, strategy in enumerate(strategies):

        ]            angle = i * angle_step

            x = 4 * (angle * 3.14159 / 180)  # Convert to radians for positioning

        strategy_positions = {}            y = 3 * (angle * 3.14159 / 180)

        num_strategies = len(strategies)

            # Clean strategy name for display and escape underscores for LaTeX

        if num_strategies > 0:            display_name = strategy.replace('SAR_', '').replace('_', '\\_')

            radius = max(5, num_strategies * 1.0)            strategy_positions[strategy] = f"strategy_{i}"

            angle_step = 360 / num_strategies

            tikz_lines.append(f"\\node[draw, fill=green!10, rounded corners] ({strategy_positions[strategy]}) at ({x:.2f},{y:.2f}) {{{display_name}}};")

            for i, strategy in enumerate(strategies):

                node_id = f"P_{i}"        tikz_lines.append("")

                strategy_positions[strategy] = node_id

        # Add elaboration arrows

                angle = 90 - (i * angle_step)        arrow_count = 0

                x = radius * math.cos(math.radians(angle))        for pattern, relationships in pattern_relationships.items():

                y = radius * math.sin(math.radians(angle))            color = self._get_pattern_color(pattern)

            # Escape underscores in pattern names for LaTeX

                display_label = self._format_strategy_label(strategy)            escaped_pattern = pattern.replace('_', '\\_')

                tikz_lines.append(rf"\node[pnode] ({node_id}) at ({x:.2f},{y:.2f}) {{{display_label}}};")

            for elab in relationships:

        tikz_lines.append("")                base_pos = strategy_positions[elab['base_strategy']]

                elab_pos = strategy_positions[elab['elaborated_strategy']]

        arrow_count = 1

        for (base_strategy, elaborated_strategy), patterns in elaboration_relationships.items():                tikz_lines.append(f"\\draw[{color}, ->, thick] ({base_pos}) -- ({elab_pos})")

            if base_strategy in strategy_positions and elaborated_strategy in strategy_positions:                tikz_lines.append(f"    node[midway, above, font=\\footnotesize] {{{escaped_pattern}}};")

                base_pos = strategy_positions[base_strategy]

                elab_pos = strategy_positions[elaborated_strategy]                arrow_count += 1



                escaped_patterns = [p.replace('_', r'\_') for p in sorted(patterns)]        tikz_lines.extend([

                pattern_label = ", ".join(escaped_patterns)            "",

            "\\end{tikzpicture}"

                box_content = rf"P\textsubscript{{AlgEl}} {arrow_count}: PP-suff \\ ({pattern_label})"        ])



                tikz_lines.append(rf"\draw[solidarrow] ({base_pos}) -- node[graybox, midway, sloped] {{{box_content}}} ({elab_pos});")        return "\n".join(tikz_lines)



                arrow_count += 1    def _get_pattern_color(self, pattern: str) -> str:

        """Get color for pattern arrows."""

        tikz_lines.extend([        color_map = {

            "",            'base_decomposition': 'red',

            r"\end{tikzpicture}"            'incremental_counting': 'blue',

        ])            'counting_loop': 'green',

            'iterative_arithmetic': 'purple',

        return "\n".join(tikz_lines)            'value_adjustment': 'orange'

        }

        return color_map.get(pattern, 'black')

# --- ReportGenerator Class ---

    def _generate_mud_summary(self, operation: str, elaborations: List[Dict]) -> str:

class ReportGenerator:        """Generate a textual summary of the MUD."""

    """Generates reports in multiple formats from analysis results."""        if not elaborations:

            return f"No algorithmic elaborations detected for {operation}."

    def __init__(self, analysis_results: Dict[str, Any], mud_diagrams: Dict[str, Any] = None):

        self.analysis_results = analysis_results        summary_lines = [

        self.mud_diagrams = mud_diagrams or {}            f"Automated MUD Analysis for {operation.replace('_', ' ').title()}",

            "=" * 60,

    def generate_latex_report(self, strategy_name: str = None) -> str:            f"Total strategies analyzed: {len(set(e['base_strategy'] for e in elaborations) | set(e['elaborated_strategy'] for e in elaborations))}",

        """Generate a LaTeX report."""            f"Total elaborations detected: {len(elaborations)}",

        lines = [            "",

            r"\documentclass{article}",            "Key Patterns Identified:"

            r"\usepackage[utf8]{inputenc}",        ]

            r"\usepackage{geometry}",

            r"\usepackage{hyperref}",        # Count pattern usage

            r"\usepackage{booktabs}",        pattern_counts = defaultdict(int)

            r"\usepackage{xcolor}",        for elab in elaborations:

            r"\usepackage{tikz}",            for pattern in elab['shared_patterns']:

            r"\usetikzlibrary{positioning, shapes.geometric, arrows.meta, fit, backgrounds, calc, chains}",                pattern_counts[pattern] += 1

            r"\geometry{margin=1in}",

            r"\title{Algorithmic Elaboration Analysis Report}",        for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):

            f"\\date{{{self._get_timestamp()}}}",            summary_lines.append(f"  • {pattern}: {count} relationships")

            r"\begin{document}",

            r"\maketitle",        summary_lines.extend([

            r"\section{Overview}"            "",

        ]            "Notable Elaborations:"

        ])

        if self.mud_diagrams:

            lines.extend(self._generate_mud_diagrams_latex_section())        # Show high-confidence elaborations

        high_confidence = [e for e in elaborations if e['confidence'] > 0.5]

        lines.append(r"\end{document}")        for elab in high_confidence[:5]:  # Show top 5

        return "\n".join(lines)            summary_lines.append(f"  • {elab['base_strategy']} → {elab['elaborated_strategy']}")

            summary_lines.append(f"    Shared: {', '.join(elab['shared_patterns'])} (confidence: {elab['confidence']:.2f})")

    def _generate_mud_diagrams_latex_section(self) -> List[str]:

        """Generate a LaTeX section containing all MUD diagrams."""        if len(high_confidence) > 5:

        lines = [r"\section{Meaning-Use Diagrams}"]            summary_lines.append(f"  ... and {len(high_confidence) - 5} more high-confidence relationships")



        for operation, diagram_data in self.mud_diagrams.items():        return "\n".join(summary_lines)

            operation_title = operation.replace('_', ' ').title()

            lines.append(f"\\subsection{{{operation_title}}}")class ReportGenerator:

            lines.append(r"\begin{center}")    """Generates reports in multiple formats from analysis results."""

            lines.append(diagram_data.get('tikz_diagram', ''))

            lines.append(r"\end{center}")    def __init__(self, analysis_results: Dict[str, Any], mud_diagrams: Dict[str, Any] = None):

            lines.append(r"\newpage")        self.analysis_results = analysis_results

        self.mud_diagrams = mud_diagrams or {}

        return lines

    def generate_markdown_report(self, strategy_name: str = None) -> str:

    def _get_timestamp(self) -> str:        """Generate a Markdown report for a specific strategy or general overview."""

        """Get current timestamp for reports."""        lines = [

        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")            "# Algorithmic Elaboration Analysis Report",

            "",

def main():            f"Generated on: {self._get_timestamp()}",

    """Main entry point for MUD generation."""            "",

    parser = argparse.ArgumentParser(description="Consolidated MUD Generator")            "## Overview",

    subparsers = parser.add_subparsers(dest='command', help='Available commands')            "",

            f"- **Computational Patterns Detected**: {len(self.analysis_results.get('patterns', {}))}",

    analyze_parser = subparsers.add_parser('analyze', help='Analyze automata for patterns')            f"- **Algorithmic Elaborations Found**: {len(self.analysis_results.get('elaborations', []))}",

    analyze_parser.add_argument('--automata-dir', default='src/automata', help='Directory containing automata')            f"- **MUD Diagrams Generated**: {len(self.mud_diagrams)}",

            ""

    generate_parser = subparsers.add_parser('generate', help='Generate MUD diagrams')        ]

    generate_parser.add_argument('--input', required=True, help='Input analysis results JSON file')

    generate_parser.add_argument('--output', default='diagrams/', help='Output directory for diagrams')        if strategy_name:

            lines.extend(self._generate_strategy_report(strategy_name))

    report_parser = subparsers.add_parser('report', help='Generate reports')        else:

    report_parser.add_argument('--input', required=True, help='Input analysis results JSON file')            lines.extend(self._generate_overview_report())

    report_parser.add_argument('--strategy', help='Specific strategy to analyze')

    report_parser.add_argument('--format', choices=['markdown', 'latex', 'html'], default='latex', help='Output format')        # Add MUD diagrams section if diagrams are available

    report_parser.add_argument('--output', help='Output file (default: stdout)')        if self.mud_diagrams:

            lines.extend(self._generate_mud_diagrams_markdown_section())

    args = parser.parse_args()

        return "\n".join(lines)

    if args.command == 'analyze':

        analyzer = AutomatonAnalyzer(args.automata_dir)    def _generate_strategy_report(self, strategy_name: str) -> List[str]:

        results = analyzer.analyze_all_automata()        """Generate a detailed report for a specific strategy."""

        lines = [

        output_file = 'automated_mud_results.json'            f"## Strategy Analysis: {strategy_name}",

        with open(output_file, 'w') as f:            "",

            json.dump(results, f, indent=2)            "### Computational Patterns Used",

        ]

        print(f"✅ Analysis complete. Results saved to {output_file}", file=sys.stderr)

        patterns = self.analysis_results.get('strategy_patterns', {}).get(strategy_name, [])

    elif args.command == 'generate':        if patterns:

        with open(args.input, 'r') as f:            for pattern in patterns:

            data = json.load(f)                pattern_info = self.analysis_results.get('patterns', {}).get(pattern, {})

                lines.append(f"- **{pattern}** ({pattern_info.get('type', 'unknown')})")

        analysis_results = data.get('analysis_results', data)                lines.append(f"  - Used by {pattern_info.get('usage_count', 0)} other strategies")

        else:

        mud_generator = MUDGenerator(analysis_results)            lines.append("- No patterns detected")

        mud_diagrams = mud_generator.generate_mud_diagrams()

        lines.extend([

        combined_results = {            "",

            'analysis_results': analysis_results,            "### Algorithmic Elaborations",

            'mud_diagrams': mud_diagrams            "",

        }            "#### As Base Strategy:"

        ])

        output_file = os.path.join(args.output, 'mud_diagrams.json')

        os.makedirs(args.output, exist_ok=True)        base_elabs = [e for e in self.analysis_results.get('elaborations', [])

        with open(output_file, 'w') as f:                     if e['base_strategy'] == strategy_name]

            json.dump(combined_results, f, indent=2)        if base_elabs:

            for elab in base_elabs:

        print(f"✅ MUD diagrams generated. Results saved to {output_file}", file=sys.stderr)                lines.extend([

                    f"- **Elaborates** → {elab['elaborated_strategy']}",

    elif args.command == 'report':                    f"  - Shared patterns: {', '.join(elab['shared_patterns'])}",

        with open(args.input, 'r') as f:                    f"  - Confidence: {elab['confidence']:.2f}",

            data = json.load(f)                    ""

                ])

        analysis_results = data.get('analysis_results', data)        else:

        mud_diagrams = data.get('mud_diagrams', {})            lines.append("- None found")



        report_gen = ReportGenerator(analysis_results, mud_diagrams)        lines.extend([

            "",

        if args.format == 'latex':            "#### As Elaborated Strategy:"

            report = report_gen.generate_latex_report(args.strategy)        ])

        else:

            print(f"Format '{args.format}' not fully implemented in this version.", file=sys.stderr)        elab_elabs = [e for e in self.analysis_results.get('elaborations', [])

            report = ""                     if e['elaborated_strategy'] == strategy_name]

        if elab_elabs:

        if args.output:            for elab in elab_elabs:

            with open(args.output, 'w') as f:                lines.extend([

                f.write(report)                    f"- **Elaborated from** ← {elab['base_strategy']}",

            print(f"✅ Report saved to {args.output}", file=sys.stderr)                    f"  - Shared patterns: {', '.join(elab['shared_patterns'])}",

        else:                    f"  - Confidence: {elab['confidence']:.2f}",

            print(report)                    ""

                ])

    else:        else:

        parser.print_help()            lines.append("- None found")



        return lines

if __name__ == "__main__":

    main()    def _generate_overview_report(self) -> List[str]:

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