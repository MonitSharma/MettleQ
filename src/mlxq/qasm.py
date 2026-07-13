from __future__ import annotations

import ast
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


_REG_RE = re.compile(r"^(qreg|creg)\s+([A-Za-z_]\w*)\[(\d+)\]\s*;$")
_GATE_HEADER_RE = re.compile(
    r"^gate\s+([A-Za-z_]\w*)\s*(?:\(([^)]*)\))?\s+([^{}]+?)\s*\{\s*$"
)
_APPLICATION_RE = re.compile(
    r"^([A-Za-z_]\w*)\s*(?:\((.*)\))?\s+([^;]+?)\s*;$"
)
_INDEXED_OPERAND_RE = re.compile(r"^([A-Za-z_]\w*)\[(\d+)\]$")
_MEASURE_RE = re.compile(r"^measure\s+(.+?)\s*->\s*(.+?)\s*;$")
_INCLUDE_RE = re.compile(r'^include\s+"([^"]+)"\s*;$')

_SUPPORTED_INCLUDES = {"qelib1.inc", "stdgates.inc"}

# (parameter count, wire count) after alias normalization.
_GATE_SIGNATURES: Dict[str, Tuple[int, int]] = {
    "H": (0, 1), "X": (0, 1), "Y": (0, 1), "Z": (0, 1),
    "S": (0, 1), "SDG": (0, 1), "T": (0, 1), "TDG": (0, 1),
    "SX": (0, 1), "RX": (1, 1), "RY": (1, 1), "RZ": (1, 1),
    "U1": (1, 1), "U2": (2, 1), "U3": (3, 1),
    "CNOT": (0, 2), "CZ": (0, 2), "SWAP": (0, 2), "ISWAP": (0, 2),
    "CH": (0, 2), "CPHASE": (1, 2), "CRX": (1, 2),
    "CRY": (1, 2), "CRZ": (1, 2), "CCX": (0, 3), "CSWAP": (0, 3),
}


class QASMParseError(ValueError):
    """A precise failure while parsing Qupertino's supported OpenQASM subset."""

    def __init__(self, path: Path, line: int, message: str, source: str = ""):
        detail = f"{path}:{line}: {message}"
        if source:
            detail += f" [{source.strip()}]"
        super().__init__(detail)
        self.path = path
        self.line = int(line)
        self.source = source


def _fail(context: Tuple[Path, int, str], message: str):
    path, line, source = context
    raise QASMParseError(path, line, message, source)


def _expression_tree(expression: str, context: Tuple[Path, int, str]) -> ast.Expression:
    if not expression.strip():
        _fail(context, "empty parameter expression")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        _fail(context, f"malformed expression {expression!r}")
    assert isinstance(tree, ast.Expression)
    return tree


def _validate_expression_node(
    node: ast.AST,
    allowed_symbols: Sequence[str],
    context: Tuple[Path, int, str],
):
    if isinstance(node, ast.Expression):
        _validate_expression_node(node.body, allowed_symbols, context)
        return
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return
    if isinstance(node, ast.Name):
        if node.id == "pi" or node.id in allowed_symbols:
            return
        _fail(context, f"unknown symbol {node.id!r} in expression")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        _validate_expression_node(node.operand, allowed_symbols, context)
        return
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
    ):
        _validate_expression_node(node.left, allowed_symbols, context)
        _validate_expression_node(node.right, allowed_symbols, context)
        return
    _fail(context, f"unsupported expression syntax {ast.dump(node, include_attributes=False)}")


def _evaluate_expression_node(
    node: ast.AST,
    bindings: Dict[str, float],
    context: Tuple[Path, int, str],
) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate_expression_node(node.body, bindings, context)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id == "pi":
            return math.pi
        if node.id in bindings:
            return float(bindings[node.id])
        _fail(context, f"unbound symbol {node.id!r} in expression")
    if isinstance(node, ast.UnaryOp):
        value = _evaluate_expression_node(node.operand, bindings, context)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
    if isinstance(node, ast.BinOp):
        left = _evaluate_expression_node(node.left, bindings, context)
        right = _evaluate_expression_node(node.right, bindings, context)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0.0:
                _fail(context, "division by zero in expression")
            return left / right
    _fail(context, "unsupported expression")


def _evaluate_expression(
    expression: str,
    bindings: Optional[Dict[str, float]],
    context: Tuple[Path, int, str],
) -> float:
    values = bindings or {}
    tree = _expression_tree(expression, context)
    _validate_expression_node(tree, tuple(values), context)
    value = _evaluate_expression_node(tree, values, context)
    if not math.isfinite(value):
        _fail(context, "expression result must be finite")
    return float(value)


def _split_parameters(raw: str, context: Tuple[Path, int, str]) -> List[str]:
    if not raw.strip():
        return []
    parts = [part.strip() for part in raw.split(",")]
    if any(not part for part in parts):
        _fail(context, "empty parameter between commas")
    return parts


def _parse_application(
    source: str,
    context: Tuple[Path, int, str],
) -> Tuple[str, List[str], List[str]]:
    match = _APPLICATION_RE.fullmatch(source)
    if not match:
        _fail(context, "unsupported or malformed statement")
    name = _normalize_gate(match.group(1))
    parameters = _split_parameters(match.group(2) or "", context)
    targets = [target.strip() for target in match.group(3).split(",")]
    if any(not target for target in targets):
        _fail(context, "empty wire operand between commas")
    return name, parameters, targets


def _normalize_gate(gate: str) -> str:
    aliases = {
        "cnot": "CNOT", "cx": "CNOT", "cz": "CZ", "swap": "SWAP",
        "iswap": "ISWAP", "x": "X", "y": "Y", "z": "Z", "h": "H",
        "ch": "CH", "s": "S", "sdg": "SDG", "t": "T", "tdg": "TDG",
        "sx": "SX", "rx": "RX", "ry": "RY", "rz": "RZ", "p": "U1",
        "phase": "U1", "u1": "U1", "u2": "U2", "u3": "U3", "u": "U",
        "crx": "CRX", "cry": "CRY", "crz": "CRZ", "cu1": "CU1",
        "cp": "CP", "cphase": "CP", "ccx": "CCX", "toffoli": "CCX",
        "cswap": "CSWAP", "fredkin": "CSWAP",
    }
    return aliases.get(gate.lower(), gate.upper())


def _normalize_primitive(name: str) -> str:
    if name == "U":
        return "U3"
    if name in ("CU1", "CP"):
        return "CPHASE"
    return name


def _validate_primitive(
    name: str,
    parameters: Sequence[float],
    wires: Sequence[int],
    context: Tuple[Path, int, str],
) -> str:
    normalized = _normalize_primitive(name)
    signature = _GATE_SIGNATURES.get(normalized)
    if signature is None:
        _fail(context, f"unsupported gate {name!r}")
    parameter_count, wire_count = signature
    if len(parameters) != parameter_count:
        _fail(
            context,
            f"gate {name} expects {parameter_count} parameters, got {len(parameters)}",
        )
    if len(wires) != wire_count:
        _fail(context, f"gate {name} expects {wire_count} wires, got {len(wires)}")
    if len(set(wires)) != len(wires):
        _fail(context, f"gate {name} requires distinct wires")
    return normalized


def _resolve_path(path: str) -> Path:
    requested = Path(path)
    candidates = [requested]
    if not requested.exists():
        relative = Path(requested.name)
        here = Path(__file__).resolve()
        for parent in list(here.parents)[:5]:
            candidates.extend(
                [
                    parent / "datasets" / "qasm" / "local" / relative,
                    parent / "src" / "qasm_circuits" / relative,
                    parent / "qasm_circuits" / relative,
                ]
            )
        cwd = Path.cwd()
        candidates.extend(
            [
                cwd / "datasets" / "qasm" / "local" / relative,
                cwd / "src" / "qasm_circuits" / relative,
                cwd / "qasm_circuits" / relative,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"QASM file not found: {path}")


def _resolve_indexed_operand(
    operand: str,
    registers: Dict[str, Tuple[int, int]],
    kind: str,
    context: Tuple[Path, int, str],
) -> int:
    match = _INDEXED_OPERAND_RE.fullmatch(operand)
    if not match:
        _fail(context, f"expected indexed {kind} operand, got {operand!r}")
    register, index = match.group(1), int(match.group(2))
    if register not in registers:
        _fail(context, f"unknown {kind} register {register!r}")
    base, size = registers[register]
    if index >= size:
        _fail(context, f"{kind} index {register}[{index}] is out of range for size {size}")
    return base + index


def _validate_barrier(
    source: str,
    qregisters: Dict[str, Tuple[int, int]],
    context: Tuple[Path, int, str],
):
    match = re.fullmatch(r"barrier\s+([^;]+)\s*;", source)
    if not match:
        _fail(context, "malformed barrier statement")
    for operand in (part.strip() for part in match.group(1).split(",")):
        if operand in qregisters:
            continue
        _resolve_indexed_operand(operand, qregisters, "quantum", context)


def _validate_measurement(
    source: str,
    qregisters: Dict[str, Tuple[int, int]],
    cregisters: Dict[str, Tuple[int, int]],
    context: Tuple[Path, int, str],
):
    match = _MEASURE_RE.fullmatch(source)
    if not match:
        _fail(context, "malformed measurement statement")
    quantum, classical = match.group(1).strip(), match.group(2).strip()
    if quantum in qregisters or classical in cregisters:
        if quantum not in qregisters or classical not in cregisters:
            _fail(context, "whole-register measurement requires quantum and classical registers")
        if qregisters[quantum][1] != cregisters[classical][1]:
            _fail(context, "measurement register sizes must match")
        return
    _resolve_indexed_operand(quantum, qregisters, "quantum", context)
    _resolve_indexed_operand(classical, cregisters, "classical", context)


def _parse_gate_body(
    body_lines: List[Tuple[int, str]],
    path: Path,
    parameter_names: Sequence[str],
    argument_names: Sequence[str],
) -> List[Dict[str, Any]]:
    body: List[Dict[str, Any]] = []
    allowed_parameters = tuple(parameter_names)
    allowed_arguments = set(argument_names)
    for line, source in body_lines:
        context = (path, line, source)
        name, parameter_tokens, targets = _parse_application(source, context)
        for expression in parameter_tokens:
            tree = _expression_tree(expression, context)
            _validate_expression_node(tree, allowed_parameters, context)
        for target in targets:
            if target not in allowed_arguments:
                _fail(context, f"unknown gate wire argument {target!r}")
        body.append(
            {
                "name": name,
                "parameter_tokens": parameter_tokens,
                "arguments": targets,
                "context": context,
            }
        )
    return body


def _expand_gate(
    name: str,
    wires: List[int],
    parameters: List[float],
    gate_defs: Dict[str, Dict[str, Any]],
    context: Tuple[Path, int, str],
    stack: Tuple[str, ...] = (),
) -> List[Dict[str, Any]]:
    definition = gate_defs[name]
    formal_arguments: List[str] = definition["arguments"]
    formal_parameters: List[str] = definition["parameters"]
    if len(wires) != len(formal_arguments):
        _fail(context, f"gate {name} expects {len(formal_arguments)} wires, got {len(wires)}")
    if len(parameters) != len(formal_parameters):
        _fail(
            context,
            f"gate {name} expects {len(formal_parameters)} parameters, got {len(parameters)}",
        )
    if name in stack:
        _fail(context, f"recursive gate definition involving {name}")
    argument_map = dict(zip(formal_arguments, wires))
    parameter_map = dict(zip(formal_parameters, parameters))
    operations: List[Dict[str, Any]] = []
    for body_op in definition["body"]:
        body_context = body_op["context"]
        target_wires = [argument_map[arg] for arg in body_op["arguments"]]
        values = [
            _evaluate_expression(expression, parameter_map, body_context)
            for expression in body_op["parameter_tokens"]
        ]
        body_name = body_op["name"]
        if body_name in gate_defs:
            operations.extend(
                _expand_gate(
                    body_name,
                    target_wires,
                    values,
                    gate_defs,
                    body_context,
                    stack + (name,),
                )
            )
            continue
        normalized = _validate_primitive(body_name, values, target_wires, body_context)
        operations.append(
            {"name": normalized, "wires": target_wires, "parameters": values}
        )
    return operations


def parse_qasm_file(path: str) -> Tuple[int, List[Dict[str, Any]]]:
    """Parse Qupertino's strict unitary subset of OpenQASM 2.0.

    Supported:
    - ``OPENQASM 2.0`` and built-in ``qelib1.inc``/``stdgates.inc`` includes;
    - quantum/classical register declarations and validated primitive gates;
    - register-wide single-qubit gates;
    - non-recursive user gates with nested calls and bound parameter arithmetic;
    - numeric literals, ``pi``, bound symbols, parentheses, unary ``+``/``-``,
      and binary ``+``, ``-``, ``*``, ``/`` expressions;
    - barriers as unitary no-ops and terminal measurements as output markers.

    Terminal measurements are validated but omitted because this API returns a
    unitary operation list. Mid-circuit measurement, reset, classical control,
    opaque gates, arbitrary includes, and other statements raise
    :class:`QASMParseError` with source line context.
    """
    resolved = _resolve_path(path)
    raw_lines = resolved.read_text(encoding="utf-8", errors="strict").splitlines()
    lines = [
        (line_number, raw.split("//", 1)[0].strip())
        for line_number, raw in enumerate(raw_lines, start=1)
    ]

    qregisters: Dict[str, Tuple[int, int]] = {}
    cregisters: Dict[str, Tuple[int, int]] = {}
    gate_defs: Dict[str, Dict[str, Any]] = {}
    operations: List[Dict[str, Any]] = []
    n_qubits = 0
    n_classical = 0
    seen_version = False
    terminal_measurement_started = False

    index = 0
    while index < len(lines):
        line_number, source = lines[index]
        index += 1
        if not source:
            continue
        context = (resolved, line_number, source)

        if not seen_version:
            if source != "OPENQASM 2.0;":
                _fail(context, "expected 'OPENQASM 2.0;' as the first statement")
            seen_version = True
            continue
        if source.startswith("OPENQASM"):
            _fail(context, "duplicate or unsupported OPENQASM version declaration")

        include_match = _INCLUDE_RE.fullmatch(source)
        if include_match:
            include = include_match.group(1)
            if include not in _SUPPORTED_INCLUDES:
                _fail(context, f"include {include!r} is not supported")
            continue
        if source.startswith("include"):
            _fail(context, "malformed include statement")

        register_match = _REG_RE.fullmatch(source)
        if register_match:
            kind, name, size_text = register_match.groups()
            size = int(size_text)
            if size < 1:
                _fail(context, "register size must be positive")
            if name in qregisters or name in cregisters:
                _fail(context, f"duplicate register name {name!r}")
            if kind == "qreg":
                qregisters[name] = (n_qubits, size)
                n_qubits += size
            else:
                cregisters[name] = (n_classical, size)
                n_classical += size
            continue
        if source.startswith(("qreg", "creg")):
            _fail(context, "malformed register declaration")

        if source.startswith("gate "):
            if terminal_measurement_started:
                _fail(context, "measurements must be terminal")
            header = source
            while "{" not in header:
                if index >= len(lines):
                    _fail(context, "unterminated gate definition header")
                next_line, next_source = lines[index]
                index += 1
                if next_source:
                    header += " " + next_source
            if "}" in header:
                _fail(context, "single-line gate definitions are not supported")
            header_match = _GATE_HEADER_RE.fullmatch(header)
            if not header_match:
                _fail(context, "malformed gate definition header")
            gate_name = _normalize_gate(header_match.group(1))
            parameter_names = [
                item.strip()
                for item in (header_match.group(2) or "").split(",")
                if item.strip()
            ]
            argument_names = [
                item.strip() for item in header_match.group(3).split(",") if item.strip()
            ]
            if len(set(parameter_names)) != len(parameter_names):
                _fail(context, "duplicate gate parameter name")
            if not argument_names or len(set(argument_names)) != len(argument_names):
                _fail(context, "gate wire arguments must be non-empty and unique")
            if gate_name in gate_defs or gate_name in _GATE_SIGNATURES:
                _fail(context, f"duplicate or reserved gate name {gate_name!r}")

            body_lines: List[Tuple[int, str]] = []
            closed = False
            while index < len(lines):
                body_line, body_source = lines[index]
                index += 1
                if not body_source:
                    continue
                if body_source == "}":
                    closed = True
                    break
                if "{" in body_source or "}" in body_source:
                    _fail((resolved, body_line, body_source), "malformed gate body braces")
                body_lines.append((body_line, body_source))
            if not closed:
                _fail(context, "unterminated gate definition")
            gate_defs[gate_name] = {
                "parameters": parameter_names,
                "arguments": argument_names,
                "body": _parse_gate_body(
                    body_lines, resolved, parameter_names, argument_names
                ),
            }
            continue

        lowered = source.lower()
        if lowered.startswith("opaque "):
            _fail(context, "opaque gates are not supported")
        if lowered.startswith("reset"):
            _fail(context, "reset is not supported by the unitary QASM importer")
        if lowered.startswith("if"):
            _fail(context, "classical control is not supported by the unitary QASM importer")
        if lowered.startswith("barrier"):
            _validate_barrier(source, qregisters, context)
            continue
        if lowered.startswith("measure"):
            _validate_measurement(source, qregisters, cregisters, context)
            terminal_measurement_started = True
            continue
        if terminal_measurement_started:
            _fail(context, "measurements must be terminal; quantum work follows measurement")

        gate_name, parameter_tokens, targets = _parse_application(source, context)
        parameter_values = [
            _evaluate_expression(expression, {}, context) for expression in parameter_tokens
        ]

        # A bare register is supported only for a primitive single-qubit gate.
        if len(targets) == 1 and targets[0] in qregisters:
            if gate_name in gate_defs:
                _fail(context, "user-defined gates require explicit indexed wires")
            normalized = _normalize_primitive(gate_name)
            signature = _GATE_SIGNATURES.get(normalized)
            if signature is None or signature[1] != 1:
                _fail(context, "register-wide application requires a single-qubit gate")
            base, size = qregisters[targets[0]]
            for offset in range(size):
                wire = base + offset
                checked = _validate_primitive(
                    gate_name, parameter_values, [wire], context
                )
                operations.append(
                    {"name": checked, "wires": [wire], "parameters": parameter_values}
                )
            continue

        wires = [
            _resolve_indexed_operand(target, qregisters, "quantum", context)
            for target in targets
        ]
        if gate_name in gate_defs:
            operations.extend(
                _expand_gate(
                    gate_name, wires, parameter_values, gate_defs, context
                )
            )
            continue
        normalized = _validate_primitive(gate_name, parameter_values, wires, context)
        operations.append(
            {"name": normalized, "wires": wires, "parameters": parameter_values}
        )

    if not seen_version:
        raise QASMParseError(resolved, 1, "missing OPENQASM 2.0 version declaration")
    return n_qubits, operations


__all__ = ["QASMParseError", "parse_qasm_file"]
