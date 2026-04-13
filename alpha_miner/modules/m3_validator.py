from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .config_loader import load_operator_config


_IDENTIFIER = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")
_FUNCTION = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
_RESERVED = {"if", "else", "true", "false", "nan"}


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    errors: list[str]
    warnings: list[str]
    depth: int
    complexity: int
    operators: list[str]
    fields: list[str]


class ExpressionValidator:
    def __init__(
        self,
        operators: Iterable[str] | None = None,
        fields: Iterable[str] | None = None,
        min_depth: int = 1,
        max_depth: int = 5,
        max_length: int = 450,
        max_complexity: int = 32,
    ):
        config = load_operator_config()
        configured_ops = []
        for values in (config.get("operators") or {}).values():
            configured_ops.extend(values or [])
        self.operators = set(operators or configured_ops)
        self.fields = set(fields or config.get("fields") or [])
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.max_length = max_length
        self.max_complexity = max_complexity

    def validate(self, expression: str) -> ValidationResult:
        expression = str(expression or "").strip()
        errors: list[str] = []
        warnings: list[str] = []
        if not expression:
            errors.append("Expression is empty.")
        if len(expression) > self.max_length:
            errors.append(f"Expression length {len(expression)} exceeds max_length {self.max_length}.")
        if not self._balanced_parentheses(expression):
            errors.append("Parentheses are not balanced.")

        operators = sorted(set(_FUNCTION.findall(expression)))
        hallucinated = [operator for operator in operators if operator not in self.operators]
        if hallucinated:
            errors.append(f"Unknown operators: {', '.join(hallucinated)}.")

        identifiers = set(_IDENTIFIER.findall(expression))
        field_names = sorted(identifiers - set(operators) - _RESERVED)
        unknown_fields = [field for field in field_names if field not in self.fields and not field.isupper()]
        if unknown_fields:
            errors.append(f"Unknown fields: {', '.join(unknown_fields)}.")

        depth = self._max_depth(expression)
        if depth < self.min_depth:
            warnings.append(f"Expression depth {depth} is shallow.")
        if depth > self.max_depth:
            errors.append(f"Expression depth {depth} exceeds max_depth {self.max_depth}.")

        complexity = len(operators) + expression.count("+") + expression.count("-") + expression.count("*") + expression.count("/")
        if complexity > self.max_complexity:
            errors.append(f"Expression complexity {complexity} exceeds max_complexity {self.max_complexity}.")

        return ValidationResult(
            is_valid=not errors,
            errors=errors,
            warnings=warnings,
            depth=depth,
            complexity=complexity,
            operators=operators,
            fields=field_names,
        )

    @staticmethod
    def _balanced_parentheses(expression: str) -> bool:
        depth = 0
        for char in expression:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth < 0:
                    return False
        return depth == 0

    @staticmethod
    def _max_depth(expression: str) -> int:
        depth = 0
        max_depth = 0
        for char in expression:
            if char == "(":
                depth += 1
                max_depth = max(max_depth, depth)
            elif char == ")":
                depth = max(0, depth - 1)
        return max_depth
