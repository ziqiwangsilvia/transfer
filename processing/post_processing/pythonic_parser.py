import ast
import json
import re
from typing import Any, List, Optional

from processing.post_processing.utils import ToolCall

# Text fixers — applied before AST parsing


def _normalize_quotes(text: str) -> str:
    """Replace Unicode smart/curly quotes with ASCII equivalents."""
    return (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )


def _fix_missing_closing_quote(text: str) -> str:
    """
    Fix model outputs where a single-quoted JSON-object arg is missing its closing quote.

    Gemma3 sometimes emits:
        data_source='{...json...}, next_arg='...'
    when it should be:
        data_source='{...json...}', next_arg='...'
    """
    return re.sub(r"(=')(\{[^']+?\})(,\s*\w+=)", r"\1\2'\3", text)


def _fix_unquoted_string_args(text: str) -> str:
    """
    Wrap bare (unquoted) function kwarg values in single quotes.
    Skips content inside {...} dict literals (handled by _fix_dict_equals_syntax).
    """
    result = []
    i = 0
    depth_brace = 0

    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth_brace += 1
            result.append(ch)
            i += 1
        elif ch == "}":
            depth_brace -= 1
            result.append(ch)
            i += 1
        elif ch in ('"', "'"):
            quote = ch
            result.append(ch)
            i += 1
            while i < len(text):
                c = text[i]
                result.append(c)
                i += 1
                if c == "\\":
                    if i < len(text):
                        result.append(text[i])
                        i += 1
                elif c == quote:
                    break
        elif depth_brace == 0:
            m = re.match(r"(\w+=)([^'\"\d\[{(\s][^,)]*?)(?=(,\s*\w+=|\)))", text[i:])
            if m:
                result.append(m.group(1) + "'" + m.group(2).strip() + "'")
                i += len(m.group(0))
            else:
                result.append(ch)
                i += 1
        else:
            result.append(ch)
            i += 1

    return "".join(result)


def _fix_dict_equals_syntax(text: str) -> str:
    """
    Fix bare key=value syntax inside dict literals, converting to 'key': value.

    Models sometimes output: {source='transactions', data_type='spending'}
    instead of: {'source': 'transactions', 'data_type': 'spending'}
    """
    result = []
    i = 0
    depth_paren = 0
    depth_brace = 0

    while i < len(text):
        ch = text[i]
        if ch == "(":
            depth_paren += 1
            result.append(ch)
            i += 1
        elif ch == ")":
            depth_paren -= 1
            result.append(ch)
            i += 1
        elif ch == "{":
            depth_brace += 1
            result.append(ch)
            i += 1
        elif ch == "}":
            depth_brace -= 1
            result.append(ch)
            i += 1
        elif ch in ('"', "'"):
            quote = ch
            result.append(ch)
            i += 1
            while i < len(text):
                c = text[i]
                result.append(c)
                i += 1
                if c == "\\":
                    if i < len(text):
                        result.append(text[i])
                        i += 1
                elif c == quote:
                    break
        elif depth_brace > 0:
            m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)", text[i:])
            if m:
                result.append(f"'{m.group(1)}': ")
                i += len(m.group(0))
            else:
                result.append(ch)
                i += 1
        else:
            result.append(ch)
            i += 1

    return "".join(result)


def _apply_pythonic_fixers(text: str) -> str:
    """Apply all pre-parse fixers to a pythonic call string."""
    text = _normalize_quotes(text)
    text = _fix_dict_equals_syntax(text)
    text = _fix_missing_closing_quote(text)
    text = _fix_unquoted_string_args(text)
    return text


# AST evaluation helpers


def _eval_ast_node(node: ast.AST) -> Any:
    """Recursively evaluate an AST node to a plain Python value."""
    if isinstance(node, ast.Constant):
        val = node.value
        if isinstance(val, str):
            stripped = val.strip()
            if stripped.startswith(("{", "[")):
                try:
                    return json.loads(stripped)
                except json.JSONDecodeError:
                    pass
        return val
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval_ast_node(el) for el in node.elts]
    if isinstance(node, ast.Dict):
        return {
            _eval_ast_node(k): _eval_ast_node(v)
            for k, v in zip(node.keys, node.values)
            if k is not None
        }
    if isinstance(node, ast.Call):
        return {
            kw.arg: _eval_ast_node(kw.value)
            for kw in node.keywords
            if kw.arg is not None
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_ast_node(node.operand)
    try:
        return ast.literal_eval(node)
    except Exception:
        return ast.unparse(node)


def _calls_from_tree(tree: ast.Expression) -> Optional[List[ToolCall]]:
    """Extract a ToolCall list from a parsed AST expression, or None if not a call."""
    body = tree.body
    if isinstance(body, ast.List):
        top_level_calls = [el for el in body.elts if isinstance(el, ast.Call)]
    elif isinstance(body, ast.Call):
        top_level_calls = [body]
    else:
        return None

    calls = []
    for call_node in top_level_calls:
        try:
            arguments = {
                kw.arg: _eval_ast_node(kw.value)
                for kw in call_node.keywords
                if kw.arg is not None
            }
            calls.append(
                ToolCall(name=ast.unparse(call_node.func), parameters=arguments)
            )
        except Exception:
            continue
    return calls or None


def _try_parse_pythonic(text: str) -> Optional[List[ToolCall]]:
    """Attempt ast.parse on text; return ToolCalls or None on SyntaxError."""
    try:
        return _calls_from_tree(ast.parse(text, mode="eval"))
    except SyntaxError:
        return None


# Main functions


def _extract_pythonic_block(text: str) -> Optional[str]:
    """
    Scan text for a balanced [...] block containing a function call.

    Handles mixed outputs like "Okay, let me show you.\n\n[show_pie_chart(...)]"
    """
    start = text.find("[")
    while start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    if "(" in candidate:
                        return candidate
                    break
        start = text.find("[", start + 1)
    return None


def parse_pythonic_output(output: str) -> Optional[List[ToolCall]]:
    """
    Parse Pythonic function-call syntax into a list of ToolCalls.

    Handles:
    - Pure pythonic:           [show_pie_chart(...)]
    - NLP prefix + pythonic:  "Okay...\n\n[show_pie_chart(...)]"
    - Fenced tool_code block:  ```tool_code\nshow_pie_chart(...)\n```
    - Dict = syntax:           data_source={key='val'} instead of {'key': 'val'}

    Returns None if no tool call can be extracted.
    """
    # Extract the [...] block first (if present) so NLP-prefix apostrophes
    # don't confuse the dict-equals fixer's quote scanner.
    block = _extract_pythonic_block(_normalize_quotes(output))
    candidates = [block] if block else []

    # Also try a fence-stripped version — bare (unwrapped) call lines
    stripped = re.sub(r"```[\w_]*", "", output).replace("```", "").strip()
    if stripped != output.strip():
        call_lines = [
            line.strip()
            for line in stripped.splitlines()
            if "(" in line and ")" in line
        ]
        if call_lines:
            candidates.append("[" + call_lines[-1] + "]")
    candidates.append(output)  # full text as last resort

    for candidate in candidates:
        fixed = _apply_pythonic_fixers(candidate)
        result = _try_parse_pythonic(fixed)
        if result:
            return result

    return None
