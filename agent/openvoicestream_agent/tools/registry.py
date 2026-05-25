"""Tool registry + ``@tool`` decorator.

Builds OpenAI-style ``tools[]`` schemas from Python type hints and
dispatches function calls (sync or async) with timeout + error
isolation. Designed for local, in-process tools — every entry is
trusted code in the same Python process (no sandboxing, no MCP).
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import types
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema (OpenAI-style)
    fn: Callable[..., Any]
    timeout_s: float = 10.0


def _py_type_to_schema(t: Any) -> dict[str, Any]:
    """Map a Python type hint to a JSON Schema fragment.

    Supports: ``str`` / ``int`` / ``float`` / ``bool`` / ``list`` /
    ``dict`` / ``Literal[...]`` / ``Optional[T]`` / ``T | None``.
    Anything unknown falls back to ``{"type": "string"}`` — the LLM
    will still send something stringy that ``dispatch`` will reject if
    it's wrong.
    """
    origin = get_origin(t)
    args = get_args(t)

    # Literal[...] → enum
    if origin is Literal:
        sample = args[0]
        if isinstance(sample, bool):
            jtype = "boolean"
        elif isinstance(sample, int):
            jtype = "integer"
        elif isinstance(sample, float):
            jtype = "number"
        else:
            jtype = "string"
        return {"type": jtype, "enum": list(args)}

    # Optional[T] / Union[T, None] / ``T | None`` → unwrap and recurse.
    # Python 3.10+ ``X | Y`` uses ``types.UnionType``, not typing.Union;
    # check both.
    if origin is Union or origin is types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _py_type_to_schema(non_none[0])
        # Real union — fall back to string.
        return {"type": "string"}

    # Parameterized generics: list[X], dict[K, V]
    if origin in (list, tuple, set, frozenset):
        item_schema = (
            _py_type_to_schema(args[0]) if args else {"type": "string"}
        )
        return {"type": "array", "items": item_schema}
    if origin is dict:
        return {"type": "object"}

    # Plain builtins
    if t is str:
        return {"type": "string"}
    if t is bool:
        return {"type": "boolean"}
    if t is int:
        return {"type": "integer"}
    if t is float:
        return {"type": "number"}
    if t is list:
        return {"type": "array"}
    if t is dict:
        return {"type": "object"}

    return {"type": "string"}


class ToolRegistry:
    """Holds a set of registered tools, exports their OpenAI schemas, and
    dispatches calls. A module-level :data:`default_registry` is the
    one builtin tools attach to; tests may construct their own."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def tool(
        self,
        *,
        name: str | None = None,
        description: str = "",
        timeout_s: float = 10.0,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator: register ``fn`` as a tool.

        Parameter schema is built from type hints, excluding ``ctx``
        (injected by the registry at dispatch time, not LLM-visible).
        Description defaults to the function's docstring."""

        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            sig = inspect.signature(fn)
            try:
                hints = get_type_hints(fn)
            except Exception:  # pragma: no cover - defensive
                hints = {}
            props: dict[str, Any] = {}
            required: list[str] = []
            for pname, param in sig.parameters.items():
                if pname == "ctx":
                    continue
                t = hints.get(pname, str)
                props[pname] = _py_type_to_schema(t)
                if param.default is inspect.Parameter.empty:
                    required.append(pname)
            params: dict[str, Any] = {"type": "object", "properties": props}
            if required:
                params["required"] = required
            tname = name or fn.__name__
            self._tools[tname] = Tool(
                name=tname,
                description=description or (fn.__doc__ or "").strip(),
                parameters=params,
                fn=fn,
                timeout_s=timeout_s,
            )
            return fn

        return deco

    def list_openai_tools(
        self, allow: set[str] | None = None
    ) -> list[dict[str, Any]]:
        """Return tools[] in OpenAI's chat-completions format. ``allow``
        filters by name; ``None`` exposes everything registered."""
        out: list[dict[str, Any]] = []
        for tname, t in self._tools.items():
            if allow is not None and tname not in allow:
                continue
            out.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            })
        return out

    def has(self, name: str) -> bool:
        return name in self._tools

    async def dispatch(
        self,
        name: str,
        arguments: dict[str, Any] | None,
        ctx: Any,
    ) -> dict[str, Any]:
        """Invoke the named tool with ``arguments``.

        Always returns a JSON-serialisable dict. On error, returns
        ``{"success": False, "error": str}`` so the LLM can self-recover
        rather than crashing the voice loop."""
        t = self._tools.get(name)
        if t is None:
            return {"success": False, "error": f"unknown tool: {name}"}
        allowed = set(t.parameters.get("properties", {}).keys())
        clean: dict[str, Any] = {
            k: v for k, v in (arguments or {}).items() if k in allowed
        }
        try:
            if "ctx" in inspect.signature(t.fn).parameters:
                clean["ctx"] = ctx
            result = t.fn(**clean)
            if inspect.iscoroutine(result):
                result = await asyncio.wait_for(result, timeout=t.timeout_s)
            if isinstance(result, dict):
                return result
            return {"value": result}
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"tool {name} timed out after {t.timeout_s}s",
            }
        except Exception as e:  # noqa: BLE001
            logger.warning("tool %s raised %r", name, e)
            return {"success": False, "error": str(e)}


# Module-level default registry. Builtins register against this; the
# app wires it into ToolCallCtx. Tests can construct dedicated
# ToolRegistry() instances for isolation.
default_registry = ToolRegistry()
