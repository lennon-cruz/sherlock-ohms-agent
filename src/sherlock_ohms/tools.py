"""Turn a plain Python function into an OpenAI function-calling tool schema."""

import inspect
from typing import Any, Callable, Literal, Optional, get_args, get_origin, get_type_hints


class Tool:
    def __init__(self, func: Callable, name: Optional[str] = None, description: Optional[str] = None):
        self.func = func
        self.name = name or func.__name__
        self.description = description or inspect.getdoc(func) or ""
        self.signature = inspect.signature(func)
        self.type_hints = get_type_hints(func)
        self.parameters = [
            self._param_schema(param_name, param)
            for param_name, param in self.signature.parameters.items()
        ]

    def _param_schema(self, name: str, param: inspect.Parameter) -> dict:
        return {
            "name": name,
            "schema": self._json_type(self.type_hints.get(name, str)),
            "required": param.default is inspect.Parameter.empty,
        }

    def _json_type(self, typ: Any) -> dict:
        origin = get_origin(typ)
        if origin is Literal:
            return {"type": "string", "enum": list(get_args(typ))}
        mapping = {str: "string", int: "integer", float: "number", bool: "boolean"}
        return {"type": mapping.get(typ, "string")}

    def dict(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {p["name"]: p["schema"] for p in self.parameters},
                    "required": [p["name"] for p in self.parameters if p["required"]],
                },
            },
        }

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __repr__(self):
        return f"<Tool name={self.name}>"


def tool(func=None, *, name: str = None, description: str = None):
    def wrapper(f):
        return Tool(f, name=name, description=description)

    return wrapper(func) if func else wrapper
