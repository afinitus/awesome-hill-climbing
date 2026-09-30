"""Turn a chapter's ``@dataclass Config`` into a command line, with no argparse boilerplate.

    @dataclass
    class Config:
        seed: int = 0
        lr: float = field(default=3e-4, metadata={"help": "Adam step size"})
        eval_sets: list[str] = field(default_factory=lambda: ["eval"])
        render: bool = False
        quick: bool = False
        QUICK: ClassVar[dict] = {"iterations": 3}   # applied by --quick

    cfg = parse(Config)   # python algos/01_ars.py --lr 1e-3 --eval-sets eval stress --no-render

Rules: every field becomes ``--field-name`` (``--field_name`` also works); bools become
``--flag`` / ``--no-flag``; lists and tuples take several values; ``Literal`` fields get
choices. If the class defines ``QUICK`` and ``--quick`` is passed, those overrides apply to
every field the user did not set explicitly.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import types
from collections.abc import Sequence
from typing import Any, Literal, TypeVar, Union, get_args, get_origin, get_type_hints

T = TypeVar("T")


def _argparse_kwargs(tp: Any) -> dict[str, Any]:
    """argparse options for one field type."""
    origin, args = get_origin(tp), get_args(tp)
    if origin in (Union, types.UnionType):  # Optional[X] -> X
        inner = [a for a in args if a is not type(None)]
        if len(inner) == 1:
            return _argparse_kwargs(inner[0])
        raise TypeError(f"unsupported union type {tp}")
    if tp is bool:
        return {"action": argparse.BooleanOptionalAction}
    if origin is Literal:
        return {"choices": list(args), "type": type(args[0])}
    if origin in (list, tuple, Sequence) or tp in (list, tuple):
        item = args[0] if args else str
        return {"nargs": "*", "type": item}
    return {"type": tp}


def _default(f: dataclasses.Field) -> Any:
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return dataclasses.MISSING


def parse(config_cls: type[T], args: Sequence[str] | None = None, description: str | None = None) -> T:
    """Parse ``args`` (default: ``sys.argv[1:]``) into an instance of ``config_cls``.

    The help text defaults to the calling script's module docstring.
    """
    if description is None:
        description = getattr(sys.modules.get("__main__"), "__doc__", None)
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    hints = get_type_hints(config_cls)
    defaults: dict[str, Any] = {}
    for f in dataclasses.fields(config_cls):
        if not f.init:
            continue
        default = _default(f)
        if default is not dataclasses.MISSING:
            defaults[f.name] = default
        flags = [f"--{f.name.replace('_', '-')}"] + ([f"--{f.name}"] if "_" in f.name else [])
        help_text = f.metadata.get("help", "")
        help_text += " (required)" if default is dataclasses.MISSING else f" (default: {default})"
        parser.add_argument(
            *flags,
            dest=f.name,
            default=argparse.SUPPRESS,  # so the namespace holds only what the user typed
            required=default is dataclasses.MISSING,
            help=help_text.strip().replace("%", "%%"),
            **_argparse_kwargs(hints[f.name]),
        )
    explicit = vars(parser.parse_args(args))

    values = dict(defaults)
    if explicit.get("quick", defaults.get("quick", False)):
        values.update(getattr(config_cls, "QUICK", {}))
    values.update(explicit)
    for name, value in values.items():  # lists typed as tuples come back as lists
        if isinstance(value, list) and get_origin(hints[name]) is tuple:
            values[name] = tuple(value)
    return config_cls(**values)
