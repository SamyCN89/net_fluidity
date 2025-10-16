#!/usr/bin/env python3
"""Compatibility shim for the centralized Ines preprocessing entrypoint."""

from scripts.preprocessing.ines import (
    GroupingPayload,
    PrepResult,
    build_parser,
    main,
    prepare_cognitive_dataset,
    write_outputs,
)

__all__ = [
    "GroupingPayload",
    "PrepResult",
    "prepare_cognitive_dataset",
    "write_outputs",
    "build_parser",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
