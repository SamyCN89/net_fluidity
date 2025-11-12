#!/usr/bin/env python3
"""Compatibility shim for the central preprocessing CLI."""

from scripts.preprocessing.julien import main


if __name__ == "__main__":
    raise SystemExit(main())
