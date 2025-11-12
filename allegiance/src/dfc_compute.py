#!/usr/bin/env python3
"""Compatibility shim forwarding to the centralized DFC compute module."""

from scripts.dfc.dfc_compute import (
    DATASET_DEFAULTS,
    build_parser,
    expected_shape,
    load_timeseries,
    main,
)

__all__ = ["DATASET_DEFAULTS", "build_parser", "expected_shape", "load_timeseries", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
