#!/usr/bin/env python3
"""Compatibility wrapper for the legacy preprocessing script.

Delegates to the consolidated implementation in
`metaconnectivity/cognitive_data_ts_sorted.py`.
"""

from metaconnectivity.cognitive_data_ts_sorted import main


if __name__ == "__main__":
    main()
