#!/usr/bin/env python3
import re
import sys
from pathlib import Path


PATTERNS = [
    # Direct function imports
    re.compile(r"^\s*from\s+metaconnectivity\.fun_dfcspeed\s+import\s+.*\bts2dfc_stream\b"),
    re.compile(r"^\s*from\s+metaconnectivity\.fun_metaconnectivity\s+import\s+.*\bcompute_metaconnectivity\b"),
    # Module imports (discouraged for stable funcs)
    re.compile(r"^\s*import\s+metaconnectivity\.fun_dfcspeed\b"),
    re.compile(r"^\s*import\s+metaconnectivity\.fun_metaconnectivity\b"),
]


SKIP_PATHS = {
    # Allow explicit comparison tool to import legacy modules
    Path("tools/compare_shared_vs_meta.py").as_posix(),
}


def main(argv):
    bad = []
    for arg in argv[1:]:
        p = Path(arg)
        if not p.exists():
            continue
        # Skip files under the legacy module tree
        if p.as_posix().startswith("metaconnectivity/"):
            continue
        if p.as_posix() in SKIP_PATHS:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for i, line in enumerate(text, 1):
            for rx in PATTERNS:
                if rx.search(line):
                    bad.append((p.as_posix(), i, line.strip()))
    if bad:
        print("Found forbidden imports of stable funcs from metaconnectivity.\n")
        print("Please import from shared_code.shared_code instead:")
        print(" - shared_code.shared_code.fun_dfcspeed.ts2dfc_stream")
        print(" - shared_code.shared_code.fun_metaconnectivity.compute_metaconnectivity\n")
        for path, ln, src in bad:
            print(f"{path}:{ln}: {src}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

