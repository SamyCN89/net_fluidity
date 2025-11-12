#!/usr/bin/env python3
"""
Heuristic report of potentially unused functions/modules and TODO/FIXME/XXX index.

Outputs:
- reports/dead_code.md
- reports/todo_index.md

Notes:
- Function usage is approximated via AST Name/Attribute references across files.
- Module usage is inferred from import statements; scripts under allegiance/ and julien_data/ are excluded.
"""
from __future__ import annotations

import ast
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

EXCLUDE_DIRS = {
    ".git",
    ".vscode",
    "allegiance/env",
    "build",
    "dist",
    "__pycache__",
    ".pytest_cache",
    "shared_code/shared_code.egg-info",
    "reports",
    "docs/patches",
}


def iter_py_files() -> list[Path]:
    files: list[Path] = []
    for p in ROOT.rglob("*.py"):
        rel = p.relative_to(ROOT)
        parts = rel.as_posix()
        if any(parts.startswith(d + "/") or parts == d for d in EXCLUDE_DIRS):
            continue
        files.append(p)
    return files


def module_name_from_path(p: Path) -> str:
    rel = p.relative_to(ROOT).with_suffix("")
    return rel.as_posix().replace("/", ".")


def collect_defs(files: list[Path]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    func_defs: dict[str, set[str]] = {}
    class_defs: dict[str, set[str]] = {}
    for p in files:
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        except Exception:
            continue
        fset: set[str] = set()
        cset: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                fset.add(node.name)
            elif isinstance(node, ast.ClassDef):
                cset.add(node.name)
        if fset:
            func_defs[str(p.relative_to(ROOT))] = fset
        if cset:
            class_defs[str(p.relative_to(ROOT))] = cset
    return func_defs, class_defs


def collect_imports(files: list[Path]) -> set[str]:
    imports: set[str] = set()
    for p in files:
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
    return imports


def collect_name_refs(files: list[Path]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in files:
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                counts[node.id] = counts.get(node.id, 0) + 1
            elif isinstance(node, ast.Attribute):
                counts[node.attr] = counts.get(node.attr, 0) + 1
    return counts


def compute_dead_code():
    files = iter_py_files()
    func_defs, class_defs = collect_defs(files)
    name_refs = collect_name_refs(files)
    imports = collect_imports(files)

    # Potentially unused functions: never referenced by name (heuristic)
    unused_funcs: list[tuple[str, str]] = []
    for path, names in func_defs.items():
        for name in sorted(names):
            # Skip dunder and main
            if name.startswith("__") or name == "main":
                continue
            if name_refs.get(name, 0) <= 1:  # 1 may account for the def itself
                unused_funcs.append((path, name))

    # Potentially unused modules: modules under metaconnectivity/ and shared_code/shared_code/ not imported
    unused_modules: list[str] = []
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        if rel.startswith("metaconnectivity/") or rel.startswith("shared_code/shared_code/"):
            mod = module_name_from_path(p)
            imported = any(
                mod == imp or imp.startswith(mod + ".") or mod.startswith(imp + ".")
                for imp in imports
            )
            if not imported:
                unused_modules.append(rel)

    return sorted(unused_funcs), sorted(set(unused_modules))


def report_dead_code() -> str:
    unused_funcs, unused_modules = compute_dead_code()

    lines: list[str] = []
    lines.append("# Potentially Unused Code (Heuristic)\n\n")
    lines.append("These results are static heuristics — please validate before removal or deprecation.\n\n")
    lines.append("## Functions with no detected references\n")
    if unused_funcs:
        for path, name in unused_funcs:
            lines.append(f"- {path} :: {name}\n")
    else:
        lines.append("- None found\n")

    lines.append("\n## Modules not imported elsewhere (libs only)\n")
    if unused_modules:
        for m in unused_modules:
            lines.append(f"- {m}\n")
    else:
        lines.append("- None found\n")

    return "".join(lines)


TAG_RX = re.compile(r"\b(TODO|FIXME|XXX|HACK|BUG)\b", re.IGNORECASE)


def report_todos() -> str:
    files: list[Path] = []
    for p in ROOT.rglob("*"):
        if p.is_dir():
            continue
        # simple filter: common text/code files
        if p.suffix.lower() in {".py", ".md", ".sh", ".yml", ".yaml", ".toml", ".ini", ".json"}:
            rel = p.relative_to(ROOT).as_posix()
            if any(rel.startswith(d + "/") or rel == d for d in EXCLUDE_DIRS):
                continue
            files.append(p)

    items: list[tuple[str, int, str]] = []
    for p in files:
        try:
            lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, 1):
            if TAG_RX.search(line):
                items.append((p.relative_to(ROOT).as_posix(), i, line.strip()))

    out: list[str] = []
    out.append("# TODO/FIXME/XXX Index\n\n")
    if not items:
        out.append("No TODO-like tags found.\n")
    else:
        for path, ln, text in sorted(items):
            out.append(f"- {path}:{ln}: {text}\n")
    return "".join(out)


def main() -> int:
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    # Write MD
    (reports_dir / "dead_code.md").write_text(report_dead_code(), encoding="utf-8")
    # Write CSV with function paths and modules
    unused_funcs, unused_modules = compute_dead_code()
    csv_lines = ["type,path,module,name\n"]
    for path, name in unused_funcs:
        modname = module_name_from_path((ROOT / path).resolve())
        csv_lines.append(f"function,{path},{modname},{name}\n")
    for mod in unused_modules:
        modname = module_name_from_path((ROOT / mod).resolve())
        csv_lines.append(f"module,{mod},{modname},\n")
    (reports_dir / "dead_code.csv").write_text("".join(csv_lines), encoding="utf-8")
    (reports_dir / "todo_index.md").write_text(report_todos(), encoding="utf-8")
    print("Wrote reports/dead_code.md, reports/dead_code.csv and reports/todo_index.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
