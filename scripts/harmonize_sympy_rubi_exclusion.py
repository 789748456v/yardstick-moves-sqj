#!/usr/bin/env python3
"""
Uniformly exclude SymPy's machine-generated Mathematica translation dump (`sympy/integrals/rubi/**`,
introduced in 1.2 and upstream-removed in 1.12) across linter_runs, full_newcode_lines, and sq_runs
so all 15 analyzer/taxonomy regimes evaluate the exact same human-authored Python file scope on SymPy.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

BASE_DIR = Path(os.environ.get("D04_BASE_DIR", Path(__file__).resolve().parent.parent))
RUBI_PREFIX = "sympy/integrals/rubi/"


def main() -> None:
    # 1. Clean linter_runs for sympy__sympy
    l_dir = BASE_DIR / "data" / "linter_runs"
    cleaned_l = 0
    for jf in sorted(l_dir.glob("sympy__sympy__*.json")):
        d = json.loads(jf.read_text(encoding="utf-8"))
        changed = False
        for tname, issues in d.get("tools", {}).items():
            filtered = [x for x in issues if not str(x.get("file", "")).startswith(RUBI_PREFIX)]
            if len(filtered) != len(issues):
                d["tools"][tname] = filtered
                changed = True
        if changed:
            jf.write_text(json.dumps(d), encoding="utf-8")
            cleaned_l += 1
    print(f"Harmonized {cleaned_l} SymPy linter_runs JSONs (removed {RUBI_PREFIX}*).")

    # 2. Clean full_newcode_lines.json for sympy__sympy
    nc_path = BASE_DIR / "data" / "full_newcode_lines.json"
    if nc_path.exists():
        nc = json.loads(nc_path.read_text(encoding="utf-8"))
        nc_changed = 0
        for sid, fmap in nc.items():
            if sid.startswith(("sympy/sympy::", "sympy__sympy::")):
                drop_keys = [fp for fp in fmap if fp.startswith(RUBI_PREFIX)]
                if drop_keys:
                    for k in drop_keys:
                        del fmap[k]
                    nc_changed += 1
        if nc_changed:
            nc_path.write_text(json.dumps(nc), encoding="utf-8")
        print(f"Harmonized {nc_changed} SymPy entries in full_newcode_lines.json.")

    # 3. Clear sympy__sympy SonarQube JSONs ONLY if --clear-sq is explicitly requested
    import sys
    if "--clear-sq" in sys.argv:
        sq_root = BASE_DIR / "data" / "sq_runs"
        removed_sq = 0
        for reg in ["sq99_std", "sq103_n09", "sq26_mqr", "sq26_comp_mqr"]:
            rdir = sq_root / reg
            if rdir.exists():
                for jf in rdir.glob("sympy__sympy__*.json"):
                    jf.unlink()
                    removed_sq += 1
        print(f"Cleared {removed_sq} old SymPy SonarQube JSONs for clean 4-regime re-scan.")


if __name__ == "__main__":
    main()
