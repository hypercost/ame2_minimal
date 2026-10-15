"""Layer 4: merge closed_loop_abc.py result JSONs into a markdown table (no Isaac).

    python ame2/scripts/mapping/eval/compare_abc.py logs/mapping_eval/abc/teacher_*.json
"""

from __future__ import annotations

import json
import os
import sys

import _common  # noqa: F401  (sys.path)

from ame2.mapping.eval_metrics import compare_modes_markdown


def compare(paths):
    res = {}
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        res[d["map_mode"]] = d["outcomes"]
    table = compare_modes_markdown(res)
    out = os.path.join(os.path.dirname(os.path.abspath(paths[0])), "abc_table.md")
    with open(out, "w") as f:
        f.write(table + "\n")
    print(table)
    print(f"[layer4] wrote {out}")
    return table


if __name__ == "__main__":
    compare(sys.argv[1:])
