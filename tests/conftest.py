"""Pure-torch tests (no Isaac Lab). Run from the repo root: ``python -m pytest tests``.

Requires torch, tensordict and the bundled rsl_rl (``pip install -e rsl_rl``).
"""

import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (os.path.join(REPO, "ame2"), os.path.join(REPO, "rsl_rl"), os.path.join(REPO, "ame2", "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
