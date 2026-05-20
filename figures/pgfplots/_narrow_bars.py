"""Narrow box-and-whisker bars in pgfplots paired-delta plots.

Each bar in P017 / P022_P027 has:
  box edges  at center ± 0.068
  cap edges  at center ± 0.046
This script rewrites every (axis cs:X,Y) coordinate so that:
  box edges  -> center ± 0.050
  cap edges  -> center ± 0.034
Bar centres (and scatter-dot positions) are left untouched.
"""
import re, sys

# Fractional parts (4 dp) that identify a bar edge or cap edge.
# Bar centres have fracs in {0.76, 0.92, 0.08, 0.24}.
RIGHT_EDGE_FRACS = {0.828, 0.988, 0.148, 0.308}
LEFT_EDGE_FRACS  = {0.692, 0.852, 0.012, 0.172}
RIGHT_CAP_FRACS  = {0.806, 0.966, 0.126, 0.286}
LEFT_CAP_FRACS   = {0.714, 0.874, 0.034, 0.194}

NEW_HW_BOX = 0.050  # was 0.068
NEW_HW_CAP = 0.034  # was 0.046

TOL = 1e-4


def _match(frac, refs):
    return any(abs(frac - r) < TOL for r in refs)


def transform_x(x: float) -> float:
    intpart = int(x)
    frac = round(x - intpart, 4)
    if _match(frac, RIGHT_EDGE_FRACS):
        center_frac = frac - 0.068
        return intpart + center_frac + NEW_HW_BOX
    if _match(frac, LEFT_EDGE_FRACS):
        center_frac = frac + 0.068
        return intpart + center_frac - NEW_HW_BOX
    if _match(frac, RIGHT_CAP_FRACS):
        center_frac = frac - 0.046
        return intpart + center_frac + NEW_HW_CAP
    if _match(frac, LEFT_CAP_FRACS):
        center_frac = frac + 0.046
        return intpart + center_frac - NEW_HW_CAP
    return x


def process(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(r"\(axis cs:(-?\d+\.?\d*),(-?\d+\.?\d*)\)")
    n_changed = 0

    def repl(m):
        nonlocal n_changed
        try:
            x = float(m.group(1))
        except ValueError:
            return m.group(0)
        new_x = transform_x(x)
        if abs(new_x - x) > TOL:
            n_changed += 1
            return f"(axis cs:{new_x:.4f},{m.group(2)})"
        return m.group(0)

    new_content = pattern.sub(repl, content)
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    return n_changed


if __name__ == "__main__":
    for path in sys.argv[1:]:
        n = process(path)
        print(f"{path}: {n} coordinates rewritten")
