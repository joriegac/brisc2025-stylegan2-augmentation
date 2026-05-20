"""
Compatibility entrypoint for the v2 Random Forest pipeline.

The maintained implementation lives in classify_rf_v2.py. This wrapper keeps
older commands that invoke classify_v2.py working without duplicating pipeline
logic.
"""

from __future__ import annotations

from classify_rf_v2 import main


if __name__ == "__main__":
    main()
