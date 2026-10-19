#!/usr/bin/env python3
"""CLI entry point to execute the STAGE One User, One Agent sequential replay and simulator."""

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from stage_agent.replay import main

if __name__ == "__main__":
    main()
