#!/usr/bin/env python3
"""Launch a composed system profile without changing the standalone GraspGen entry point."""
from pipeline.system.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
