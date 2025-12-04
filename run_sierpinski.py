#!/usr/bin/env python3
"""
Sierpinski Test Runner
Convenience script to run the sierpinski test from the project root.
"""

import sys
import os
from pathlib import Path

# Add src/py to path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent / "src" / "py"))

# Import and run the main function
from sierpinski_test import main

if __name__ == "__main__":
    sys.exit(main())