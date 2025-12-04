#!/usr/bin/env python3
"""
Sierpinski Visualization Runner
Convenience script to run the sierpinski visualization from the project root.
"""

import sys
from pathlib import Path

# Add src/py to path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent / "src" / "py"))

# Import and run the visualization
from visualize_sierpinski import main

if __name__ == "__main__":
    main()