#!/usr/bin/env python3
"""
Analysis Runner
Convenience script to run the NVSim cluster analysis from the project root.
"""

import sys
from pathlib import Path

# Add src/py to path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent / "src" / "py"))

# Import and run the analysis
from analyze_nvsim_clusters import main

if __name__ == "__main__":
    main()