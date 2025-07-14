#!/usr/bin/env python3
"""Run the ATLAS dashboard."""

import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "src/atlas_atc/frontend/dashboard.py",
        "--server.port", "8501",
        "--server.address", "localhost"
    ])