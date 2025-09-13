"""Sphinx extension to generate example plots before building docs."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def generate_plots(app: Any, config: Any) -> None:
    """Generate example plots before building documentation."""
    # Check if plot generation is disabled
    build_plots = os.environ.get('BUILD_PLOTS', 'true').lower() == 'true'

    if not build_plots:
        print("📋 Plot generation disabled (BUILD_PLOTS=false). Using existing SVG files.")
        return

    docs_dir = Path(app.srcdir)
    script_path = docs_dir / "generate_example_plots.py"

    if script_path.exists():
        print("Generating example plots...")
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=docs_dir,
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
            if result.stderr:
                print("Warnings:", result.stderr)
            print("✅ Example plots generated successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to generate plots: {e}")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            # Don't fail the build, just warn
    else:
        print(f"⚠️  Plot generation script not found: {script_path}")


def setup(app: Any) -> dict[str, Any]:
    """Setup the Sphinx extension."""
    app.connect("config-inited", generate_plots)
    return {"version": "1.0", "parallel_read_safe": True}
