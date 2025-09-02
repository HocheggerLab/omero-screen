"""Sphinx extension to generate example plots before building docs."""

import subprocess
import sys
from pathlib import Path
from typing import Any


def generate_plots(app: Any, config: Any) -> None:
    """Generate example plots before building documentation."""
    docs_dir = Path(app.srcdir)
    script_path = docs_dir / "generate_example_plots.py"

    if script_path.exists():
        print("Generating example plots...")
        try:
            subprocess.run(
                [sys.executable, str(script_path)], cwd=docs_dir, check=True
            )
            print("✅ Example plots generated successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to generate plots: {e}")
            # Don't fail the build, just warn
    else:
        print(f"⚠️  Plot generation script not found: {script_path}")


def setup(app: Any) -> dict[str, Any]:
    """Setup the Sphinx extension."""
    app.connect("config-inited", generate_plots)
    return {"version": "1.0", "parallel_read_safe": True}
