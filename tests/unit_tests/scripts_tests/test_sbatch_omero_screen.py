"""``bin/sbatch_omero_screen.py`` writes a job script that works from anywhere."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SBATCH = REPO_ROOT / "bin" / "sbatch_omero_screen.py"


def _job_script(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SBATCH), "123", "--no-submit", *args],
        cwd=tmp_path,  # not bin/, and no .env anywhere near
        capture_output=True,
        text=True,
        check=False,
    )


def test_job_script_runs_the_programs_with_this_python(tmp_path: Path) -> None:
    result = _job_script(tmp_path, "--stitch")
    assert result.returncode == 0, result.stderr
    (script,) = tmp_path.glob("os123.*.sh")
    text = script.read_text()
    assert "uv run" not in text
    assert f"{sys.executable} {REPO_ROOT / 'bin' / 'run_omero_screen.py'} 123" in text
    assert f"export PYTHONPATH={REPO_ROOT}" in text
    assert "--stitch" in text
    assert "#SBATCH --gres=gpu" in text


def test_a_missing_env_file_is_only_an_error_when_asked_for(tmp_path: Path) -> None:
    result = _job_script(tmp_path, "--env", "no-such-env")
    assert result.returncode != 0
    assert ".env.no-such-env" in result.stderr
