import json
import subprocess
import sys
from pathlib import Path


def run_generate(output_dir: Path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "alpha_miner.main",
            "--mode",
            "generate",
            "--objective",
            "US equity momentum alphas",
            "--batch-size",
            "3",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads((output_dir / "summary.json").read_text())


def test_generate_is_schema_reproducible(tmp_path):
    first = run_generate(tmp_path / "first")
    second = run_generate(tmp_path / "second")
    assert first["engine"] == second["engine"] == "python-v2"
    assert first["generatedCandidates"] == second["generatedCandidates"] == 3
    assert first["rejected_by_stage"].keys() == second["rejected_by_stage"].keys()
