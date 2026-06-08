"""Tests for the CLI pipeline (end-to-end, offline)."""

from __future__ import annotations

import json
from pathlib import Path

from refusalclimb.cli import PipelineReport, main, run_pipeline


def test_run_pipeline_produces_expected_shape() -> None:
    """The full pipeline runs offline and reports sane, consistent numbers."""
    report = run_pipeline(seeds=5, budget=200, fanout=4)
    assert isinstance(report, PipelineReport)
    # Sample-efficiency: guided beats binary.
    assert report.median_guided < report.median_binary
    assert report.speedup > 1.0
    # Discovery surfaced multiple classes.
    assert report.n_successful_prompts > 0
    assert report.n_classes >= 2
    assert len(report.classes) == report.n_classes
    # Transfer + defense rates are valid fractions.
    assert 0.0 <= report.transfer_rate <= 1.0
    assert 0.0 <= report.blocked_rate <= 1.0


def test_cli_main_writes_artifacts(tmp_path: Path) -> None:
    """`refusalclimb run --out <dir>` writes JSON + Markdown and exits 0."""
    out = tmp_path / "results"
    rc = main(["run", "--seeds", "3", "--out", str(out)])
    assert rc == 0
    json_path = out / "last_run.json"
    md_path = out / "last_run.md"
    assert json_path.exists() and md_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["forbidden_token"] == "BANANA"
    assert data["median_guided"] < data["median_binary"]
