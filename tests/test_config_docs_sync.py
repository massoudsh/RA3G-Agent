"""
Guards against config.yml and README.md drifting apart on governance thresholds.

The README documents THRESHOLDS as the expected/production values for the
shipped config.yml example block. If someone changes one without the other,
this test fails loudly instead of letting the mismatch reach an operator
silently (see portfolio audit finding: RA3G-Agent governance threshold
mismatch).
"""

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_config_thresholds() -> dict:
    with open(REPO_ROOT / "config.yml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["THRESHOLDS"]


def _load_readme_documented_thresholds() -> dict:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    # Look inside the "## Configuration" section's yaml code block for THRESHOLDS.
    match = re.search(r"THRESHOLDS:\s*\n\s*retriever:\s*([\d.]+)\s*\n\s*reasoner:\s*([\d.]+)", readme)
    assert match, "Could not find documented THRESHOLDS block in README.md"
    return {"retriever": float(match.group(1)), "reasoner": float(match.group(2))}


def test_config_thresholds_match_readme():
    config_thresholds = _load_config_thresholds()
    readme_thresholds = _load_readme_documented_thresholds()

    assert config_thresholds["retriever"] == readme_thresholds["retriever"], (
        f"config.yml retriever threshold ({config_thresholds['retriever']}) does not match "
        f"README.md documented value ({readme_thresholds['retriever']}) — "
        "update one or the other so operators can trust the docs."
    )
    assert config_thresholds["reasoner"] == readme_thresholds["reasoner"], (
        f"config.yml reasoner threshold ({config_thresholds['reasoner']}) does not match "
        f"README.md documented value ({readme_thresholds['reasoner']}) — "
        "update one or the other so operators can trust the docs."
    )
