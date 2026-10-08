"""The numbers in the README must come from the saved result files, not from memory."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = json.loads((ROOT / "ml_pipeline" / "final_results.json").read_text())
IMPACT = json.loads((ROOT / "ml_pipeline" / "business_impact.json").read_text())
README = (ROOT / "README.md").read_text()


def test_selected_model_is_the_one_in_the_readme():
    rf = FINAL["results"]["random forest"]
    assert FINAL["selected"].startswith("random forest")
    assert f"{rf['pr_auc']:.3f}" in README
    assert f"{rf['caught']} of {rf['caught'] + rf['missed']}" in README


def test_dollar_figures_match():
    assert f"{IMPACT['model']['share_of_loss_flagged'] * 100:.1f}%" in README
    assert f"${IMPACT['total_loss_dollars'] / 1000:.1f}K" in README
