"""The reference distribution contains components, not experiment recipes."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_reference_does_not_ship_model_recipes_or_campaigns():
    excluded = (
        "docs", "experiments", "src/tools", "src/edgeflow_split/models",
        "src/edgeflow_split/role_pipeline.py",
        "tools/mac_role_arrangements.py",
        "tools/benchmark_mac_single_request.py",
        "QCS6490/edgeflow_qcs6490/scripts",
        "RK3588/edgeflow_rk3588/scripts",
        "RK3588/edgeflow_rk3588/runtime/rknn_vision.py",
    )
    assert not [name for name in excluded if (ROOT / name).exists()]


def test_readme_describes_a_reference_subset():
    readme = (ROOT / "README.md").read_text()
    assert "reference implementation" in readme
    assert "not a complete reproduction package" in readme
    assert "https://arxiv.org/abs/2610.35309" in readme
