"""Offline release checks: source closure, portable paths and public docs."""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_reference_tree_excludes_raw_experiments():
    assert not (ROOT / "experiments").exists()


def test_reference_tree_ignores_private_run_artifacts():
    ignored = set((ROOT / ".gitignore").read_text().splitlines())
    assert {"/experiments/", "/artifacts/"} <= ignored


def test_release_markdown_is_readme_only():
    allowed = {"README.md"}
    actual = {str(path.relative_to(ROOT)) for path in ROOT.rglob("*.md")
              if not any(part.startswith(".") for part in path.relative_to(ROOT).parts)}
    assert actual == allowed


@pytest.mark.parametrize("relative", [
    "tools/edgeflow_llm_runtime/CMakeLists.txt",
    "tools/edgeflow_llm_runtime/llama_src/LICENSE",
    "tools/edgeflow_llm_runtime/llama_src/ggml/LICENSE",
    "tools/edgeflow_llm_runtime/src/edgeflow_llm_ane_glue.mm",
    "tools/edgeflow_llm_runtime/src/qwen25_prefill.h",
    "crane/src/ane_bridge.m", "crane/src/ane_bridge.h",
    "tools/mac_fastvlm_coreml_native.mm",
    "src/edgeflow_split/ane_surface_pool.py",
    "src/edgeflow_split/coreml_surface.py",
    "src/edgeflow_split/overlap_pipeline/surface_session.py",
    "src/edgeflow_split/crossinfer_session.py",
    "src/edgeflow_split/crossinfer_context.py",
    "src/edgeflow_split/shared_tensor.py",
    "requirements-mac.txt",
])
def test_mac_core_build_sources_are_present(relative):
    assert (ROOT / relative).is_file(), relative


def test_mac_runtime_imports_resolve_inside_release():
    for path in (ROOT / "src/edgeflow_split").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            elif isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            else:
                continue
            for name in modules:
                if not name.startswith("edgeflow_split."):
                    continue
                target = ROOT / "src" / name.replace(".", "/")
                assert target.with_suffix(".py").is_file() or (target / "__init__.py").is_file(), (path, name)


def test_native_adapters_do_not_depend_on_private_machine_paths():
    paths = [ROOT / "tools" / name for name in (
        "mac_fastvlm_coreml_native.py", "mac_fastvlm_coreml_native.mm",
        "mac_fastvlm_ggml.py", "mac_fastvlm_ggml.cpp",
    )]
    for path in paths:
        text = path.read_text()
        assert not re.search(r"/(?:Users|home)/[\w.-]+/|/Volumes/[\w.-]+/", text), path


def test_readme_relative_links_exist():
    readme = (ROOT / "README.md").read_text()
    links = re.findall(r"\]\(([^)]+)\)|src=\"([^\"]+)\"", readme)
    for markdown, html in links:
        target = markdown or html
        if target.startswith(("https://", "http://", "#")):
            continue
        assert (ROOT / target.split("#", 1)[0]).is_file(), target
