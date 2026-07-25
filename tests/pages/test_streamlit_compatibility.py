import ast
from pathlib import Path


def test_streamlit_calls_do_not_use_removed_container_width_argument():
    project_root = Path(__file__).resolve().parents[2]
    sources = [project_root / "dispatch_app_st.py"]
    sources.extend(sorted((project_root / "pages").glob("*.py")))
    offenders = []

    for source in sources:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if any(
                keyword.arg == "use_container_width" for keyword in node.keywords
            ):
                offenders.append(f"{source.name}:{node.lineno}")

    assert not offenders, f"已移除参数 use_container_width: {offenders}"
