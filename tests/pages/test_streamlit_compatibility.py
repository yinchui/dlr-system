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


def test_main_page_does_not_expose_static_rating_baseline():
    project_root = Path(__file__).resolve().parents[2]
    source = (project_root / "dispatch_app_st.py").read_text(
        encoding="utf-8"
    )

    for token in (
        "static_p",
        "static_val",
        "静态额定值",
        "对比静态",
        "增容空间",
    ):
        assert token not in source

    for label in (
        "最低载流量（系统瓶颈）",
        "最高载流量",
        "平均载流量",
        "动态增容（SRTM地形修正）",
    ):
        assert label in source


def test_legacy_line_data_uses_shared_dlr_publication_policy():
    project_root = Path(__file__).resolve().parents[2]
    source = (project_root / "dispatch_app_st.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source, filename="dispatch_app_st.py")
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "calculate_legacy_line_data"
    )
    assignments = {}
    for node in function.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "line_data"
                and isinstance(target.slice, ast.Constant)
            ):
                assignments[target.slice.value] = node.value

    current_assignment = assignments.get("max_currents")
    assert isinstance(current_assignment, ast.Call)
    assert isinstance(current_assignment.func, ast.Name)
    assert current_assignment.func.id == "publish_dlr_currents"
    safety_assignment = assignments.get("safety_factor")
    assert isinstance(safety_assignment, ast.Name)
    assert safety_assignment.id == "DLR_SAFETY_FACTOR"
