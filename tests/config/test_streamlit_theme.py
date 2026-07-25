import tomllib
from pathlib import Path


def test_sidebar_theme_has_explicit_nonempty_core_colors():
    config_path = Path(__file__).resolve().parents[2] / ".streamlit" / "config.toml"
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)

    sidebar = config["theme"]["sidebar"]
    assert sidebar["primaryColor"]
    assert sidebar["textColor"]


def test_requirements_declare_streamlit_version_for_current_page_api():
    requirements_path = Path(__file__).resolve().parents[2] / "requirements.txt"
    requirements = {
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "streamlit>=1.60,<2" in requirements
