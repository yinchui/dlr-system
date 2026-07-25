import re
from pathlib import Path

from streamlit.testing.v1 import AppTest

import config.config as app_config
from modules.sag_validation import publish_sag_snapshot
from tests.fixtures.sag_data import drake_conductor, make_line_data


PAGE_PATH = Path(__file__).resolve().parents[2] / "pages" / "弧垂后验证.py"


def test_page_smoke_runs_without_main_snapshot():
    app = AppTest.from_file(str(PAGE_PATH))

    app.run(timeout=20)

    assert not app.exception
    assert len(app.file_uploader) == 1


def test_page_uses_only_sag_prefixed_widget_keys():
    source = PAGE_PATH.read_text(encoding="utf-8")
    keys = re.findall(r"key=[\"']([^\"']+)[\"']", source)

    assert keys
    assert all(key.startswith("sag_") for key in keys)


def test_page_keeps_backend_only_fields_out_of_visible_table():
    source = PAGE_PATH.read_text(encoding="utf-8")

    assert "build_visible_sag_result" in source
    assert "parameter_sources" not in source
    assert "st.dataframe" in source
    assert "st.download_button" in source


def _app_with_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(app_config, "SAG_RESULT_DIR", tmp_path / "results")
    monkeypatch.setattr(app_config, "AUDIT_LOG_DIR", tmp_path / "logs")
    app = AppTest.from_file(str(PAGE_PATH))
    app.session_state["sag_validation_snapshot"] = publish_sag_snapshot(
        {}, make_line_data(), drake_conductor(), source_run_id="dlr-run-page"
    )
    app.run(timeout=20)
    app.file_uploader[0].set_value(
        ("angles.csv", "倾角\n1.0\n1.1\n".encode(), "text/csv")
    )
    app.run(timeout=20)
    return app


def test_page_keeps_result_and_download_after_widget_rerun(tmp_path, monkeypatch):
    app = _app_with_snapshot(tmp_path, monkeypatch)

    app.button[0].click().run(timeout=20)

    assert not app.exception
    assert len(app.dataframe) == 1
    assert len(app.download_button) == 1
    app.run(timeout=20)
    assert len(app.dataframe) == 1
    assert len(app.download_button) == 1
    assert app.session_state["sag_result_id"]
    assert app.session_state["sag_result_persisted"] is True


def test_page_warns_when_backend_result_cannot_be_saved(tmp_path, monkeypatch):
    blocked_output = tmp_path / "blocked-output"
    blocked_output.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(app_config, "SAG_RESULT_DIR", blocked_output)
    monkeypatch.setattr(app_config, "AUDIT_LOG_DIR", tmp_path / "logs")
    app = AppTest.from_file(str(PAGE_PATH))
    app.session_state["sag_validation_snapshot"] = publish_sag_snapshot(
        {}, make_line_data(), drake_conductor(), source_run_id="dlr-run-page"
    )
    app.run(timeout=20)
    app.file_uploader[0].set_value(
        ("angles.csv", "倾角\n1.0\n1.1\n".encode(), "text/csv")
    )
    app.run(timeout=20)

    app.button[0].click().run(timeout=20)

    assert not app.exception
    assert len(app.dataframe) == 1
    assert len(app.warning) == 1
    assert "未保存" in app.warning[0].value
    assert app.session_state["sag_result_persisted"] is False


def test_page_clears_cached_result_when_tower_changes(tmp_path, monkeypatch):
    app = _app_with_snapshot(tmp_path, monkeypatch)
    app.button[0].click().run(timeout=20)
    assert len(app.dataframe) == 1

    app.selectbox[0].set_value("002").run(timeout=20)

    assert len(app.dataframe) == 0
    assert len(app.download_button) == 0


def test_page_clears_cached_result_when_upload_changes(tmp_path, monkeypatch):
    app = _app_with_snapshot(tmp_path, monkeypatch)
    app.button[0].click().run(timeout=20)
    assert len(app.dataframe) == 1

    app.file_uploader[0].set_value(
        ("other.csv", "倾角\n2.0\n2.1\n".encode(), "text/csv")
    ).run(timeout=20)

    assert len(app.dataframe) == 0
    assert len(app.download_button) == 0


def test_page_clears_cached_result_when_snapshot_changes(tmp_path, monkeypatch):
    app = _app_with_snapshot(tmp_path, monkeypatch)
    app.button[0].click().run(timeout=20)
    assert len(app.dataframe) == 1

    app.session_state["sag_validation_snapshot"] = publish_sag_snapshot(
        {}, make_line_data(), drake_conductor(), source_run_id="dlr-run-new"
    )
    app.run(timeout=20)

    assert len(app.dataframe) == 0
    assert len(app.download_button) == 0
