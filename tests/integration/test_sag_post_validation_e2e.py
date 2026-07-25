import copy
import json
from collections.abc import Mapping

import numpy as np
import pandas as pd

from modules.sag_validation import publish_sag_snapshot, run_sag_validation_upload
from tests.fixtures.sag_data import drake_conductor, make_line_data
from utils.audit_log import JsonAuditLogger


class _InclinationUpload:
    name = "inclination.csv"

    @staticmethod
    def getvalue():
        return "倾角\n1.0\n1.1\n".encode()


def _assert_deep_equal(actual, expected):
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
        return
    if isinstance(expected, (pd.Index, pd.Series, pd.DataFrame)):
        if isinstance(expected, pd.DataFrame):
            pd.testing.assert_frame_equal(actual, expected)
        elif isinstance(expected, pd.Series):
            pd.testing.assert_series_equal(actual, expected)
        else:
            pd.testing.assert_index_equal(actual, expected)
        return
    if isinstance(expected, Mapping):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_deep_equal(actual[key], expected[key])
        return
    if isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_deep_equal(actual_item, expected_item)
        return
    assert actual == expected


def test_sag_end_to_end_isolated_persisted_and_audited(tmp_path):
    line_data = make_line_data()
    before = copy.deepcopy(line_data)
    snapshot = publish_sag_snapshot(
        {},
        line_data,
        drake_conductor(),
        tower_coords=line_data["tower_coords"],
        source_run_id="dlr-e2e-run",
        line_id="line-a",
    )
    logger = JsonAuditLogger(tmp_path / "logs")

    result = run_sag_validation_upload(
        snapshot,
        _InclinationUpload(),
        selected_tower_id="001",
        output_dir=tmp_path / "results",
        audit_logger=logger,
    )

    assert result.rows
    assert all(row.final_current_a <= row.original_current_a for row in result.rows)
    assert result.result_path is not None and result.result_path.exists()
    assert result.audit_events
    assert result.audit_persisted is True
    assert json.loads(result.result_path.read_text())["result_id"] == result.result_id
    audit_events = [
        json.loads(line)
        for line in logger.log_path.read_text().splitlines()
        if line.strip()
    ]
    assert any(event["result_id"] == result.result_id for event in audit_events)
    _assert_deep_equal(line_data, before)
