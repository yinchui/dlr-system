import hashlib
import importlib
from io import BytesIO

import pandas as pd
import pytest

from tests.fixtures.sample_data import make_tower_time_weather_dataframe


class FakeUploadedFile:
    def __init__(self, name, content):
        self.name = name
        self._content = content
        self.getvalue_calls = 0

    def getvalue(self):
        self.getvalue_calls += 1
        return self._content

    def read(self):
        raise AssertionError("上传文件不得通过 read() 读取")


def weather_upload_module():
    return importlib.import_module("modules.weather_upload")


def to_csv_upload(name, frame, encoding="utf-8-sig"):
    return FakeUploadedFile(name, frame.to_csv(index=False).encode(encoding))


def test_freeze_uploaded_file_reads_bytes_once():
    weather_upload = weather_upload_module()
    uploaded = FakeUploadedFile("weather.csv", b"abc")

    frozen = weather_upload.freeze_uploaded_file(uploaded)

    assert frozen.name == "weather.csv"
    assert frozen.content == b"abc"
    assert uploaded.getvalue_calls == 1


def test_freeze_uploaded_file_calculates_sha256():
    weather_upload = weather_upload_module()
    content = b"weather-data"

    frozen = weather_upload.freeze_uploaded_file(
        FakeUploadedFile("weather.csv", content)
    )

    assert frozen.sha256 == hashlib.sha256(content).hexdigest()


def test_freeze_uploaded_file_snapshots_mutable_content():
    weather_upload = weather_upload_module()
    mutable_content = bytearray(b"abc")
    uploaded = FakeUploadedFile("weather.csv", mutable_content)

    frozen = weather_upload.freeze_uploaded_file(uploaded)
    mutable_content[0] = ord("z")

    assert frozen.content == b"abc"
    assert isinstance(frozen.content, bytes)
    assert frozen.sha256 == hashlib.sha256(b"abc").hexdigest()
    assert uploaded.getvalue_calls == 1


def test_freeze_uploaded_file_rejects_non_bytes_convertible_content():
    weather_upload = weather_upload_module()
    uploaded = FakeUploadedFile("weather.csv", object())

    with pytest.raises(TypeError, match="bytes"):
        weather_upload.freeze_uploaded_file(uploaded)

    assert uploaded.getvalue_calls == 1


def test_freeze_uploaded_files_returns_empty_tuple_for_empty_input():
    weather_upload = weather_upload_module()

    assert weather_upload.freeze_uploaded_files([]) == ()


def test_mixed_files_are_normalized_before_concat():
    weather_upload = weather_upload_module()
    legacy = pd.DataFrame(
        {
            "位置": ["001号"],
            "日期": ["2026-07-23"],
            "时刻": ["00:00"],
            "环境温度": [20.0],
            "风速": [2.0],
            "风向": [90.0],
        }
    )
    tower_time = make_tower_time_weather_dataframe().iloc[[0]]
    uploads = [
        to_csv_upload("legacy.csv", legacy),
        to_csv_upload("tower.csv", tower_time),
    ]
    expected_hashes = {
        hashlib.sha256(upload.getvalue()).hexdigest() for upload in uploads
    }
    for upload in uploads:
        upload.getvalue_calls = 0

    result = weather_upload.normalize_uploaded_weather_files(
        uploads,
        role="physical",
    )

    assert len(result.frame) == 2
    assert result.frame["tower_id"].tolist() == ["001", "014"]
    assert set(result.frame["source_file_hash"]) == expected_hashes
    assert result.report.input_rows == 2
    assert result.report.valid_rows == 2
    assert result.report.dropped_rows == 0
    assert [item.name for item in result.files] == ["legacy.csv", "tower.csv"]
    assert all(not hasattr(item, "content") for item in result.files)
    assert [upload.getvalue_calls for upload in uploads] == [1, 1]


def test_normalize_uploaded_weather_files_supports_gbk_csv():
    weather_upload = weather_upload_module()
    raw = pd.DataFrame(
        {
            "位置": ["008号"],
            "日期": ["2026-07-23"],
            "时刻": ["00:00"],
            "环境温度": [20.0],
            "风速": [2.0],
            "风向": [90.0],
        }
    )
    uploaded = to_csv_upload("中文气象.csv", raw, encoding="gb18030")

    result = weather_upload.normalize_uploaded_weather_files(
        [uploaded], role="truth"
    )

    assert result.frame.loc[0, "tower_id"] == "008"


def test_normalize_uploaded_weather_files_supports_xlsx():
    weather_upload = weather_upload_module()
    output = BytesIO()
    make_tower_time_weather_dataframe().iloc[[0]].to_excel(output, index=False)
    uploaded = FakeUploadedFile("weather.xlsx", output.getvalue())

    result = weather_upload.normalize_uploaded_weather_files(
        [uploaded], role="physical"
    )

    assert result.frame.loc[0, "tower_id"] == "014"
    assert uploaded.getvalue_calls == 1


@pytest.mark.parametrize("extension", ["json", "xls"])
def test_normalize_uploaded_weather_files_rejects_unsupported_extension(extension):
    weather_upload = weather_upload_module()

    with pytest.raises(ValueError, match=f"不支持的气象文件格式.*{extension}"):
        weather_upload.normalize_uploaded_weather_files(
            [FakeUploadedFile(f"weather.{extension}", b"unsupported")],
            role="physical",
        )


def test_distinct_dataset_hashes_compares_every_nonempty_hash():
    weather_upload = weather_upload_module()
    physical = pd.DataFrame({"source_file_hash": ["", "physical-a", "shared"]})
    truth = pd.DataFrame({"source_file_hash": ["truth-a", "shared"]})

    with pytest.raises(ValueError, match="不能同时作为"):
        weather_upload.ensure_distinct_dataset_hashes(physical, truth)


def test_distinct_dataset_hashes_allows_disjoint_or_empty_hashes():
    weather_upload = weather_upload_module()
    physical = pd.DataFrame({"source_file_hash": ["", "physical-a"]})
    truth = pd.DataFrame({"source_file_hash": ["truth-a", None]})

    weather_upload.ensure_distinct_dataset_hashes(physical, truth)


def test_distinct_dataset_hashes_includes_files_removed_as_cross_file_duplicates():
    weather_upload = weather_upload_module()
    first = pd.DataFrame(
        {
            "位置": ["001号"],
            "日期": ["2026-07-23"],
            "时刻": ["00:00"],
            "环境温度": [20.0],
            "风速": [2.0],
            "风向": [90.0],
        }
    )
    duplicate = first.assign(风速=[3.0])
    physical = weather_upload.normalize_uploaded_weather_files(
        [
            to_csv_upload("first.csv", first),
            to_csv_upload("duplicate.csv", duplicate),
        ],
        role="physical",
    )
    duplicate_hash = physical.files[1].sha256
    truth = pd.DataFrame({"source_file_hash": [duplicate_hash]})
    assert duplicate_hash not in set(physical.frame["source_file_hash"])

    with pytest.raises(ValueError, match="不能同时作为"):
        weather_upload.ensure_distinct_dataset_hashes(physical, truth)
