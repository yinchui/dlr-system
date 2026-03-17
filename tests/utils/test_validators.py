import pandas as pd

from utils.validators import validate_weather_columns


def test_validate_weather_columns_reports_missing_fields():
    errors = validate_weather_columns(pd.DataFrame({"位置": [36]}))
    assert any("环境温度" in error for error in errors)
