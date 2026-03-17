# utils/validators.py
REQUIRED_WEATHER_COLUMNS = ["位置", "日期", "时刻", "环境温度", "风速", "风向"]
REQUIRED_TOWER_COLUMNS = ["运行编号", "经度", "纬度"]
REQUIRED_MONITOR_COLUMNS = ["timestamp", "tower_id"]


def validate_weather_columns(df):
    return [f"缺少列: {column}" for column in REQUIRED_WEATHER_COLUMNS if column not in df.columns]


def validate_tower_columns(df):
    return [f"缺少列: {column}" for column in REQUIRED_TOWER_COLUMNS if not any(column in str(c) for c in df.columns)]


def validate_monitor_columns(df):
    return [f"缺少列: {column}" for column in REQUIRED_MONITOR_COLUMNS if column not in df.columns]
