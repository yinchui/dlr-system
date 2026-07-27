import pandas as pd

from modules.visualization import build_line_rating_figure


def test_build_line_rating_figure_contains_only_dynamic_rating():
    fig = build_line_rating_figure(
        timestamps=pd.to_datetime(["2025-12-10 00:00", "2025-12-10 01:00"]),
        dynamic_current=[640, 656],
    )
    assert len(fig.data) == 1
    assert fig.data[0].name == "动态额定值"
