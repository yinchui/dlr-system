import pandas as pd

from modules.visualization import build_line_rating_figure


def test_build_line_rating_figure_contains_dynamic_and_static_traces():
    fig = build_line_rating_figure(
        timestamps=pd.to_datetime(["2025-12-10 00:00", "2025-12-10 01:00"]),
        dynamic_current=[800, 820],
        static_current=700,
    )
    assert len(fig.data) == 2
