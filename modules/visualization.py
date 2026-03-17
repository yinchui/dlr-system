# modules/visualization.py
import plotly.graph_objects as go


def build_line_rating_figure(timestamps, dynamic_current, static_current):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timestamps, y=dynamic_current, mode="lines+markers", name="动态载流量"))
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=[static_current] * len(dynamic_current),
            mode="lines",
            line={"dash": "dash"},
            name="静态额定值",
        )
    )
    fig.update_layout(xaxis_title="日期时间", yaxis_title="允许电流 (A)", hovermode="x unified")
    return fig


def build_correction_comparison_figure(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["wind_speed_raw"], name="修正前风速"))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["wind_speed_corrected"], name="修正后风速"))
    return fig


def build_prediction_comparison_figure(df, target_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df[f"{target_name}_final"], name="最终预测"))
    if f"{target_name}_actual" in df.columns:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df[f"{target_name}_actual"], name="监测值"))
    return fig
