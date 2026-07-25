from __future__ import annotations

import streamlit as st

from config.config import AUDIT_LOG_DIR, SAG_RESULT_DIR
from modules.sag_validation import (
    SagValidationSnapshot,
    build_visible_sag_result,
    run_sag_validation_upload,
)
from utils.audit_log import JsonAuditLogger


_RESULT_STATE_KEYS = (
    "sag_visible_result",
    "sag_result_id",
    "sag_result_persisted",
    "sag_result_context",
)


def _clear_cached_result() -> None:
    for state_key in _RESULT_STATE_KEYS:
        st.session_state.pop(state_key, None)


def _upload_identity(uploaded_file) -> str:
    file_id = getattr(uploaded_file, "file_id", None)
    if file_id:
        return str(file_id)
    return f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 0)}"


st.set_page_config(page_title="弧垂后验证", layout="wide")
st.title("弧垂后验证")

inclination_file = st.file_uploader(
    "上传倾角数据",
    type=["csv", "xlsx"],
    key="sag_inclination_upload",
)

snapshot = st.session_state.get("sag_validation_snapshot")
if not isinstance(snapshot, SagValidationSnapshot):
    _clear_cached_result()
    st.info("请先在主页面完成一次 DLR 计算，再返回本页进行后验证。")
elif inclination_file is not None:
    selected_tower_id = st.selectbox(
        "杆塔",
        options=snapshot.tower_ids,
        key="sag_selected_tower",
    )
    result_context = (
        snapshot.source_run_id,
        selected_tower_id,
        _upload_identity(inclination_file),
    )
    if st.session_state.get("sag_result_context") not in (
        None,
        result_context,
    ):
        _clear_cached_result()
    if st.button("运行后验证", type="primary", key="sag_run_validation"):
        _clear_cached_result()
        try:
            result = run_sag_validation_upload(
                snapshot,
                inclination_file,
                selected_tower_id=selected_tower_id,
                output_dir=SAG_RESULT_DIR,
                audit_logger=JsonAuditLogger(AUDIT_LOG_DIR),
            )
            visible = build_visible_sag_result(result)
            st.session_state["sag_visible_result"] = visible
            st.session_state["sag_result_id"] = result.result_id
            st.session_state["sag_result_persisted"] = (
                result.result_path is not None
            )
            st.session_state["sag_result_context"] = result_context
        except Exception as exc:
            st.error(f"弧垂后验证失败：{exc}")

    if st.session_state.get("sag_result_context") == result_context:
        visible = st.session_state["sag_visible_result"]
        result_id = st.session_state["sag_result_id"]
        if not st.session_state["sag_result_persisted"]:
            st.warning("计算已完成，但后台结果未保存，请检查运行目录。")
        st.dataframe(visible, width="stretch", hide_index=True)
        st.download_button(
            "下载结果",
            data=visible.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"sag_validation_{result_id}.csv",
            mime="text/csv",
            key="sag_download_result",
        )
else:
    _clear_cached_result()
