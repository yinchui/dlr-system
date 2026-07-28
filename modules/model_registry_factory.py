from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional

from config.config import (
    MODEL_DIR,
    SUPABASE_MODEL_SETTING_NAMES,
    load_supabase_model_config,
)
from modules.model_registry import ModelRegistry
from modules.supabase_model_registry import (
    SupabaseModelRegistry,
    SupabaseModelStore,
)


def _runtime_streamlit_secrets() -> Mapping[str, object]:
    import streamlit as st
    from streamlit.errors import StreamlitSecretNotFoundError

    try:
        runtime_secrets = st.secrets
        return {
            name: runtime_secrets[name]
            for name in SUPABASE_MODEL_SETTING_NAMES
            if name in runtime_secrets
        }
    except StreamlitSecretNotFoundError:
        return {}


def create_model_registry(
    model_dir: Path | str = MODEL_DIR,
    *,
    ai_enabled: bool = True,
    secrets: Optional[Mapping[str, object]] = None,
    environ: Optional[Mapping[str, str]] = None,
    audit_logger: Optional[Any] = None,
    audit_run_id: Optional[str] = None,
    audit_result_id: Optional[str] = None,
) -> Optional[ModelRegistry]:
    secret_values = (
        _runtime_streamlit_secrets() if secrets is None else secrets
    )
    config = load_supabase_model_config(
        secrets=secret_values,
        environ=os.environ if environ is None else environ,
    )
    if not ai_enabled:
        return None
    audit_options = {
        "audit_logger": audit_logger,
        "audit_run_id": audit_run_id,
        "audit_result_id": audit_result_id,
    }
    try:
        if config is None:
            return ModelRegistry(model_dir, **audit_options)
        store = SupabaseModelStore.from_credentials(
            config.url,
            config.secret_key,
            bucket=config.bucket,
        )
        return SupabaseModelRegistry(store, **audit_options)
    except OSError:
        return None
