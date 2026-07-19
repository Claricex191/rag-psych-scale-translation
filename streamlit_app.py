import json
import logging
import os
import re
import tempfile
import uuid
from pathlib import Path

import streamlit as st

from rag_client import RAGClient

logger = logging.getLogger(__name__)

FORWARD_URL = os.getenv("FORWARD_SERVER_URL", "http://127.0.0.1:8000")
BACKWARD_URL = os.getenv("BACKWARD_SERVER_URL", "http://127.0.0.1:8001")
APP_PASSWORD = os.getenv("APP_PASSWORD")

UPLOAD_DIR = Path(tempfile.gettempdir()) / "rag_mt_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OPTIONS = {
    "GPT (OpenAI)": [
        "gpt-4.1-2025-04-14",
        "gpt-4o-2024-11-20",
        "o3-2025-04-16",
        "gpt-5.2-2025-12-11",
    ],
    "Claude (Anthropic)": [
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-6",
        "claude-opus-4-8",
        "claude-haiku-4-5-20251001",
    ],
    "Gemini (Google)": [
        "gemini-2.5-pro-preview-06-05",
        "gemini-2.5-flash-preview-05-20",
        "gemini-3-pro-preview",
    ],
    "Qwen (Alibaba)": [
        "qwen3-vl-plus-2025-12-19",
        "qwen-max",
        "qwen-plus",
    ],
}


# ─────────────────────────── JSON parser ───────────────────────────────────
# The RAG servers return the LLM's translation as a JSON *string* embedded in
# their response (server_forward.py's _extract_json_translation / same in
# server_backward.py) — this turns that string into a dict for the UI.

def extract_json_translation(text) -> dict:
    if isinstance(text, dict):
        return text

    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
        cleaned = re.sub(r'\n?```\s*$', '', cleaned)
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        if "items" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    start = cleaned.find('{')
    if start == -1:
        return {"items": [], "parse_error": "No JSON object found in response"}

    depth, end = 0, start
    for i in range(start, len(cleaned)):
        if cleaned[i] == '{':
            depth += 1
        elif cleaned[i] == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    try:
        parsed = json.loads(cleaned[start:end])
        if "items" in parsed or "translation" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    return {"items": [], "parse_error": "Could not extract valid JSON from response"}


def _forward_client() -> RAGClient:
    client = RAGClient(FORWARD_URL)
    client.session_id = st.session_state.get("session_id")
    return client


def _write_temp_json(data: dict, tag: str) -> str:
    path = UPLOAD_DIR / f"{tag}_{uuid.uuid4()}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return str(path)


def _render_translation_table(items: list, key_prefix: str):
    for item in items:
        col_num, col_orig, col_trans = st.columns([0.5, 3, 3])
        col_num.markdown(f"**{item.get('number', '')}**")
        col_orig.markdown(item.get("original", ""))
        col_trans.markdown(item.get("translation", ""))
        if item.get("log"):
            st.caption(f"Decision log: {item['log']}")
        st.markdown("---")


def _guideline_uploader(label: str, base_url: str, state_key: str):
    with st.expander(label, expanded=False):
        st.caption(
            "Only needed the first time a guideline document is used — indexed "
            "PDFs persist on the server. Skip if reference documents are already indexed."
        )
        files = st.file_uploader(
            "Upload guideline PDF(s)", type=["pdf"], accept_multiple_files=True, key=f"{state_key}_uploader"
        )
        if files and st.button("Index PDF(s)", key=f"{state_key}_button"):
            client = RAGClient(base_url)
            for f in files:
                with st.spinner(f"Indexing {f.name}…"):
                    result = client.upload_guideline(f.name, f.getvalue())
                if result.get("success"):
                    st.success(f"{f.name}: {result.get('message', 'indexed')}")
                else:
                    st.error(f"{f.name}: {result.get('error', 'failed to index')}")


def _password_gate():
    if not APP_PASSWORD:
        return
    if st.session_state.get("authenticated"):
        return

    st.title("🧪 Scale Translation")
    entered = st.text_input("Password", type="password")
    if st.button("Enter"):
        if entered == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


def main():
    st.set_page_config(
        page_title="Scale Forward Translation",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _password_gate()

    st.title("Psychological Scale Forward Translation")
    st.caption(
        "RAG-grounded translation of psychological measurement scales from English "
        "to Chinese, with an optional backward-translation evaluation step."
    )

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("Model")
        provider_label = st.selectbox("Provider", list(MODEL_OPTIONS.keys()))
        model_list = MODEL_OPTIONS[provider_label]
        model_input_mode = st.radio("Model", ["Choose from list", "Enter manually"], horizontal=True)
        if model_input_mode == "Choose from list":
            selected_model = st.selectbox("Model name", model_list)
        else:
            selected_model = st.text_input("Model ID", placeholder="e.g. gpt-4o-2024-11-20")
        st.session_state.selected_model = selected_model

        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)

        st.divider()
        st.caption(
            "API keys for the selected model's provider are configured on the "
            "server — nothing to enter here."
        )

    model = st.session_state.selected_model
    if not model:
        st.warning("Select or enter a model in the sidebar.")
        st.stop()

    # ── Session state defaults ──────────────────────────────────────────────
    for key, default in [
        ("stage", 0),
        ("familiarity_text", ""),
        ("translation_result", None),
        ("backward_result", None),
        ("scale_data", None),
        ("scale_path", None),
        ("session_id", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Reference documents ─────────────────────────────────────────────────
    st.subheader("1. Reference Documents (optional)")
    _guideline_uploader("Forward-translation guideline PDFs", FORWARD_URL, "fwd_guidelines")
    _guideline_uploader("Backward-translation guideline PDFs", BACKWARD_URL, "bwd_guidelines")

    st.divider()

    # ── Scale upload ─────────────────────────────────────────────────────────
    st.subheader("2. Upload Scale File")
    uploaded = st.file_uploader(
        "Upload a JSON scale file",
        type=["json"],
        help="JSON must have `scale_info` (name, short_name, target_population) and `items` (number, text).",
    )

    if uploaded:
        try:
            scale_data = json.load(uploaded)
            st.session_state.scale_data = scale_data
            st.session_state.scale_path = _write_temp_json(scale_data, "scale")

            info = scale_data.get("scale_info", {})
            items = scale_data.get("items", [])
            col1, col2, col3 = st.columns(3)
            col1.metric("Scale", info.get("short_name", "—"))
            col2.metric("Items", len(items))
            col3.metric("Population", info.get("target_population", "—"))

            with st.expander("Preview items"):
                for item in items[:5]:
                    st.markdown(f"**{item['number']}.** {item['text']}")
                if len(items) > 5:
                    st.caption(f"… and {len(items) - 5} more items")
        except Exception as e:
            st.error(f"Could not parse JSON: {e}")
            st.session_state.scale_data = None
            st.session_state.scale_path = None

    st.divider()

    if not st.session_state.scale_data:
        st.info("Upload a scale file above to begin.")
        st.stop()

    scale_data = st.session_state.scale_data

    # ── Stage 1: Familiarity check ────────────────────────────────────────────
    st.subheader("3. Familiarity Check")
    st.caption("Ask the model whether it recognises this scale before translating.")

    if st.button("Run familiarity check", type="primary"):
        with st.spinner(f"Asking {model} about this scale…"):
            try:
                result = _forward_client().check_familiarity(
                    scale_path=st.session_state.scale_path, model=model
                )
                if result.get("success"):
                    st.session_state.familiarity_text = result.get("familiarity_summary", "")
                    st.session_state.session_id = result.get("session_id")
                    st.session_state.stage = 1
                else:
                    st.error(f"Familiarity check failed: {result.get('error', 'unknown error')}")
            except Exception as e:
                st.error(f"Familiarity check failed: {e}")

    if st.session_state.familiarity_text:
        with st.expander("Familiarity check result", expanded=True):
            st.markdown(st.session_state.familiarity_text)

    st.divider()

    # ── Stage 2: Translation ──────────────────────────────────────────────────
    st.subheader("4. Translate")

    if st.session_state.stage < 1:
        st.info("Complete the familiarity check above to unlock translation.")
        st.stop()

    if st.button("Translate scale", type="primary"):
        with st.spinner(f"Translating with {model}… this may take a minute."):
            try:
                result = _forward_client().translate(
                    scale_path=st.session_state.scale_path, model=model, temperature=temperature
                )
                if result.get("translation"):
                    st.session_state.translation_result = extract_json_translation(result["translation"])
                    st.session_state.stage = 2
                else:
                    st.error(f"Translation failed: {result.get('error', 'unknown error')}")
            except Exception as e:
                st.error(f"Translation failed: {e}")

    if st.session_state.translation_result:
        result = st.session_state.translation_result

        if "parse_error" in result:
            st.warning(f"JSON parsing issue: {result['parse_error']}")

        items = result.get("items", [])
        if items:
            st.success(f"Translated {len(items)} items.")

            with st.expander("Translation table", expanded=True):
                _render_translation_table(items, "fwd")

            logs = [i for i in items if i.get("log")]
            if logs:
                with st.expander("Decision logs"):
                    for item in logs:
                        st.markdown(f"**Item {item.get('number')}:** {item.get('log')}")

        st.divider()
        scale_name = scale_data["scale_info"].get("short_name", "translation")
        output = {
            "scale_info": scale_data["scale_info"],
            "model": model,
            "translated_items": items,
        }
        st.download_button(
            label="Download forward translation JSON",
            data=json.dumps(output, ensure_ascii=False, indent=2),
            file_name=f"{scale_name}_forward_translation.json",
            mime="application/json",
        )

        # ── Stage 3: Optional backward translation ──────────────────────────
        st.divider()
        st.subheader("5. Backward Translation (optional evaluation)")
        st.caption(
            "Translates the forward-translation output back into English, so you can "
            "visually compare it against the original as a fidelity/stability check."
        )

        if items and st.checkbox("Run backward translation"):
            if st.button("Run", type="primary", key="run_backward"):
                with st.spinner(f"Back-translating with {model}…"):
                    try:
                        bwd_path = _write_temp_json({"items": items}, "backward")
                        bwd_client = RAGClient(BACKWARD_URL)
                        bwd_result = bwd_client.back_translate(
                            scale_path=bwd_path, model=model, temperature=temperature
                        )
                        if bwd_result.get("translation"):
                            st.session_state.backward_result = extract_json_translation(bwd_result["translation"])
                        else:
                            st.error(f"Backward translation failed: {bwd_result.get('error', 'unknown error')}")
                    except Exception as e:
                        st.error(f"Backward translation failed: {e}")

            if st.session_state.backward_result:
                bwd = st.session_state.backward_result
                if "parse_error" in bwd:
                    st.warning(f"JSON parsing issue: {bwd['parse_error']}")

                bwd_items = bwd.get("items", [])
                if bwd_items:
                    st.success(f"Back-translated {len(bwd_items)} items.")
                    with st.expander("Backward translation table", expanded=True):
                        _render_translation_table(bwd_items, "bwd")

                    st.download_button(
                        label="Download backward translation JSON",
                        data=json.dumps({"back_translated_items": bwd_items}, ensure_ascii=False, indent=2),
                        file_name=f"{scale_name}_backward_translation.json",
                        mime="application/json",
                    )

    # ── Reset ─────────────────────────────────────────────────────────────────
    if st.session_state.stage > 0:
        st.divider()
        if st.button("Start over with a new scale"):
            for key in ["stage", "familiarity_text", "translation_result", "backward_result",
                        "scale_data", "scale_path", "session_id"]:
                st.session_state[key] = None if key not in ("stage",) else 0
            st.session_state.familiarity_text = ""
            st.rerun()


if __name__ == "__main__":
    main()
