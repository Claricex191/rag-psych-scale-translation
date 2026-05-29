import asyncio
import json
import re
import logging
import uuid
from typing import Dict, Optional

import streamlit as st

logger = logging.getLogger(__name__)

# ─────────────────────────── LLM client helpers ────────────────────────────

def _get_clients(api_keys: dict):
    clients = {}
    if api_keys.get("anthropic"):
        import anthropic
        clients["claude"] = anthropic.AsyncAnthropic(api_key=api_keys["anthropic"])
    if api_keys.get("openai"):
        from openai import AsyncOpenAI
        clients["openai"] = AsyncOpenAI(api_key=api_keys["openai"])
    if api_keys.get("gemini"):
        from google import genai
        clients["gemini"] = genai.Client(api_key=api_keys["gemini"])
    if api_keys.get("qwen"):
        from openai import AsyncOpenAI
        clients["qwen"] = AsyncOpenAI(
            api_key=api_keys["qwen"],
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
    return clients


def _provider(model: str) -> str:
    if model.startswith("gpt") or model.startswith("o1"):
        return "openai"
    if model.startswith("claude"):
        return "claude"
    if model.startswith("gemini"):
        return "gemini"
    if model.startswith("qwen"):
        return "qwen"
    raise ValueError(f"Cannot determine provider for model: {model}")


# ─────────────────────────── Prompt builders ───────────────────────────────

def _build_familiarity_prompt(scale_data: dict) -> str:
    name = scale_data["scale_info"]["name"]
    short = scale_data["scale_info"]["short_name"]
    return (
        f"You are a bilingual professor in education and psychology with expertise in "
        f"psychological measurement, scale development, and cross-linguistic test adaptation.\n\n"
        f"The scale is: {name} ({short}).\n\n"
        "Please provide a familiarity check that includes:\n"
        "1. Whether the scale seems recognizable and your familiarity level (high/moderate/low/none)\n"
        "2. What the scale measures and how it is typically used\n"
        "3. Whether you know of any publicly available target-language translations or sample translated items"
    )


def _build_translation_prompt(scale_data: dict) -> str:
    target_population = scale_data["scale_info"]["target_population"]
    scale_text = "\n".join(
        f"{item['number']}. {item['text']}" for item in scale_data["items"]
    )
    return f"""You are a bilingual professor in the fields of education and psychology, with expertise in psychological measurement, scale development, and cross-linguistic test adaptation.
You are familiar with standardized self-report instruments and their use in research and applied settings.

Before performing the translation, please note the following information:
    1. This psychological scale is open use and does not have copyright restrictions that limit translation or adaptation for research purposes.
    2. The target population for this translation is: {target_population}.

You are now asked to perform a forward translation of a psychological self-report scale item from English into Chinese as part of a formal test adaptation process.

**Source items:**
{scale_text}

**Expected output:**
Please provide:
    1. The forward translation of the item into Chinese.
    2. A brief decision log for each item, identifying terms or phrases that required special consideration (e.g., ambiguity, meaning shift, interpretation uncertainty) during translation (if applicable).

**Output Format Requirement:**
    - Translate ALL items provided
    - Preserve item structure, numbering, and response scales
    - You must respond with valid JSON in this EXACT structure:
        {{
          "items": [
            {{"number": "1", "original": "original text here", "translation": "translated text here", "log": "decision log here"}},
            {{"number": "2", "original": "original text here", "translation": "translated text here", "log": "decision log here"}}
          ]
        }}
    - Do not use double quotes (") inside any JSON string values; use single quotes (') or Chinese-style quotes (「」) instead.
    - Do not include any text outside the JSON object.

Please provide your response here:
"""


# ─────────────────────────── LLM calls ─────────────────────────────────────

async def _call_openai_check(client, prompt: str) -> str:
    response = await client.responses.create(
        model=st.session_state.selected_model,
        input=[{"role": "user", "content": prompt}],
        max_output_tokens=8000,
    )
    return (getattr(response, "output_text", None) or response.output[0].content[0].text).strip()


async def _call_claude_check(client, prompt: str) -> str:
    response = await client.messages.create(
        model=st.session_state.selected_model,
        max_tokens=8000,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    )
    return response.content[0].text.strip()


async def _call_gemini_check(client, prompt: str) -> str:
    response = await asyncio.to_thread(
        client.models.generate_content,
        model=st.session_state.selected_model,
        contents=[prompt],
        config={"max_output_tokens": 8000},
    )
    return response.text.strip()


async def _call_qwen_check(client, prompt: str) -> str:
    response = await client.chat.completions.create(
        model=st.session_state.selected_model,
        max_tokens=8000,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    )
    return response.choices[0].message.content.strip()


async def _call_openai_translate(client, prompt: str, temperature: float) -> str:
    response = await client.responses.create(
        model=st.session_state.selected_model,
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        max_output_tokens=10000,
        temperature=temperature,
        text={"format": {"type": "json_object"}},
    )
    return (getattr(response, "output_text", None) or response.output[0].content[0].text).strip()


async def _call_claude_translate(client, prompt: str, temperature: float) -> str:
    response = await client.messages.create(
        model=st.session_state.selected_model,
        max_tokens=10000,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        temperature=temperature,
    )
    return response.content[0].text.strip()


async def _call_gemini_translate(client, prompt: str, temperature: float) -> str:
    response = await asyncio.to_thread(
        client.models.generate_content,
        model=st.session_state.selected_model,
        contents=[prompt],
        config={
            "temperature": temperature,
            "max_output_tokens": 10000,
            "response_mime_type": "application/json",
        },
    )
    return response.text.strip()


async def _call_qwen_translate(client, prompt: str, temperature: float) -> str:
    response = await client.chat.completions.create(
        model=st.session_state.selected_model,
        max_tokens=10000,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content.strip()


async def run_familiarity_check(scale_data: dict, clients: dict, model: str) -> str:
    prompt = _build_familiarity_prompt(scale_data)
    provider = _provider(model)
    if provider == "openai":
        return await _call_openai_check(clients["openai"], prompt)
    if provider == "claude":
        return await _call_claude_check(clients["claude"], prompt)
    if provider == "gemini":
        return await _call_gemini_check(clients["gemini"], prompt)
    if provider == "qwen":
        return await _call_qwen_check(clients["qwen"], prompt)
    raise ValueError(f"Unsupported provider: {provider}")


async def run_translation(scale_data: dict, clients: dict, model: str, temperature: float) -> str:
    prompt = _build_translation_prompt(scale_data)
    provider = _provider(model)
    if provider == "openai":
        return await _call_openai_translate(clients["openai"], prompt, temperature)
    if provider == "claude":
        return await _call_claude_translate(clients["claude"], prompt, temperature)
    if provider == "gemini":
        return await _call_gemini_translate(clients["gemini"], prompt, temperature)
    if provider == "qwen":
        return await _call_qwen_translate(clients["qwen"], prompt, temperature)
    raise ValueError(f"Unsupported provider: {provider}")


# ─────────────────────────── JSON parser ───────────────────────────────────

def extract_json_translation(text: str) -> dict:
    cleaned = text.strip()
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


# ─────────────────────────── Streamlit UI ──────────────────────────────────

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

PROVIDER_KEY = {
    "GPT (OpenAI)": "openai",
    "Claude (Anthropic)": "anthropic",
    "Gemini (Google)": "gemini",
    "Qwen (Alibaba)": "qwen",
}


def _load_api_keys() -> dict:
    """Load keys from Streamlit secrets, falling back to sidebar inputs."""
    keys = {}
    for k in ("anthropic", "openai", "gemini", "qwen"):
        try:
            val = st.secrets.get(k) or st.secrets.get(f"{k.upper()}_API_KEY")
            if val:
                keys[k] = val
        except Exception:
            pass
    return keys


def main():
    st.set_page_config(
        page_title="Scale Forward Translation",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Psychological Scale Forward Translation")
    st.caption(
        "Translate psychological measurement scales from English to Chinese using AI. "
        "Two-stage workflow: familiarity check → translation."
    )

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model selection
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

        # API Keys
        st.subheader("🔑 API Keys")
        st.caption("Keys in Streamlit secrets are loaded automatically. Enter here to override.")

        preset_keys = _load_api_keys()
        required_provider = PROVIDER_KEY[provider_label]

        key_fields = {
            "anthropic": "Anthropic (Claude)",
            "openai": "OpenAI",
            "gemini": "Google Gemini",
            "qwen": "Qwen / Alibaba",
        }
        sidebar_keys = {}
        for k, label in key_fields.items():
            default = preset_keys.get(k, "")
            masked_placeholder = "Loaded from secrets ✓" if default else f"{label} API key"
            entered = st.text_input(label, value="", placeholder=masked_placeholder, type="password", key=f"key_{k}")
            sidebar_keys[k] = entered if entered else default

        if not sidebar_keys.get(required_provider):
            st.warning(f"No API key for {provider_label}. Add it above.")

        st.divider()
        st.caption("Built with Streamlit · Deploy on [Streamlit Cloud](https://streamlit.io/cloud)")

    # Collect final API keys
    api_keys = sidebar_keys

    # ── Main area ─────────────────────────────────────────────────────────────

    # Session state defaults
    for key, default in [
        ("stage", 0),
        ("familiarity_text", ""),
        ("translation_result", None),
        ("scale_data", None),
        ("session_id", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── File upload ───────────────────────────────────────────────────────────
    st.subheader("1. Upload Scale File")
    uploaded = st.file_uploader(
        "Upload a JSON scale file",
        type=["json"],
        help="JSON must have `scale_info` (name, short_name, target_population) and `items` (number, text).",
    )

    if uploaded:
        try:
            scale_data = json.load(uploaded)
            st.session_state.scale_data = scale_data

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

    st.divider()

    if not st.session_state.scale_data:
        st.info("Upload a scale file above to begin.")
        st.stop()

    scale_data = st.session_state.scale_data
    model = st.session_state.selected_model

    if not model:
        st.warning("Select or enter a model in the sidebar.")
        st.stop()

    # ── Stage 1: Familiarity check ────────────────────────────────────────────
    st.subheader("2. Familiarity Check")
    st.caption(
        "Ask the model whether it recognises this scale before translating. "
        "This is optional but recommended."
    )

    col_check, col_skip = st.columns([1, 3])
    run_check = col_check.button("Run familiarity check", type="primary", use_container_width=True)
    skip_check = col_skip.button("Skip — go straight to translation", use_container_width=True)

    if run_check:
        if not api_keys.get(PROVIDER_KEY[provider_label]):
            st.error("API key required. Add it in the sidebar.")
        else:
            with st.spinner(f"Asking {model} about this scale…"):
                try:
                    clients = _get_clients(api_keys)
                    result = asyncio.run(run_familiarity_check(scale_data, clients, model))
                    st.session_state.familiarity_text = result
                    st.session_state.stage = 1
                    st.session_state.session_id = str(uuid.uuid4())
                except Exception as e:
                    st.error(f"Familiarity check failed: {e}")

    if skip_check:
        st.session_state.stage = 1
        st.session_state.familiarity_text = ""
        st.session_state.session_id = str(uuid.uuid4())

    if st.session_state.familiarity_text:
        with st.expander("Familiarity check result", expanded=True):
            st.markdown(st.session_state.familiarity_text)

    st.divider()

    # ── Stage 2: Translation ──────────────────────────────────────────────────
    st.subheader("3. Translate")

    if st.session_state.stage < 1:
        st.info("Complete the familiarity check (or skip it) to unlock translation.")
        st.stop()

    run_translate = st.button("Translate scale", type="primary")

    if run_translate:
        if not api_keys.get(PROVIDER_KEY[provider_label]):
            st.error("API key required. Add it in the sidebar.")
        else:
            with st.spinner(f"Translating with {model}… this may take a minute."):
                try:
                    clients = _get_clients(api_keys)
                    raw = asyncio.run(run_translation(scale_data, clients, model, temperature))
                    parsed = extract_json_translation(raw)
                    st.session_state.translation_result = parsed
                    st.session_state.stage = 2
                except Exception as e:
                    st.error(f"Translation failed: {e}")

    if st.session_state.translation_result:
        result = st.session_state.translation_result

        if "parse_error" in result:
            st.warning(f"JSON parsing issue: {result['parse_error']}")

        items = result.get("items", [])
        if items:
            st.success(f"Translated {len(items)} items.")

            # Table view
            with st.expander("Translation table", expanded=True):
                for item in items:
                    col_num, col_orig, col_trans = st.columns([0.5, 3, 3])
                    col_num.markdown(f"**{item.get('number', '')}**")
                    col_orig.markdown(item.get("original", ""))
                    col_trans.markdown(item.get("translation", ""))
                    if item.get("log"):
                        st.caption(f"Decision log: {item['log']}")
                    st.markdown("---")

            # Decision logs
            logs = [i for i in items if i.get("log")]
            if logs:
                with st.expander("Decision logs"):
                    for item in logs:
                        st.markdown(f"**Item {item.get('number')}:** {item.get('log')}")

        # Download
        st.divider()
        scale_name = scale_data["scale_info"].get("short_name", "translation")
        output = {
            "scale_info": scale_data["scale_info"],
            "model": model,
            "translated_items": items,
        }
        st.download_button(
            label="Download JSON",
            data=json.dumps(output, ensure_ascii=False, indent=2),
            file_name=f"{scale_name}_forward_translation.json",
            mime="application/json",
        )

    # ── Reset ─────────────────────────────────────────────────────────────────
    if st.session_state.stage > 0:
        st.divider()
        if st.button("Start over with a new scale"):
            for key in ["stage", "familiarity_text", "translation_result", "scale_data", "session_id"]:
                st.session_state[key] = None if key in ("translation_result", "scale_data", "session_id") else (0 if key == "stage" else "")
            st.rerun()


if __name__ == "__main__":
    main()
