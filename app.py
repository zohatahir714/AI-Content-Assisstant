"""
AI Content Assistant
---------------------
Generate platform-optimized content (captions, keywords, hashtags) for
multiple social platforms at once, powered by Groq's free LLM API.

Run locally:
    streamlit run app.py

Deploy:
    Push to GitHub, then deploy on https://share.streamlit.io
    Set GROQ_API_KEY in Streamlit Cloud "Secrets" (or paste it in the sidebar).
"""

import json
import re
from datetime import datetime

import streamlit as st
from groq import Groq

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="ContentForge AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# STATIC DATA
# --------------------------------------------------------------------------
PLATFORM_SPECS = {
    "Instagram": {
        "icon": "📸",
        "accent": "#E0637E",
        "limit": 2200,
        "hashtags": "15-20",
        "style": "Visually descriptive, engaging, uses line breaks and emojis "
                 "naturally, ends with a call-to-action or question to boost comments.",
    },
    "X (Twitter)": {
        "icon": "🐦",
        "accent": "#6E7B99",
        "limit": 280,
        "hashtags": "1-2",
        "style": "Punchy, concise, conversational, witty. Every word must earn "
                 "its place. No fluff.",
    },
    "LinkedIn": {
        "icon": "💼",
        "accent": "#4C6FA5",
        "limit": 3000,
        "hashtags": "3-5",
        "style": "Professional, insight-driven, storytelling opener, structured "
                 "with short paragraphs or line breaks, thought-leadership tone.",
    },
    "Facebook": {
        "icon": "👍",
        "accent": "#7C93C9",
        "limit": 2000,
        "hashtags": "2-4",
        "style": "Friendly, conversational, community-oriented, slightly longer "
                 "storytelling, encourages shares and comments.",
    },
    "TikTok": {
        "icon": "🎵",
        "accent": "#3FB6A8",
        "limit": 300,
        "hashtags": "4-6",
        "style": "Short, high-energy, trend-aware hook in the first line, casual "
                 "Gen-Z tone, uses trending phrasing.",
    },
    "YouTube": {
        "icon": "▶️",
        "accent": "#D6555A",
        "limit": 1000,
        "hashtags": "3-5",
        "style": "SEO-aware video description: strong hook line, keyword-rich "
                 "summary, structured with short paragraphs.",
    },
    "Pinterest": {
        "icon": "📌",
        "accent": "#C77B8A",
        "limit": 500,
        "hashtags": "2-5",
        "style": "Descriptive, keyword-rich, inspirational, search-friendly "
                 "(Pinterest behaves like a search engine).",
    },
}

CONTENT_TYPES = [
    "Social Media Post",
    "Product Launch / Promo",
    "Blog Post Summary",
    "Ad Copy",
    "Event Announcement",
    "Educational / Tip Post",
    "Brand Story",
]

TONES = [
    "Professional", "Casual & Friendly", "Witty & Humorous", "Inspirational",
    "Bold & Confident", "Warm & Empathetic", "Luxury & Elegant", "Playful",
    "Minimalist", "Persuasive / Sales-driven",
]

GROQ_MODELS = {
    "GPT-OSS 120B (best quality)": "openai/gpt-oss-120b",
    "GPT-OSS 20B (fastest)": "openai/gpt-oss-20b",
}

# --------------------------------------------------------------------------
# CUSTOM CSS — an "editorial desk" concept: a dark workspace where each
# generated caption appears as a warm paper proof card, like copy pulled
# fresh off a press. One accent color carries every action; platform colors
# are informational only (left-edge bar on each card), never decorative.
# --------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;1,9..144,500&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --desk: #12141F;
        --panel: #1B1E2E;
        --rule: #313548;
        --proof: #F4EEDD;
        --proof-rule: #E4D8B8;
        --ink: #22201A;
        --text: #F1F1F7;
        --text-muted: #9397AC;
        --signal: #6C63FF;
        --signal-2: #8B5CF6;
        --signal-ink: #FFFFFF;
        --danger: #E0757B;
    }

    .stApp { background: var(--desk); }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ---------- Hero ---------- */
    .hero {
        padding: 2.3rem 2.2rem; border-radius: 20px; margin-bottom: 1.8rem;
        background: linear-gradient(135deg, var(--signal) 0%, var(--signal-2) 100%);
        box-shadow: 0 10px 34px rgba(108, 99, 255, 0.32);
    }
    .hero .mark {
        font-family: 'Fraunces', serif; font-weight: 600; font-style: italic;
        font-size: 2.4rem; color: white; letter-spacing: -0.3px; margin: 0 0 0.4rem 0;
    }
    .hero .tagline { color: rgba(255,255,255,0.92); font-size: 1.02rem; margin: 0; }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--rule); }
    section[data-testid="stSidebar"] h3 { font-weight: 600; color: var(--text); }

    /* ---------- Form panel ---------- */
    div[data-testid="stForm"] {
        border: 1px solid var(--rule); border-radius: 12px;
        padding: 1.6rem 1.7rem; background: var(--panel);
    }
    div[data-testid="stForm"] label p { font-weight: 500; color: var(--text) !important; }

    /* Inputs */
    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
        background: var(--desk) !important; border: 1px solid var(--rule) !important;
        border-radius: 8px !important; color: var(--text) !important;
    }
    .stMultiSelect div[data-baseweb="select"] > div {
        background: var(--desk) !important; border: 1px solid var(--rule) !important;
        border-radius: 8px !important;
    }
    span[data-baseweb="tag"] { background: var(--signal) !important; color: var(--signal-ink) !important; }

    /* Focus ring — accessibility */
    .stTextInput input:focus, .stTextArea textarea:focus,
    .stButton>button:focus-visible, .stDownloadButton>button:focus-visible {
        outline: 2px solid var(--signal) !important; outline-offset: 1px;
    }

    /* ---------- Buttons ---------- */
    .stButton>button {
        border-radius: 8px; font-weight: 600; border: none;
        background: linear-gradient(135deg, var(--signal) 0%, var(--signal-2) 100%);
        color: var(--signal-ink);
        transition: filter 0.15s ease;
    }
    .stButton>button:hover { filter: brightness(1.08); }
    .stDownloadButton>button {
        border-radius: 8px; font-weight: 600;
        background: transparent; color: var(--text); border: 1px solid var(--rule);
        transition: border-color 0.15s ease;
    }
    .stDownloadButton>button:hover { border-color: var(--signal); color: var(--signal); }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] { gap: 1.5rem; border-bottom: 1px solid var(--rule); }
    .stTabs [data-baseweb="tab"] {
        background: transparent; color: var(--text-muted); font-weight: 500;
        padding: 0 0.1rem 0.7rem 0.1rem;
    }
    .stTabs [aria-selected="true"] { color: var(--text) !important; }
    .stTabs [data-baseweb="tab-highlight"] { background-color: var(--signal) !important; height: 2px; }

    /* ---------- Proof card (a generated result) ---------- */
    .proof-card {
        background: var(--proof); border-radius: 6px;
        border-left: 4px solid var(--accent, var(--signal));
        padding: 1.3rem 1.5rem 0.4rem 1.4rem;
        margin: 1.1rem 0 0.8rem 0;
        box-shadow: 0 6px 18px rgba(0,0,0,0.22);
    }
    .proof-header {
        display: flex; align-items: baseline; justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    .proof-title { font-weight: 600; color: var(--ink); font-size: 1.02rem; }
    .char-badge {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        background: rgba(34,32,26,0.08); color: var(--ink); font-size: 0.78rem;
        font-weight: 500; opacity: 0.75;
    }
    .char-badge.over { background: rgba(224,117,123,0.18); color: #A23A40; opacity: 1; }

    .proof-card .stTextArea textarea {
        background: transparent !important; border: none !important; color: var(--ink) !important;
        font-family: 'Fraunces', serif; font-size: 1.02rem; line-height: 1.65; padding: 0 !important;
    }
    .proof-card .stTextArea textarea:focus { outline: none !important; }

    .pill-label { color: var(--ink); opacity: 0.6; font-size: 0.82rem; font-weight: 500;
                  margin: 0.7rem 0 0.35rem 0; }
    .keyword-pill {
        display: inline-block; background: transparent; color: var(--ink);
        padding: 3px 11px; border-radius: 999px; font-size: 0.83rem;
        margin: 0 6px 6px 0; border: 1px solid var(--proof-rule);
    }
    .hashtag-pill {
        display: inline-block; background: var(--signal); color: white;
        padding: 3px 11px; border-radius: 999px; font-size: 0.83rem;
        margin: 0 6px 6px 0;
    }

    .proof-card + div[data-testid="stDownloadButton"] { margin-bottom: 0.6rem; }

    @media (prefers-reduced-motion: reduce) {
        .stButton>button, .stDownloadButton>button { transition: none !important; }
    }
    @media (max-width: 640px) {
        .hero { padding: 1.6rem 1.4rem; }
        .hero .mark { font-size: 1.8rem; }
        div[data-testid="stForm"] { padding: 1.1rem; }
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = {}
if "generated_at" not in st.session_state:
    st.session_state.generated_at = None


# --------------------------------------------------------------------------
# GROQ HELPERS
# --------------------------------------------------------------------------
def get_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


def build_prompt(content_type, platform, topic, audience, tone, extra_notes, spec):
    return f"""You are an expert social media copywriter and SEO strategist.

Create a "{content_type}" for the platform **{platform}**.

TOPIC: {topic}
TARGET AUDIENCE: {audience}
VOICE & TONE: {tone}
PLATFORM STYLE RULES: {spec['style']}
HARD CHARACTER LIMIT FOR THE CAPTION: {spec['limit']} characters (must not exceed this).
NUMBER OF HASHTAGS TO GENERATE: {spec['hashtags']}
{"ADDITIONAL INSTRUCTIONS: " + extra_notes if extra_notes else ""}

Respond with STRICT JSON only, no markdown fences, no commentary, using exactly this schema:
{{
  "caption": "the full ready-to-post caption text, formatted naturally for {platform}",
  "keywords": ["5 to 8 relevant SEO/content keywords or phrases"],
  "hashtags": ["hashtags WITHOUT the # symbol, lowercase, no spaces"]
}}
"""


def extract_json(raw_text: str) -> dict:
    """Best-effort extraction of a JSON object from the model's raw response."""
    raw_text = raw_text.strip()
    # Strip markdown code fences if present
    raw_text = re.sub(r"^```(json)?", "", raw_text.strip())
    raw_text = re.sub(r"```$", "", raw_text.strip())
    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # Fallback: grab the first {...} block
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def generate_for_platform(client, model, content_type, platform, topic, audience, tone, extra_notes):
    spec = PLATFORM_SPECS[platform]
    prompt = build_prompt(content_type, platform, topic, audience, tone, extra_notes, spec)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise assistant that replies with valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
        max_tokens=2048,
        reasoning_effort="low",
        response_format={"type": "json_object"},
    )
    raw = completion.choices[0].message.content
    if not raw or not raw.strip():
        raise ValueError(
            "Model returned an empty response (likely ran out of tokens while "
            "reasoning). Try again or switch to the GPT-OSS 20B model."
        )
    data = extract_json(raw)

    # Normalize / defend against missing keys
    caption = str(data.get("caption", "")).strip()
    keywords = data.get("keywords", []) or []
    hashtags = data.get("hashtags", []) or []
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",") if k.strip()]
    if isinstance(hashtags, str):
        hashtags = [h.strip().lstrip("#") for h in hashtags.split(",") if h.strip()]
    hashtags = [h.lstrip("#").replace(" ", "") for h in hashtags]

    return {"caption": caption, "keywords": keywords, "hashtags": hashtags}


# --------------------------------------------------------------------------
# SIDEBAR — API KEY & MODEL
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Groq setup")
    st.caption("Get a free key at [console.groq.com/keys](https://console.groq.com/keys)")

    default_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""
    api_key_input = st.text_input(
        "Groq API key",
        value="",
        type="password",
        placeholder="gsk_...",
        help="Stored only for this session. Leave blank to use the key set in Streamlit Secrets.",
    )
    api_key = api_key_input or default_key

    if api_key:
        st.success("API key loaded")
    else:
        st.warning("No API key found. Enter one above or add GROQ_API_KEY to Secrets.")

    model_label = st.selectbox("Model", list(GROQ_MODELS.keys()), index=0)
    model_id = GROQ_MODELS[model_label]

    st.divider()
    st.markdown("### About")
    st.caption(
        "ContentForge writes platform-native captions, keywords, and "
        "hashtags for several channels at once, using Groq's free "
        "hosted LLMs."
    )

# --------------------------------------------------------------------------
# HERO
# --------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <p class="mark">ContentForge</p>
    <p class="tagline">Write the idea once. Get copy that already fits each feed.</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# INPUT FORM
# --------------------------------------------------------------------------
with st.form("content_form"):
    col1, col2 = st.columns(2)
    with col1:
        content_type = st.selectbox("Content type", CONTENT_TYPES)
        topic = st.text_area(
            "Topic or brief",
            placeholder="e.g. Launching our new eco-friendly water bottle made from recycled ocean plastic",
            height=100,
        )
        audience = st.text_input(
            "Target audience",
            placeholder="e.g. Environmentally-conscious millennials, 25-40",
        )
    with col2:
        platforms = st.multiselect(
            "Platforms",
            list(PLATFORM_SPECS.keys()),
            default=["Instagram", "X (Twitter)", "LinkedIn"],
        )
        tone = st.selectbox("Voice and tone", TONES)
        extra_notes = st.text_input(
            "Extra instructions (optional)",
            placeholder="e.g. mention our 20% launch discount code LAUNCH20",
        )

    submitted = st.form_submit_button("Generate content", use_container_width=True)

# --------------------------------------------------------------------------
# GENERATION
# --------------------------------------------------------------------------
if submitted:
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar first.")
    elif not topic.strip():
        st.error("Please describe a topic or brief.")
    elif not platforms:
        st.error("Please select at least one platform.")
    else:
        client = get_client(api_key)
        results = {}
        progress = st.progress(0, text="Warming up the model...")
        errors = []

        for i, platform in enumerate(platforms):
            progress.progress(
                (i) / len(platforms),
                text=f"Writing your {platform} content...",
            )
            try:
                results[platform] = generate_for_platform(
                    client, model_id, content_type, platform, topic, audience, tone, extra_notes
                )
            except Exception as e:
                errors.append(f"{platform}: {e}")
            progress.progress((i + 1) / len(platforms))

        progress.empty()
        st.session_state.results = results
        st.session_state.generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

        if errors:
            for err in errors:
                st.error(f"Couldn't generate for {err}")
        if results:
            st.success(f"Done — generated content for {len(results)} platform(s).")

# --------------------------------------------------------------------------
# RESULTS DISPLAY
# --------------------------------------------------------------------------
if st.session_state.results:
    st.markdown("## Your content")
    st.caption(f"Generated {st.session_state.generated_at}")

    tabs = st.tabs([f"{PLATFORM_SPECS[p]['icon']} {p}" for p in st.session_state.results.keys()])

    all_text_export = []

    for tab, (platform, data) in zip(tabs, st.session_state.results.items()):
        spec = PLATFORM_SPECS[platform]
        with tab:
            char_count = len(data["caption"])
            over_limit = char_count > spec["limit"]
            badge_class = "char-badge over" if over_limit else "char-badge"

            st.markdown(f"""
            <div class="proof-card" style="--accent: {spec['accent']};">
                <div class="proof-header">
                    <span class="proof-title">{spec['icon']} {platform}</span>
                    <span class="{badge_class}">{char_count} / {spec['limit']} chars</span>
                </div>
            """, unsafe_allow_html=True)

            st.text_area(
                "Caption",
                value=data["caption"],
                height=180,
                key=f"caption_{platform}",
                label_visibility="collapsed",
            )

            if data["keywords"]:
                st.markdown('<p class="pill-label">Keywords</p>', unsafe_allow_html=True)
                st.markdown(
                    "".join(f'<span class="keyword-pill">{k}</span>' for k in data["keywords"]),
                    unsafe_allow_html=True,
                )

            if data["hashtags"]:
                st.markdown('<p class="pill-label">Hashtags</p>', unsafe_allow_html=True)
                st.markdown(
                    "".join(f'<span class="hashtag-pill">#{h}</span>' for h in data["hashtags"]),
                    unsafe_allow_html=True,
                )
                hashtag_line = " ".join(f"#{h}" for h in data["hashtags"])
            else:
                hashtag_line = ""

            st.markdown("<div style='height: 0.9rem'></div></div>", unsafe_allow_html=True)

            full_block = f"{data['caption']}\n\n{hashtag_line}"
            st.download_button(
                f"Download the {platform} version (.txt)",
                data=full_block,
                file_name=f"{platform.replace(' ', '_').lower()}_content.txt",
                mime="text/plain",
                key=f"dl_{platform}",
            )

            all_text_export.append(
                f"===== {platform} =====\n\n{data['caption']}\n\n"
                f"Keywords: {', '.join(data['keywords'])}\n"
                f"Hashtags: {hashtag_line}\n"
            )

    st.divider()
    st.download_button(
        "Download all platforms (.txt)",
        data="\n\n".join(all_text_export),
        file_name="contentforge_all_platforms.txt",
        mime="text/plain",
        use_container_width=True,
    )
else:
    st.info("Fill in the form above and generate to see your content here.")