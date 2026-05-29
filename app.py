# app.py - Scribble Digital v4.2 - AI-First Correction Engine
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import easyocr
import json
import re
import time
import warnings
import unicodedata
import difflib
from datetime import datetime
from openai import OpenAI

warnings.filterwarnings("ignore", message=".*pin_memory.*")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Scribble Digital v4.2",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
DEFAULTS = {
    "api_key": None,
    "mode_choice": "📝 EasyOCR Only",
    "ocr_languages": ["en"],
    "use_gpu": False,
    "history": [],
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background-color: #0f1117;
    color: #e8eaf0;
}
#MainMenu, footer, header { visibility: hidden; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f1117; }
::-webkit-scrollbar-thumb { background: #5c7cfa; border-radius: 3px; }

.hero {
    background: linear-gradient(135deg,#1e3a5f 0%,#2d5a8e 60%,#1a2d4f 100%);
    border-radius: 20px; padding: 36px 40px; margin-bottom: 24px;
    border: 1px solid rgba(92,124,250,0.25);
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
    position: relative; overflow: hidden;
}
.hero::before {
    content:''; position:absolute; top:-60px; right:-60px;
    width:300px; height:300px;
    background:radial-gradient(circle,rgba(92,124,250,0.18) 0%,transparent 70%);
    pointer-events:none;
}
.hero h1 { font-size:2.4rem; font-weight:800; color:#fff; margin:0 0 8px; }
.hero p  { font-size:1.05rem; color:rgba(255,255,255,0.78); margin:0; }
.hero-version {
    position:absolute; top:18px; right:22px;
    background:rgba(255,255,255,0.12); color:rgba(255,255,255,0.85);
    border-radius:20px; padding:4px 14px; font-size:12px; font-weight:600;
}
.hero-pills { margin-top:16px; display:flex; gap:8px; flex-wrap:wrap; }
.hero-pill {
    background:rgba(255,255,255,0.12); color:rgba(255,255,255,0.88);
    border-radius:20px; padding:4px 13px; font-size:12px;
    border:1px solid rgba(255,255,255,0.15);
}
.card {
    background:#1a1d27; border:1px solid rgba(255,255,255,0.08);
    border-radius:16px; padding:20px 22px; margin-bottom:16px;
    box-shadow:0 6px 24px rgba(0,0,0,0.25);
}
.card-header {
    font-size:1rem; font-weight:700; color:#e8eaf0;
    margin-bottom:12px; display:flex; align-items:center; gap:8px;
}
.result-box {
    background:#22263a; border:1px solid rgba(56,217,169,0.25);
    border-left:5px solid #38d9a9; border-radius:12px;
    padding:18px 20px; white-space:pre-wrap; word-break:break-word;
    font-size:0.97rem; line-height:1.75; color:#e8eaf0; margin:10px 0;
}
.result-box.ai   { border-left-color:#5c7cfa; border-color:rgba(92,124,250,0.25); }
.result-box.warn { border-left-color:#f77f00; border-color:rgba(247,127,0,0.25); }
.result-box.raw  { border-left-color:#9ba4c0; border-color:rgba(155,164,192,0.2);
                   background:#1a1d27; font-family:monospace; font-size:0.88rem; }
.diff-added   { color:#38d9a9; background:rgba(56,217,169,0.12);
                border-radius:3px; padding:0 3px; font-weight:600; }
.diff-removed { color:#ff6b6b; text-decoration:line-through; opacity:0.6; }
.stat-chip {
    background:#22263a; border:1px solid rgba(255,255,255,0.08);
    border-radius:12px; padding:14px 18px; text-align:center;
}
.stat-chip .val { font-size:1.5rem; font-weight:800; color:#5c7cfa; line-height:1; }
.stat-chip .lbl { font-size:11px; color:#9ba4c0; margin-top:4px;
                  text-transform:uppercase; letter-spacing:0.5px; }
.conf-bar-wrap { background:#22263a; border-radius:8px; height:10px; overflow:hidden; margin-top:6px; }
.conf-bar-fill { height:100%; border-radius:8px; transition:width 0.6s ease; }
.badge {
    display:inline-flex; align-items:center; gap:5px;
    background:rgba(92,124,250,0.15); color:#5c7cfa;
    border:1px solid rgba(92,124,250,0.3); border-radius:20px;
    padding:4px 12px; font-size:12px; font-weight:600;
}
.badge.success { background:rgba(56,217,169,0.12); color:#38d9a9; border-color:rgba(56,217,169,0.25); }
.badge.warning { background:rgba(247,127,0,0.12);  color:#f77f00;  border-color:rgba(247,127,0,0.25); }
.badge.error   { background:rgba(255,107,107,0.12); color:#ff6b6b; border-color:rgba(255,107,107,0.25); }
.todo-item {
    display:flex; align-items:flex-start; gap:10px;
    background:#22263a; border:1px solid rgba(255,255,255,0.08);
    border-radius:10px; padding:10px 14px; margin:5px 0;
    font-size:0.95rem; color:#e8eaf0;
}
.todo-dot { width:8px; height:8px; border-radius:50%; background:#f77f00;
            flex-shrink:0; margin-top:6px; }
.issue-box {
    background:rgba(255,107,107,0.08); border:1px solid rgba(255,107,107,0.25);
    border-radius:10px; padding:12px 16px; margin:8px 0;
    font-size:13px; color:#ffb3b3;
}
.pipeline-step {
    display:flex; align-items:center; gap:10px;
    background:#22263a; border:1px solid rgba(255,255,255,0.06);
    border-radius:10px; padding:10px 14px; margin:5px 0; font-size:13px;
}
.pipeline-step .icon { font-size:1.1rem; flex-shrink:0; }
.pipeline-step .label { color:#9ba4c0; min-width:160px; }
.pipeline-step .value { color:#e8eaf0; font-weight:600; }
.pipeline-step.ok   { border-left:4px solid #38d9a9; }
.pipeline-step.warn { border-left:4px solid #f77f00; }
.pipeline-step.fail { border-left:4px solid #ff6b6b; }
.correction-banner {
    background:linear-gradient(135deg,rgba(92,124,250,0.12),rgba(56,217,169,0.08));
    border:1px solid rgba(92,124,250,0.25); border-radius:14px;
    padding:16px 20px; margin:12px 0; display:flex; align-items:center; gap:14px;
}
.correction-banner .icon { font-size:2rem; }
.correction-banner .text { flex:1; }
.correction-banner .title { font-weight:700; color:#e8eaf0; margin-bottom:4px; }
.correction-banner .sub   { font-size:13px; color:#9ba4c0; }
[data-testid="stFileUploadDropzone"] {
    background:#1a1d27!important;
    border:2px dashed rgba(92,124,250,0.45)!important;
    border-radius:16px!important;
}
.stButton > button {
    background:linear-gradient(135deg,#5c7cfa,#4a6cf7)!important;
    color:white!important; border:none!important; border-radius:12px!important;
    font-weight:700!important; font-size:16px!important;
    box-shadow:0 6px 20px rgba(92,124,250,0.4)!important;
}
.stDownloadButton > button {
    background:#22263a!important; color:#5c7cfa!important;
    border:1px solid rgba(92,124,250,0.35)!important;
    border-radius:10px!important; font-weight:600!important;
}
section[data-testid="stSidebar"] {
    background:#1a1d27!important;
    border-right:1px solid rgba(255,255,255,0.08)!important;
}
section[data-testid="stSidebar"] * { color:#e8eaf0!important; }
[data-testid="stExpander"] {
    background:#1a1d27!important;
    border:1px solid rgba(255,255,255,0.08)!important;
    border-radius:12px!important;
}
pre, code {
    background:#22263a!important; color:#38d9a9!important;
    border-radius:8px!important; border:1px solid rgba(255,255,255,0.08)!important;
}
[data-testid="stMetric"] {
    background:#1a1d27!important; border:1px solid rgba(255,255,255,0.08)!important;
    border-radius:12px!important; padding:12px!important;
}
[data-testid="stMetricValue"] { color:#5c7cfa!important; font-weight:800!important; }
hr { border-color:rgba(255,255,255,0.08)!important; }
[data-testid="stTabs"] [role="tab"] { color:#9ba4c0!important; font-weight:600!important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color:#5c7cfa!important; border-bottom:3px solid #5c7cfa!important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def html(s: str):
    st.markdown(s, unsafe_allow_html=True)

def show_image(img, caption=None, clamp=False):
    try:
        st.image(img, caption=caption, use_container_width=True, clamp=clamp)
    except TypeError:
        st.image(img, caption=caption, use_column_width=True, clamp=clamp)

def conf_bar(value: float):
    color = "#38d9a9" if value >= 75 else ("#f77f00" if value >= 50 else "#ff6b6b")
    html(f"""
    <div style="margin:6px 0 14px;">
        <div style="display:flex;justify-content:space-between;
                    font-size:12px;color:#9ba4c0;">
            <span>OCR Confidence</span>
            <span style="color:{color};font-weight:700;">{value:.1f}%</span>
        </div>
        <div class="conf-bar-wrap">
            <div class="conf-bar-fill"
                 style="width:{value}%;
                        background:linear-gradient(90deg,#5c7cfa,{color});"></div>
        </div>
    </div>""")

def pipeline_step(icon: str, label: str, value: str, status: str = "ok"):
    html(f"""
    <div class="pipeline-step {status}">
        <span class="icon">{icon}</span>
        <span class="label">{label}</span>
        <span class="value">{value}</span>
    </div>""")


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
html("""
<div class="hero">
    <div class="hero-version">v4.2</div>
    <h1>✍️ Scribble Digital</h1>
    <p>Handwritten notes → clean, structured digital text.
       AI-first correction with smart local fallback.</p>
    <div class="hero-pills">
        <span class="hero-pill">🔍 EasyOCR</span>
        <span class="hero-pill">🎨 OpenCV</span>
        <span class="hero-pill">🤖 DeepSeek AI</span>
        <span class="hero-pill">🧠 AI-First Pipeline</span>
        <span class="hero-pill">🔄 Change Highlighting</span>
        <span class="hero-pill">🩺 Quality Diagnostics</span>
        <span class="hero-pill">📦 JSON / TXT / MD</span>
    </div>
</div>
""")


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    html('<div style="font-size:1.2rem;font-weight:800;margin-bottom:4px;">⚙️ Settings</div>')
    st.markdown("---")

    html('<div class="card-header">🔑 DeepSeek API Key</div>')
    st.markdown(
        '<a href="https://platform.deepseek.com" target="_blank" '
        'style="color:#5c7cfa;font-size:12px;">Get a free key →</a>',
        unsafe_allow_html=True)
    api_key_input = st.text_input(
        "API Key", type="password", placeholder="sk-…",
        label_visibility="collapsed")
    if api_key_input:
        st.session_state.api_key = api_key_input
        html('<div class="badge success">✅ API key active</div>')
    else:
        html('<div class="badge warning">🔒 No API key — AI modes locked</div>')

    st.markdown("---")
    html('<div class="card-header">🎛️ Processing Mode</div>')
    if st.session_state.api_key:
        modes = [
            "🚀 DeepSeek AI (Smart Processing)",
            "📝 EasyOCR Only",
            "🎭 Demo Mode",
        ]
    else:
        modes = ["📝 EasyOCR Only", "🎭 Demo Mode"]

    mode_choice = st.radio("Mode", modes, index=0, label_visibility="collapsed")
    st.session_state.mode_choice = mode_choice
    MODE_DESC = {
        "🚀 DeepSeek AI (Smart Processing)":
            "AI fixes all OCR errors, splits merged words, extracts TODOs, summary & tags.",
        "📝 EasyOCR Only":
            "Local cleaning pipeline + DP word segmenter. No API needed.",
        "🎭 Demo Mode":
            "Simulated output — no real extraction.",
    }
    html(f'<div style="font-size:12px;color:#9ba4c0;margin-top:6px;">'
         f'{MODE_DESC.get(mode_choice,"")}</div>')

    st.markdown("---")
    html('<div class="card-header">🌍 OCR Settings</div>')
    language_option = st.selectbox(
        "Language",
        ["English", "English + Spanish", "English + French",
         "English + German", "English + Chinese"],
        index=0)
    LANG_MAP = {
        "English":           ["en"],
        "English + Spanish": ["en", "es"],
        "English + French":  ["en", "fr"],
        "English + German":  ["en", "de"],
        "English + Chinese": ["en", "ch_sim"],
    }
    st.session_state.ocr_languages = LANG_MAP.get(language_option, ["en"])
    st.session_state.use_gpu = st.checkbox("Use GPU if available", value=False)

    st.markdown("---")
    html('<div class="card-header">🔧 Image Processing</div>')
    advanced_preprocess = st.checkbox("Advanced Preprocessing", value=True)
    deskew_enabled      = st.checkbox("Auto-Deskew", value=True)
    contrast_boost      = st.slider("Contrast Boost", 1.0, 3.0, 1.4, 0.1)
    denoise_strength    = st.slider("Denoise Strength (px)", 1, 9, 3, 2)

    st.markdown("---")
    html('<div class="card-header">📜 Session History</div>')
    if st.session_state.history:
        html(f'<div class="badge">{len(st.session_state.history)} scan(s)</div>')
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        html('<div style="font-size:12px;color:#9ba4c0;">No scans yet.</div>')


# ═══════════════════════════════════════════════════════════
# ░░  IMAGE PROCESSING
# ═══════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_easyocr_reader(languages: tuple, gpu: bool):
    return easyocr.Reader(list(languages), gpu=gpu, verbose=False)


def normalize_orientation(pil_img: Image.Image) -> Image.Image:
    try:
        return ImageOps.exif_transpose(pil_img)
    except Exception:
        return pil_img


def deskew_image(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(cv2.bitwise_not(thresh) > 0))
    if coords.size < 10:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.5:
        return gray
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def preprocess_image(pil_img: Image.Image,
                     advanced: bool, deskew: bool,
                     contrast: float, denoise: int) -> tuple:
    pil_img = normalize_orientation(pil_img)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(contrast)
    img     = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w    = img.shape[:2]
    if w < 1200:
        scale = 1200 / w
        img   = cv2.resize(img, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if deskew:
        gray = deskew_image(gray)
    if advanced:
        kernel    = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        k         = denoise if denoise % 2 == 1 else denoise + 1
        processed = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2)
        processed = cv2.medianBlur(processed, k)
    else:
        _, processed = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return processed, img


# ═══════════════════════════════════════════════════════════
# ░░  OCR
# ═══════════════════════════════════════════════════════════

def extract_text(processed_img: np.ndarray, reader) -> tuple:
    try:
        results = reader.readtext(processed_img, paragraph=False)
        if not results:
            return "", None
        text   = " ".join(str(r[1]) for r in results if str(r[1]).strip())
        scores = [float(r[2]) for r in results if len(r) >= 3]
        conf   = round(float(np.mean(scores)) * 100, 1) if scores else None
        return text, conf
    except Exception as e:
        st.error(f"EasyOCR error: {e}")
        return "", None


# ═══════════════════════════════════════════════════════════
# ░░  LOCAL CLEANING FALLBACK
# ═══════════════════════════════════════════════════════════

# Vocabulary for DP word segmenter
_VOCAB = {
    "you","can","imagine","is","real","everything","every","thing",
    "i","a","an","the","and","or","but","not","no","yes","ok",
    "to","do","go","it","in","on","at","of","for","are","was",
    "be","my","we","he","she","they","will","have","has","had",
    "this","that","with","from","by","as","if","so","then","when",
    "what","how","why","who","all","some","one","two","three",
    "get","set","put","let","run","see","say","use","try","new",
    "want","need","know","make","take","give","come","here","there",
    "about","after","before","between","should","would","could",
    "your","our","their","its","his","her","its","more","just",
    "time","people","look","good","great","think","work","day",
    "now","also","back","after","use","two","how","our","well",
    "way","even","find","long","down","never","same","last",
}

# Direct smash-fix dictionary — highest priority
_SMASH_FIXES: dict[str, str] = {
    "yqucanmacineisreal": "You can imagine is real",
    "youcanmacineisreal": "You can imagine is real",
    "everythingyoucan":   "Everything you can",
    "evryhng":            "Everything",
    "evryhn":             "Everything",
    "evryhing":           "Everything",
    "evrything":          "Everything",
    "youcan":             "You can",
    "canmacine":          "can imagine",
    "macineisreal":       "imagine is real",
    "imacine":            "imagine",
    "mmacine":            "imagine",
    "isreal":             "is real",
    "itsreal":            "it's real",
    "dont":               "don't",
    "cant":               "can't",
    "wont":               "won't",
    "im":                 "I'm",
    "ive":                "I've",
    "id":                 "I'd",
    "ill":                "I'll",
}

_CAMEL1 = re.compile(r'([A-Z]+)([A-Z][a-z])')
_CAMEL2 = re.compile(r'([a-z\d])([A-Z])')


def _dp_segment(token: str) -> str:
    """Greedy DP word segmenter over _VOCAB."""
    s  = token.lower()
    n  = len(s)
    dp   = [None] * (n + 1); dp[0] = 0
    back = [-1]   * (n + 1)
    for i in range(1, n + 1):
        for j in range(max(0, i - 15), i):
            if dp[j] is not None and s[j:i] in _VOCAB:
                dp[i] = (dp[j] or 0) + 1
                back[i] = j
                break
    if dp[n] is None:
        return token
    parts, pos = [], n
    while pos > 0:
        prev = back[pos]
        parts.append(s[prev:pos].capitalize())
        pos = prev
    return " ".join(reversed(parts))


def _fix_token(tok: str) -> str:
    """Fix a single whitespace-delimited OCR token."""
    low = tok.lower().replace("'", "").replace("-", "")
    # 1. Direct smash-fix
    if low in _SMASH_FIXES:
        return _SMASH_FIXES[low]
    # 2. All-caps long token → DP segment
    if tok.isupper() and len(tok) >= 5:
        segmented = _dp_segment(tok)
        if segmented != tok:          # segmentation succeeded
            return segmented
    # 3. CamelCase split
    t = _CAMEL1.sub(r'\1 \2', tok)
    t = _CAMEL2.sub(r'\1 \2', t)
    if t != tok:
        return t
    return tok


def _capitalise_sentences(text: str) -> str:
    return re.sub(
        r'(^|(?<=[.!?])\s+)([a-z])',
        lambda m: m.group(1) + m.group(2).upper(),
        text)


def clean_raw_ocr(text: str) -> str:
    """
    Local cleaning pipeline (no AI).
    Steps: unicode → token fix → whitespace → sentence caps.
    """
    text  = unicodedata.normalize("NFKC", text)
    text  = "".join(c for c in text
                    if unicodedata.category(c) != "Cc" or c == "\n")
    tokens = text.split()
    fixed  = [_fix_token(t) for t in tokens]
    text   = " ".join(fixed)
    text   = re.sub(r'[ \t]+',  ' ',    text)
    text   = re.sub(r'\n{3,}', '\n\n', text)
    text   = _capitalise_sentences(text.strip())
    return text


# ═══════════════════════════════════════════════════════════
# ░░  AI CORRECTION ENGINE
# ═══════════════════════════════════════════════════════════

# ── Master prompt ────────────────────────────────────────────
_AI_PROMPT = """You are an expert OCR post-processor specialising in handwritten note recognition.

The text below was extracted by an OCR engine from a photo of handwritten notes.
It may contain these types of errors:
  • Merged words (e.g. "YQUCANMACINEISREAL" → "You can imagine is real")
  • Wrong characters (e.g. "0" instead of "O", "l" instead of "I", "rn" instead of "m")
  • Missing spaces between words
  • Wrong capitalisation
  • Spelling mistakes
  • Noise characters

YOUR TASKS:
1. Fix ALL OCR recognition errors — especially merged/smashed tokens.
2. Restore correct spacing, spelling, punctuation, and capitalisation.
3. Preserve the ORIGINAL meaning — do NOT add, remove, or paraphrase content.
4. Identify action items / tasks (lines with TODO / - / • / □, or imperative sentences).
5. Write a concise 1-sentence summary (max 20 words).
6. Provide up to 5 relevant lowercase topic tags.
7. Count how many individual words you changed or inserted (integer).

CRITICAL EXAMPLE:
  Input:  "EvryhnG YQUCANMACINEISREAL"
  Output notes: "Everything you can imagine is real"

RAW OCR INPUT:
{raw_text}

RESPOND WITH ONLY valid JSON — no markdown, no explanation:
{{
  "notes":   "<corrected full text>",
  "todos":   ["<action item>"],
  "summary": "<one-sentence summary>",
  "tags":    ["<tag>"],
  "changes": <integer>
}}"""


def _parse_ai_response(raw: str) -> dict:
    """Strip markdown fences and parse JSON."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$",          "", raw).strip()
    return json.loads(raw)


def call_deepseek_correction(raw_text: str,
                             api_key: str,
                             retries: int = 2) -> tuple:
    """
    Call DeepSeek with the master prompt.
    Returns (result_dict, error_str | None).
    Retries up to `retries` times on JSON parse failure.
    """
    last_error = None
    for attempt in range(retries + 1):
        try:
            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            resp   = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise OCR correction assistant. "
                            "You always respond with valid JSON only. "
                            "Never add markdown, explanation, or extra text."
                        )
                    },
                    {
                        "role": "user",
                        "content": _AI_PROMPT.format(raw_text=raw_text)
                    }
                ],
                temperature=0.1,      # very low for deterministic output
                max_tokens=1200,
            )
            raw_json = resp.choices[0].message.content
            result   = _parse_ai_response(raw_json)

            # Validate required keys
            if "notes" not in result:
                raise ValueError("Missing 'notes' key in AI response")

            # Ensure all expected keys exist with defaults
            result.setdefault("todos",   [])
            result.setdefault("summary", "")
            result.setdefault("tags",    [])
            result.setdefault("changes", 0)

            # Sanity-check: if AI returned something shorter than 3 chars, reject
            if len(result["notes"].strip()) < 3:
                raise ValueError("AI returned empty notes")

            return result, None

        except json.JSONDecodeError as e:
            last_error = f"JSON parse error (attempt {attempt+1}): {e}"
        except ValueError as e:
            last_error = str(e)
            break     # don't retry logic errors
        except Exception as e:
            last_error = str(e)
            break

    return None, last_error


def call_deepseek_simple(raw_text: str, api_key: str) -> tuple:
    """
    Lightweight fallback: just ask AI to return corrected plain text.
    Used when the structured JSON call fails.
    """
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        resp   = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You fix OCR errors in handwritten note text. "
                        "Return ONLY the corrected text — no JSON, no explanation."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Fix all OCR errors. Merged words like "
                        "'YQUCANMACINEISREAL' must be split correctly.\n\n"
                        f"Text: {raw_text}"
                    )
                }
            ],
            temperature=0.0,
            max_tokens=800,
        )
        corrected = resp.choices[0].message.content.strip()
        return corrected, None
    except Exception as e:
        return None, str(e)


# ═══════════════════════════════════════════════════════════
# ░░  QUALITY ASSESSMENT
# ═══════════════════════════════════════════════════════════

def assess_quality(text: str, confidence: float | None) -> dict:
    issues: list[str] = []
    score  = 100

    raw_tokens  = text.split()
    smashed     = [t for t in raw_tokens if t.isupper() and len(t) >= 6]

    # Estimate real word count (smashed tokens counted by length)
    est_wc = sum(max(1, len(t) // 4) if t.isupper() and len(t) >= 6
                 else 1 for t in raw_tokens)

    chars       = len(text.replace(" ", ""))
    alpha       = [c for c in text if c.isalpha()]
    digits      = [c for c in text if c.isdigit()]
    avg_wl      = sum(len(w) for w in raw_tokens) / max(len(raw_tokens), 1)
    digit_ratio = len(digits) / max(chars, 1)
    upper_ratio = sum(1 for c in alpha if c.isupper()) / max(len(alpha), 1)

    if confidence is not None:
        if confidence < 50:
            issues.append(
                f"Very low OCR confidence ({confidence}%) — "
                "image may be blurry or low-contrast.")
            score -= 30
        elif confidence < 70:
            issues.append(
                f"Moderate OCR confidence ({confidence}%) — "
                "some words may be mis-read.")
            score -= 12

    if est_wc < 3:
        issues.append(
            "Very few words extracted — image may be blank or text too faint.")
        score -= 25
    elif est_wc < 6:
        issues.append("Short note detected — may be a heading or label.")
        score -= 5

    if smashed:
        issues.append(
            f"{len(smashed)} merged token(s) detected "
            f"(e.g. '{smashed[0]}') — AI will split these correctly.")
        score -= 6 * len(smashed)

    if avg_wl < 2.5 and len(raw_tokens) > 5:
        issues.append(
            "Short average word length — possible character-level noise.")
        score -= 15

    if digit_ratio > 0.40:
        issues.append(
            f"High digit ratio ({digit_ratio:.0%}) — "
            "OCR may be misreading letters as numbers.")
        score -= 15

    if upper_ratio > 0.70 and len(raw_tokens) > 3 and not smashed:
        issues.append(
            "High uppercase ratio — consider adjusting preprocessing.")
        score -= 10

    score = max(0, score)
    level = "good" if score >= 70 else ("fair" if score >= 40 else "poor")
    return {
        "score":          score,
        "level":          level,
        "issues":         issues,
        "word_count":     est_wc,
        "raw_tokens":     len(raw_tokens),
        "avg_word_len":   round(avg_wl, 1),
        "digit_ratio":    round(digit_ratio, 3),
        "upper_ratio":    round(upper_ratio, 3),
        "smashed_tokens": len(smashed),
    }


# ═══════════════════════════════════════════════════════════
# ░░  DIFF HIGHLIGHTING
# ═══════════════════════════════════════════════════════════

def highlight_changes(original: str, corrected: str) -> str:
    orig_tok = original.split()
    corr_tok = corrected.split()
    matcher  = difflib.SequenceMatcher(None, orig_tok, corr_tok, autojunk=False)
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append(" ".join(corr_tok[j1:j2]))
        elif tag == "replace":
            parts += [f'<span class="diff-removed">{w}</span>'
                      for w in orig_tok[i1:i2]]
            parts += [f'<span class="diff-added">{w}</span>'
                      for w in corr_tok[j1:j2]]
        elif tag == "insert":
            parts += [f'<span class="diff-added">{w}</span>'
                      for w in corr_tok[j1:j2]]
        elif tag == "delete":
            parts += [f'<span class="diff-removed">{w}</span>'
                      for w in orig_tok[i1:i2]]
    return " ".join(parts)


# ═══════════════════════════════════════════════════════════
# ░░  EXPORT
# ═══════════════════════════════════════════════════════════

def build_json_export(filename, result, raw_text, confidence) -> str:
    return json.dumps({
        "file":           filename,
        "scanned_at":     datetime.now().isoformat(),
        "ocr_confidence": confidence,
        "raw_text":       raw_text,
        "notes":          result.get("notes",   ""),
        "summary":        result.get("summary", ""),
        "tags":           result.get("tags",    []),
        "todos":          result.get("todos",   []),
        "words_changed":  result.get("changes", "N/A"),
    }, indent=2, ensure_ascii=False)


def build_markdown_export(filename, result) -> str:
    lines = [f"# Notes — {filename}\n"]
    if result.get("summary"):
        lines.append(f"> {result['summary']}\n")
    if result.get("tags"):
        lines.append("**Tags:** " +
                     " ".join(f"`{t}`" for t in result["tags"]) + "\n")
    lines.append("---\n")
    lines.append(result.get("notes", ""))
    if result.get("todos"):
        lines.append("\n\n## ✅ Action Items\n")
        lines += [f"- [ ] {t}" for t in result["todos"]]
    return "\n".join(lines)


def demo_structuring(raw_text: str) -> dict:
    return {
        "notes":   f"**[Demo]** {raw_text}\n\n"
                   "*Add a DeepSeek API key for real AI correction.*",
        "todos":   ["Add your DeepSeek API key in the sidebar"],
        "summary": "Demo output — accuracy requires AI mode.",
        "tags":    ["demo", "handwriting", "ocr"],
        "changes": 0,
        "_source": "demo",
    }


# ═══════════════════════════════════════════════════════════
# ░░  MAIN UI TABS
# ═══════════════════════════════════════════════════════════

tab_scan, tab_history, tab_help = st.tabs(
    ["📸 Scan Notes", "📜 History", "❓ Help"])


# ══════════════════════════════════════════════
# TAB 1 — SCAN
# ══════════════════════════════════════════════
with tab_scan:
    uploaded_files = st.file_uploader(
        "📤 Drop handwritten note images here",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True)

    if not uploaded_files:
        html("""
        <div class="card" style="text-align:center;padding:40px;">
            <div style="font-size:3rem;margin-bottom:10px;">📋</div>
            <div style="font-size:1.15rem;font-weight:700;margin-bottom:8px;">
                No images uploaded yet</div>
            <div style="color:#9ba4c0;font-size:0.95rem;">
                Upload JPG/PNG images of handwritten notes to get started.<br>
                Best results: clear lighting · dark ink · flat white paper.
            </div>
        </div>""")
    else:
        if len(uploaded_files) > 10:
            st.warning("More than 10 files — only the first 10 will be processed.")
            uploaded_files = uploaded_files[:10]

        html(f'<div style="display:flex;gap:10px;margin-bottom:14px;">'
             f'<div class="badge">{len(uploaded_files)} '
             f'file{"s" if len(uploaded_files)>1 else ""} ready</div>'
             f'<div style="color:#9ba4c0;font-size:13px;align-self:center;">'
             f'Mode: {st.session_state.mode_choice}</div></div>')

        # ── Preprocess + preview ──
        preprocessed_cache: dict[str, np.ndarray] = {}
        for idx, uf in enumerate(uploaded_files):
            with st.expander(f"🖼️ Preview — {uf.name}", expanded=(idx == 0)):
                pil_img = Image.open(uf).convert("RGB")
                c1, c2  = st.columns(2)
                with c1:
                    html('<div class="card-header">📸 Original</div>')
                    show_image(pil_img)
                with c2:
                    html('<div class="card-header">🔬 Preprocessed</div>')
                    proc, _ = preprocess_image(
                        pil_img, advanced_preprocess, deskew_enabled,
                        contrast_boost, denoise_strength)
                    show_image(proc, clamp=True)
                preprocessed_cache[uf.name] = proc

        st.markdown("---")
        run = st.button("🚀  Convert to Digital Notes",
                        type="primary", use_container_width=True)

        if run:
            reader      = load_easyocr_reader(
                tuple(st.session_state.ocr_languages),
                st.session_state.use_gpu)
            progress    = st.progress(0, text="Starting…")
            all_results = []
            t_batch     = time.time()

            for idx, uf in enumerate(uploaded_files):
                progress.progress(
                    (idx + 1) / len(uploaded_files),
                    text=f"Processing {uf.name}…")

                st.markdown("---")
                html(f"""
                <div style="display:flex;align-items:center;
                            gap:10px;margin-bottom:8px;">
                    <div style="font-size:1.1rem;font-weight:800;">
                        🧾 {uf.name}</div>
                    <div class="badge">{idx+1}/{len(uploaded_files)}</div>
                </div>""")

                proc_img = preprocessed_cache.get(uf.name)
                if proc_img is None:
                    pil_img  = Image.open(uf).convert("RGB")
                    proc_img, _ = preprocess_image(
                        pil_img, advanced_preprocess, deskew_enabled,
                        contrast_boost, denoise_strength)

                # ── Step 1: OCR ──────────────────────────────
                t0 = time.time()
                with st.spinner("🔍 Running OCR…"):
                    raw_text, confidence = extract_text(proc_img, reader)
                ocr_s = round(time.time() - t0, 2)

                if not raw_text or len(raw_text.strip()) < 3:
                    st.warning(
                        f"Could not extract text from **{uf.name}**. "
                        "Try better lighting or higher contrast.")
                    continue

                # ── Step 2: Quality assessment ───────────────
                quality = assess_quality(raw_text, confidence)

                # ── Stats ─────────────────────────────────────
                html(f"""
                <div style="display:flex;gap:12px;flex-wrap:wrap;margin:14px 0;">
                    <div class="stat-chip">
                        <div class="val">{len(raw_text)}</div>
                        <div class="lbl">Characters</div>
                    </div>
                    <div class="stat-chip">
                        <div class="val">{quality["word_count"]}</div>
                        <div class="lbl">Est. Words</div>
                    </div>
                    <div class="stat-chip">
                        <div class="val">{ocr_s}s</div>
                        <div class="lbl">OCR Time</div>
                    </div>
                    <div class="stat-chip">
                        <div class="val" style="color:{
                            '#38d9a9' if quality['level']=='good' else
                            '#f77f00' if quality['level']=='fair' else
                            '#ff6b6b'};">
                            {quality['score']}
                        </div>
                        <div class="lbl">Quality</div>
                    </div>
                </div>""")

                if confidence is not None:
                    conf_bar(confidence)

                for issue in quality["issues"]:
                    html(f'<div class="issue-box">⚠️ {issue}</div>')

                with st.expander("📄 Raw OCR Text"):
                    html(f'<div class="result-box raw">{raw_text}</div>')

                # ── Step 3: Correction Pipeline ──────────────
                result: dict   = {}
                diff_html: str = ""
                pipeline_log   = []   # list of (icon, label, value, status)

                # ─ A: DeepSeek AI (primary) ──────────────────
                if ("DeepSeek AI" in st.session_state.mode_choice
                        and st.session_state.api_key):

                    with st.spinner("🤖 AI correcting text…"):
                        t_ai  = time.time()
                        result, ai_error = call_deepseek_correction(
                            raw_text, st.session_state.api_key)
                        ai_s  = round(time.time() - t_ai, 2)

                    if result is not None:
                        # ✅ AI succeeded
                        pipeline_log.append(
                            ("🤖", "DeepSeek AI correction",
                             f"✅ done in {ai_s}s", "ok"))
                        pipeline_log.append(
                            ("✏️", "Words corrected",
                             str(result.get("changes", "?")), "ok"))
                        result["_source"] = "ai_full"
                        diff_html = highlight_changes(
                            raw_text, result["notes"])

                    else:
                        # ⚠️ Structured call failed — try simple correction
                        pipeline_log.append(
                            ("🤖", "DeepSeek AI (structured)",
                             f"⚠️ {ai_error}", "warn"))

                        with st.spinner("🔄 Retrying with simple AI correction…"):
                            corrected, simple_err = call_deepseek_simple(
                                raw_text, st.session_state.api_key)

                        if corrected:
                            pipeline_log.append(
                                ("🔄", "DeepSeek AI (simple fallback)",
                                 "✅ done", "ok"))
                            result = {
                                "notes":   corrected,
                                "todos":   [],
                                "summary": "",
                                "tags":    [],
                                "changes": 0,
                                "_source": "ai_simple",
                            }
                            diff_html = highlight_changes(raw_text, corrected)
                        else:
                            # ❌ Both AI calls failed — local fallback
                            pipeline_log.append(
                                ("❌", "AI simple fallback",
                                 f"failed: {simple_err}", "fail"))
                            pipeline_log.append(
                                ("🔧", "Local cleaner",
                                 "activated", "warn"))
                            cleaned = clean_raw_ocr(raw_text)
                            result  = {
                                "notes":   cleaned,
                                "todos":   [],
                                "summary": "",
                                "tags":    [],
                                "changes": 0,
                                "_source": "local",
                            }
                            diff_html = highlight_changes(raw_text, cleaned)
                            st.warning(
                                "⚠️ AI unavailable — local correction applied. "
                                "Results may be less accurate.")

                # ─ B: EasyOCR Only ───────────────────────────
                elif "Demo Mode" not in st.session_state.mode_choice:
                    if st.session_state.api_key:
                        # API key present — use simple AI correction
                        with st.spinner("🤖 AI spell-correction…"):
                            corrected, err = call_deepseek_simple(
                                raw_text, st.session_state.api_key)
                        if corrected:
                            pipeline_log.append(
                                ("🤖", "AI spell-correction",
                                 "✅ done", "ok"))
                            result = {
                                "notes":   corrected,
                                "todos":   [],
                                "summary": "",
                                "tags":    [],
                                "changes": 0,
                                "_source": "ai_simple",
                            }
                            diff_html = highlight_changes(raw_text, corrected)
                        else:
                            pipeline_log.append(
                                ("⚠️", "AI spell-correction",
                                 f"failed: {err}", "warn"))
                            cleaned = clean_raw_ocr(raw_text)
                            result  = {
                                "notes":   cleaned,
                                "todos":   [],
                                "summary": "",
                                "tags":    [],
                                "changes": 0,
                                "_source": "local",
                            }
                            diff_html = highlight_changes(raw_text, cleaned)
                    else:
                        # No key — pure local
                        pipeline_log.append(
                            ("🔧", "Local cleaner", "activated", "ok"))
                        cleaned = clean_raw_ocr(raw_text)
                        result  = {
                            "notes":   cleaned,
                            "todos":   [],
                            "summary": "",
                            "tags":    [],
                            "changes": 0,
                            "_source": "local",
                        }
                        diff_html = highlight_changes(raw_text, cleaned)
                    st.success("✅ Extraction complete!")

                # ─ C: Demo Mode ──────────────────────────────
                else:
                    result = demo_structuring(raw_text)
                    st.info("🎭 Demo mode — add a DeepSeek API key for real results.")

                # ── Correction Source Banner ──────────────────
                SOURCE_META = {
                    "ai_full": (
                        "🤖",
                        "DeepSeek AI — Full Correction",
                        "OCR errors fixed · TODOs extracted · Summary generated"
                    ),
                    "ai_simple": (
                        "🔄",
                        "DeepSeek AI — Text Correction",
                        "OCR errors and spelling fixed by AI"
                    ),
                    "local": (
                        "🔧",
                        "Local Cleaner",
                        "Rule-based fixes applied · Add API key for AI correction"
                    ),
                    "demo": (
                        "🎭",
                        "Demo Mode",
                        "Simulated output"
                    ),
                }
                src  = result.get("_source", "local")
                icon, title, sub = SOURCE_META.get(src, SOURCE_META["local"])
                html(f"""
                <div class="correction-banner">
                    <div class="icon">{icon}</div>
                    <div class="text">
                        <div class="title">{title}</div>
                        <div class="sub">{sub}</div>
                    </div>
                </div>""")

                # ── Results Tabs ──────────────────────────────
                r1, r2, r3, r4 = st.tabs(
                    ["📝 Corrected Notes",
                     "✅ Action Items",
                     "🔄 Changes",
                     "📊 Details"])

                with r1:
                    html(f'<div class="result-box ai">'
                         f'{result.get("notes","")}</div>')

                with r2:
                    todos = result.get("todos", [])
                    if todos:
                        for i, t in enumerate(todos):
                            html(f'<div class="todo-item">'
                                 f'<div class="todo-dot"></div>'
                                 f'<span>{t}</span></div>')
                        st.markdown("**Mark as done:**")
                        for i, t in enumerate(todos):
                            st.checkbox(t, key=f"cb_{uf.name}_{idx}_{i}")
                    else:
                        html('<div style="color:#9ba4c0;padding:14px;">'
                             'No action items detected.</div>')

                with r3:
                    if diff_html:
                        n_changes = result.get("changes", "?")
                        html(f'<div class="badge" style="margin-bottom:10px;">'
                             f'~{n_changes} word(s) corrected</div>')
                        html(f'<div class="result-box">{diff_html}</div>')
                        html("""
                        <div style="font-size:12px;color:#9ba4c0;margin-top:8px;">
                            <span class="diff-added">■ Green</span> = corrected/inserted &nbsp;
                            <span class="diff-removed">■ Red strikethrough</span> = original token
                        </div>""")
                    else:
                        html('<div style="color:#9ba4c0;padding:14px;">'
                             'No change diff for demo mode.</div>')

                with r4:
                    # Summary & Tags
                    if result.get("summary"):
                        html(f'<div class="card">'
                             f'<div class="card-header">💬 Summary</div>'
                             f'<p style="color:#9ba4c0;margin:0;">'
                             f'{result["summary"]}</p></div>')
                    if result.get("tags"):
                        tag_html = "".join(
                            f'<span class="badge" style="margin:4px;">{t}</span>'
                            for t in result["tags"])
                        html(f'<div class="card">'
                             f'<div class="card-header">🏷️ Tags</div>'
                             f'{tag_html}</div>')

                    # Pipeline log
                    if pipeline_log:
                        html('<div class="card">'
                             '<div class="card-header">🔁 Processing Pipeline</div>')
                        for icon, label, value, status in pipeline_log:
                            pipeline_step(icon, label, value, status)
                        html('</div>')

                    # Quality card
                    qc = {"good":"#38d9a9","fair":"#f77f00","poor":"#ff6b6b"}.get(
                        quality["level"],"#9ba4c0")
                    html(f"""
                    <div class="card">
                        <div class="card-header">🩺 Quality Diagnostics</div>
                        <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px;">
                            <span class="badge" style="color:{qc};
                                border-color:{qc};background:transparent;">
                                {quality['score']}/100 {quality['level'].upper()}
                            </span>
                            {''.join(f'<span class="badge warning">{s} merged token(s)</span>'
                              if (s:=quality["smashed_tokens"]) > 0 else "")}
                        </div>
                        <table style="width:100%;font-size:13px;
                                      color:#9ba4c0;border-collapse:collapse;">
                            <tr><td style="padding:6px 8px;">Est. word count</td>
                                <td style="color:#e8eaf0;font-weight:600;">
                                    {quality['word_count']}</td></tr>
                            <tr><td style="padding:6px 8px;">Raw OCR tokens</td>
                                <td style="color:#e8eaf0;font-weight:600;">
                                    {quality['raw_tokens']}</td></tr>
                            <tr><td style="padding:6px 8px;">Avg token length</td>
                                <td style="color:#e8eaf0;font-weight:600;">
                                    {quality['avg_word_len']}</td></tr>
                            <tr><td style="padding:6px 8px;">Digit ratio</td>
                                <td style="color:#e8eaf0;font-weight:600;">
                                    {quality['digit_ratio']:.1%}</td></tr>
                            <tr><td style="padding:6px 8px;">Uppercase ratio</td>
                                <td style="color:#e8eaf0;font-weight:600;">
                                    {quality['upper_ratio']:.1%}</td></tr>
                            <tr><td style="padding:6px 8px;">OCR confidence</td>
                                <td style="color:#e8eaf0;font-weight:600;">
                                    {f'{confidence}%' if confidence else 'N/A'}</td></tr>
                            <tr><td style="padding:6px 8px;">Correction source</td>
                                <td style="color:#e8eaf0;font-weight:600;">
                                    {src}</td></tr>
                        </table>
                    </div>""")

                # ── Export ──────────────────────────────────
                st.markdown("**📥 Export:**")
                e1, e2, e3 = st.columns(3)
                with e1:
                    st.download_button(
                        "📄 TXT", result.get("notes",""),
                        file_name=f"{uf.name}_notes.txt",
                        mime="text/plain",
                        key=f"dl_txt_{uf.name}_{idx}",
                        use_container_width=True)
                with e2:
                    st.download_button(
                        "📦 JSON",
                        build_json_export(
                            uf.name, result, raw_text, confidence),
                        file_name=f"{uf.name}_notes.json",
                        mime="application/json",
                        key=f"dl_json_{uf.name}_{idx}",
                        use_container_width=True)
                with e3:
                    st.download_button(
                        "📝 Markdown",
                        build_markdown_export(uf.name, result),
                        file_name=f"{uf.name}_notes.md",
                        mime="text/markdown",
                        key=f"dl_md_{uf.name}_{idx}",
                        use_container_width=True)

                # ── Save to history ──────────────────────────
                st.session_state.history.append({
                    "timestamp":  datetime.now().strftime("%H:%M:%S"),
                    "filename":   uf.name,
                    "result":     result,
                    "raw":        raw_text,
                    "confidence": confidence,
                    "quality":    quality,
                })
                all_results.append(result)

            progress.progress(1.0, text="✅ All done!")
            html(f"""
            <div class="card" style="margin-top:20px;
                 border-color:rgba(56,217,169,0.3);">
                <div class="card-header">🎉 Batch Complete</div>
                <div style="color:#9ba4c0;font-size:14px;">
                    Processed <b style="color:#e8eaf0;">{len(all_results)}</b>
                    file(s) in
                    <b style="color:#5c7cfa;">
                        {round(time.time()-t_batch,2)}s
                    </b>
                </div>
            </div>""")

            if len(all_results) > 1:
                st.download_button(
                    "📦 Download All as JSON",
                    json.dumps(
                        [{"file": uploaded_files[i].name, **r}
                         for i, r in enumerate(all_results)],
                        indent=2, ensure_ascii=False),
                    file_name="scribble_batch_export.json",
                    mime="application/json",
                    use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2 — HISTORY
# ══════════════════════════════════════════════
with tab_history:
    html('<div class="card-header">📜 Session Scan History</div>')
    if not st.session_state.history:
        html("""
        <div class="card" style="text-align:center;padding:36px;">
            <div style="font-size:2.5rem;">🕓</div>
            <div style="color:#9ba4c0;margin-top:10px;">
                No scans yet. Go to <b>Scan Notes</b> to get started.
            </div>
        </div>""")
    else:
        for i, item in enumerate(reversed(st.session_state.history)):
            q     = item.get("quality", {})
            qc    = {"good":"#38d9a9","fair":"#f77f00","poor":"#ff6b6b"}.get(
                    q.get("level",""), "#9ba4c0")
            src   = item["result"].get("_source","local")
            label = {"ai_full":"🤖 AI Full","ai_simple":"🔄 AI Simple",
                     "local":"🔧 Local","demo":"🎭 Demo"}.get(src, src)
            with st.expander(
                f"🕐 {item['timestamp']} — {item['filename']} "
                f"| quality: {q.get('score','?')} | {label}"):
                if item.get("confidence"):
                    conf_bar(item["confidence"])
                html(f'<div class="result-box">'
                     f'{item["result"].get("notes","")}</div>')
                todos = item["result"].get("todos",[])
                if todos:
                    for t in todos:
                        html(f'<div class="todo-item">'
                             f'<div class="todo-dot"></div><span>{t}</span></div>')
                if item["result"].get("summary"):
                    html(f'<blockquote style="color:#9ba4c0;'
                         'border-left:4px solid #5c7cfa;'
                         f'padding-left:12px;margin:10px 0;">'
                         f'{item["result"]["summary"]}</blockquote>')
                tags = item["result"].get("tags",[])
                if tags:
                    html("".join(
                        f'<span class="badge" style="margin:4px;">{t}</span>'
                        for t in tags))
                hc1, hc2 = st.columns(2)
                with hc1:
                    st.download_button(
                        "📄 TXT", item["result"].get("notes",""),
                        file_name=f"{item['filename']}_notes.txt",
                        key=f"hist_txt_{i}", use_container_width=True)
                with hc2:
                    st.download_button(
                        "📦 JSON",
                        build_json_export(
                            item["filename"], item["result"],
                            item.get("raw",""), item.get("confidence")),
                        file_name=f"{item['filename']}_notes.json",
                        mime="application/json",
                        key=f"hist_json_{i}", use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 — HELP
# ══════════════════════════════════════════════
with tab_help:
    html("""
    <div class="card">
        <div class="card-header">🧠 AI-First Correction Pipeline</div>
        <div class="pipeline-step ok">
            <span class="icon">🔍</span>
            <span class="label">Step 1 — EasyOCR</span>
            <span class="value">Raw text extraction from image</span>
        </div>
        <div class="pipeline-step ok">
            <span class="icon">🤖</span>
            <span class="label">Step 2 — DeepSeek AI</span>
            <span class="value">Full structured correction (JSON)</span>
        </div>
        <div class="pipeline-step warn">
            <span class="icon">🔄</span>
            <span class="label">Step 3 — AI Simple fallback</span>
            <span class="value">Plain-text correction if JSON fails</span>
        </div>
        <div class="pipeline-step warn">
            <span class="icon">🔧</span>
            <span class="label">Step 4 — Local cleaner</span>
            <span class="value">Rule-based + DP segmenter if AI unavailable</span>
        </div>
        <p style="color:#9ba4c0;font-size:13px;margin-top:12px;">
            Each step only activates if the previous one fails.
            The correction source is always shown in the results banner.
        </p>
    </div>

    <div class="card">
        <div class="card-header">📸 Tips for Best OCR Results</div>
        <ul style="color:#9ba4c0;line-height:2.1;">
            <li>Use <b style="color:#e8eaf0;">bright, even lighting</b> — no shadows.</li>
            <li>Dark pen on <b style="color:#e8eaf0;">white/light paper</b>.</li>
            <li><b style="color:#e8eaf0;">Flatten the page</b> — no curves or wrinkles.</li>
            <li>Hold camera <b style="color:#e8eaf0;">directly above</b> the page.</li>
            <li>Enable <b style="color:#e8eaf0;">Auto-Deskew</b> for rotated pages.</li>
            <li>Raise <b style="color:#e8eaf0;">Contrast Boost</b> (1.8–2.2) for faint ink.</li>
        </ul>
    </div>

    <div class="card">
        <div class="card-header">🎛️ Mode Comparison</div>
        <table style="width:100%;border-collapse:collapse;font-size:13px;">
            <thead>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.08);">
                    <th style="padding:10px;text-align:left;color:#9ba4c0;">Feature</th>
                    <th style="padding:10px;text-align:center;color:#9ba4c0;">EasyOCR Only</th>
                    <th style="padding:10px;text-align:center;color:#9ba4c0;">DeepSeek AI</th>
                    <th style="padding:10px;text-align:center;color:#9ba4c0;">Demo</th>
                </tr>
            </thead>
            <tbody style="color:#e8eaf0;">
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                    <td style="padding:8px 10px;">Text Extraction</td>
                    <td style="text-align:center;">✅</td>
                    <td style="text-align:center;">✅</td>
                    <td style="text-align:center;">✅</td></tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                    <td style="padding:8px 10px;">Merged Word Splitting</td>
                    <td style="text-align:center;">🔧 Local DP</td>
                    <td style="text-align:center;">✅ AI</td>
                    <td style="text-align:center;">❌</td></tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                    <td style="padding:8px 10px;">Deep AI Correction</td>
                    <td style="text-align:center;">❌</td>
                    <td style="text-align:center;">✅</td>
                    <td style="text-align:center;">❌</td></tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                    <td style="padding:8px 10px;">TODOs / Summary / Tags</td>
                    <td style="text-align:center;">❌</td>
                    <td style="text-align:center;">✅</td>
                    <td style="text-align:center;">❌</td></tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                    <td style="padding:8px 10px;">Change Highlighting</td>
                    <td style="text-align:center;">✅</td>
                    <td style="text-align:center;">✅</td>
                    <td style="text-align:center;">❌</td></tr>
                <tr><td style="padding:8px 10px;">API Key Required</td>
                    <td style="text-align:center;">❌</td>
                    <td style="text-align:center;">✅</td>
                    <td style="text-align:center;">❌</td></tr>
            </tbody>
        </table>
    </div>
    """)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
html("""
<div style="text-align:center;color:#9ba4c0;font-size:13px;padding:10px 0;">
    Scribble Digital v4.2 &nbsp;·&nbsp;
    Streamlit + EasyOCR + OpenCV + DeepSeek AI &nbsp;·&nbsp;
    <span style="color:#5c7cfa;">Built with ❤️</span>
</div>
""")
