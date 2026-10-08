"""
styles.py — Global CSS theme, cards, bubbles, and reusable UI helpers.
Import inject_theme() at the top of every page.
"""
import streamlit as st


# =====================================================================
# GLOBAL THEME
# =====================================================================
def inject_theme():
    """Inject the global CSS theme. Call once per page."""
    st.markdown(
        """
        <style>
        /* ---------- Fonts ---------- */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* ---------- App background ---------- */
        .stApp {
            background: linear-gradient(135deg, #324457 0%, #e8ecf3 100%);
        }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
            border-right: 1px solid #334155;
        }
        section[data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }
        section[data-testid="stSidebar"] .stRadio > label {
            color: #94a3b8 !important;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.08em;
        }
        section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
            padding: 0.55rem 0.75rem;
            border-radius: 10px;
            transition: all 0.2s ease;
            cursor: pointer;
        }
        section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {
            background: rgba(96, 165, 250, 0.15);
        }

        /* ---------- Headings ---------- */
        h1, h2, h3 {
            color: #0f172a;
            font-weight: 700;
        }
        h1 { font-size: 2.1rem !important; }

        /* ---------- KPI cards ---------- */
        div[data-testid="stMetric"] {
            background: #176da6;
            border-radius: 14px;
            padding: 1rem 1.2rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06),
                        0 4px 12px rgba(15, 23, 42, 0.04);
            border-left: 4px solid #3b82f6;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(15, 23, 42, 0.08),
                        0 8px 20px rgba(15, 23, 42, 0.08);
        }
        div[data-testid="stMetricLabel"] {
            color: #64748b !important;
            font-weight: 600;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        div[data-testid="stMetricValue"] {
            color: #0f172a !important;
            font-weight: 700;
        }

        /* ---------- Buttons ---------- */
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
            color: #ffffff;
            border: none;
            border-radius: 10px;
            padding: 0.55rem 1.1rem;
            font-weight: 600;
            transition: all 0.2s ease;
            box-shadow: 0 2px 6px rgba(59, 130, 246, 0.25);
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 14px rgba(59, 130, 246, 0.35);
            background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
        }
        .stDownloadButton > button {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: #fff;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            box-shadow: 0 2px 6px rgba(16, 185, 129, 0.25);
        }
        .stDownloadButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 14px rgba(16, 185, 129, 0.35);
        }

        /* ---------- Expander ---------- */
        details[data-testid="stExpander"] {
            background: #ffffff;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
            overflow: hidden;
        }
        details[data-testid="stExpander"] summary {
            padding: 0.85rem 1.1rem;
            font-weight: 600;
            color: #1e293b;
            background: #f8fafc;
            border-bottom: 1px solid #e2e8f0;
        }
        details[data-testid="stExpander"] summary:hover {
            background: #324457;
        }

        /* ---------- Tabs ---------- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            background: #ffffff;
            padding: 6px;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            color: #64748b;
            background: transparent;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
            color: #ffffff !important;
        }

        /* ---------- Dataframe ---------- */
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        }

        /* ---------- Alerts ---------- */
        div[data-testid="stAlert"] {
            border-radius: 12px;
            border-left-width: 4px;
        }

        /* ---------- Bubbles ---------- */
        .bubble {
            display: inline-block;
            padding: 0.6rem 1rem;
            border-radius: 18px;
            margin: 0.25rem 0;
            font-size: 0.92rem;
            max-width: 80%;
            line-height: 1.4;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
        }
        .bubble-user {
            background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
            color: #ffffff;
            align-self: flex-end;
            border-bottom-right-radius: 4px;
        }
        .bubble-bot {
            background: #f1f5f9;
            color: #0f172a;
            align-self: flex-start;
            border-bottom-left-radius: 4px;
        }
        .bubble-wrap {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
            margin-bottom: 0.75rem;
        }

        /* ---------- Gradient header banner ---------- */
        .hero {
            background: linear-gradient(135deg, #6366f1 0%, #3b82f6 50%, #06b6d4 100%);
            padding: 1.6rem 2rem;
            border-radius: 18px;
            color: #ffffff;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 24px rgba(99, 102, 241, 0.25);
        }
        .hero h1, .hero h2, .hero h3, .hero p {
            color: #ffffff !important;
            margin: 0;
        }
        .hero h1 { font-size: 1.8rem !important; }
        .hero p  { opacity: 0.9; margin-top: 0.35rem; }

        /* ---------- Info / stat chip ---------- */
        .chip {
            display: inline-block;
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            background: #e0e7ff;
            color: #3730a3;
            margin-right: 0.35rem;
        }
        .chip-green { background: #d1fae5; color: #065f46; }
        .chip-red   { background: #fee2e2; color: #991b1b; }
        .chip-amber { background: #fef3c7; color: #92400e; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# REUSABLE UI COMPONENTS
# =====================================================================
def hero(title: str, subtitle: str = "", icon: str = "🚀"):
    """Big gradient banner."""
    st.markdown(
        f"""
        <div class="hero">
            <h1>{icon} {title}</h1>
            {f'<p>{subtitle}</p>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def bubble(text: str, role: str = "bot"):
    """Render a chat-style bubble. role = 'user' | 'bot'."""
    cls = "bubble-user" if role == "user" else "bubble-bot"
    st.markdown(
        f'<div class="bubble-wrap"><div class="bubble {cls}">{text}</div></div>',
        unsafe_allow_html=True,
    )


def chips(items: list[tuple[str, str]]):
    """Render a row of chips. items = [(label, variant)] where
    variant ∈ {'', 'green', 'red', 'amber'}."""
    html = "".join(
        f'<span class="chip chip-{v}">{t}</span>' if v else f'<span class="chip">{t}</span>'
        for t, v in items
    )
    st.markdown(html, unsafe_allow_html=True)


def toast_success(msg: str):  st.toast(f"✅ {msg}", icon="✅")
def toast_error(msg: str):    st.toast(f"❌ {msg}", icon="❌")
def toast_info(msg: str):     st.toast(f"ℹ️ {msg}", icon="ℹ️")
def toast_warning(msg: str):  st.toast(f"⚠️ {msg}", icon="⚠️")