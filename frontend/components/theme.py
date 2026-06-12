import streamlit as st


PRIMARY_NAVY = "#183F5F"
DARK_NAVY = "#0A1F33"
ACCENT_GREEN = "#6CB33F"
SOFT_GREEN = "#A6D96A"
LIGHT_BG = "#F5F8FB"
CARD_BG = "#FFFFFF"
BORDER = "#D8E2EC"
MUTED_TEXT = "#476C8B"
WARNING = "#C76A12"
DANGER = "#B42318"
SOFT_BLUE = "#EAF1F7"
SOFT_AMBER = "#FFF7E8"
SOFT_RED = "#FFF1F2"
NAVY_HOVER = "#102F49"


THEME_COLORS = {
    "primary_navy": PRIMARY_NAVY,
    "deep_navy": DARK_NAVY,
    "fresh_green": ACCENT_GREEN,
    "soft_green": SOFT_GREEN,
    "light_background": LIGHT_BG,
    "white": CARD_BG,
    "soft_border": BORDER,
    "muted_text": MUTED_TEXT,
    "soft_blue": SOFT_BLUE,
    "soft_amber": SOFT_AMBER,
    "soft_red": SOFT_RED,
    "risk_red": DANGER,
    "warning_orange": WARNING,
    "navy_hover": NAVY_HOVER,
}


def apply_enterprise_theme() -> None:
    """Apply the shared logistics-enterprise Streamlit theme."""
    st.markdown(
        f"""
        <style>
        :root {{
            --airio-primary-navy: {PRIMARY_NAVY};
            --airio-deep-navy: {DARK_NAVY};
            --airio-green: {ACCENT_GREEN};
            --airio-soft-green: {SOFT_GREEN};
            --airio-bg: {LIGHT_BG};
            --airio-card: {CARD_BG};
            --airio-border: {BORDER};
            --airio-muted: {MUTED_TEXT};
            --airio-soft-blue: {SOFT_BLUE};
            --airio-soft-amber: {SOFT_AMBER};
            --airio-soft-red: {SOFT_RED};
            --airio-risk: {DANGER};
            --airio-warning: {WARNING};
            --airio-navy-hover: {NAVY_HOVER};
        }}

        .stApp {{
            background:
                radial-gradient(1200px 520px at 12% -8%, rgba(108,179,63,0.06), transparent 60%),
                radial-gradient(1100px 560px at 100% 0%, rgba(24,63,95,0.07), transparent 58%),
                linear-gradient(180deg, #FBFDFE 0%, var(--airio-bg) 42%, #EEF3F8 100%);
            color: var(--airio-deep-navy);
            -webkit-font-smoothing: antialiased;
            text-rendering: optimizeLegibility;
        }}
        .block-container {{
            padding-top: 2.1rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }}
        h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
            color: var(--airio-deep-navy);
            letter-spacing: -0.01em;
            font-weight: 800;
        }}
        h2, .stMarkdown h2 {{ font-size: 1.5rem; margin-top: 0.4rem; }}
        h3, .stMarkdown h3 {{ font-size: 1.18rem; }}
        h2, h3, .stMarkdown h2, .stMarkdown h3 {{
            border-left: 4px solid var(--airio-primary-navy);
            padding-left: 0.7rem;
            line-height: 1.18;
        }}
        /* Section subheaders get a soft underline rhythm for hierarchy. */
        div[data-testid="stHeadingWithActionElements"] h3 {{
            padding-bottom: 0.25rem;
        }}
        hr, div[data-testid="stDivider"] hr {{
            border-color: var(--airio-border);
            opacity: 0.8;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"],
        div[data-testid="stExpander"],
        div[data-testid="stMetric"],
        div[data-testid="stDataFrame"] {{
            border-color: var(--airio-border) !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: var(--airio-card);
            border-radius: 16px;
            border-top: 3px solid var(--airio-primary-navy);
            box-shadow: 0 10px 26px rgba(10, 31, 51, 0.06);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
            transform: translateY(-2px);
            box-shadow: 0 18px 38px rgba(10, 31, 51, 0.11);
        }}
        div[data-testid="stExpander"] {{
            border-radius: 14px;
            overflow: hidden;
            box-shadow: 0 6px 16px rgba(10, 31, 51, 0.04);
        }}
        div[data-testid="stExpander"] summary:hover {{
            color: var(--airio-primary-navy);
        }}
        div[data-testid="stMetric"] {{
            background: var(--airio-card);
            border: 1px solid var(--airio-border);
            border-radius: 16px;
            padding: 0.9rem 1rem;
            border-top: 3px solid var(--airio-primary-navy);
            box-shadow: 0 10px 22px rgba(10, 31, 51, 0.05);
            min-height: 98px;
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }}
        div[data-testid="stMetric"]:hover {{
            transform: translateY(-2px);
            box-shadow: 0 18px 34px rgba(10, 31, 51, 0.1);
        }}
        div[data-testid="stMetric"] label {{
            color: rgba(10,31,51,0.68);
            font-weight: 750;
        }}
        div[data-testid="stMetricValue"] {{
            color: var(--airio-primary-navy);
            font-weight: 800;
        }}
        .stButton > button,
        .stDownloadButton > button,
        button[kind="primary"] {{
            background: linear-gradient(180deg, #1E4E76 0%, var(--airio-primary-navy) 100%);
            color: #ffffff;
            border: 1px solid var(--airio-primary-navy);
            border-radius: 10px;
            box-shadow: 0 6px 14px rgba(24, 63, 95, 0.18);
            font-weight: 700;
            transition: transform 0.14s ease, box-shadow 0.14s ease, background 0.14s ease;
        }}
        .stButton > button:hover,
        .stDownloadButton > button:hover {{
            background: linear-gradient(180deg, #1A4568 0%, var(--airio-navy-hover) 100%);
            color: #ffffff;
            border-color: var(--airio-navy-hover);
            transform: translateY(-1px);
            box-shadow: 0 10px 20px rgba(24, 63, 95, 0.24);
        }}
        .stButton > button:active,
        .stDownloadButton > button:active {{
            transform: translateY(0);
            box-shadow: 0 4px 10px rgba(24, 63, 95, 0.18);
        }}
        .stButton > button:focus,
        .stDownloadButton > button:focus {{
            box-shadow: 0 0 0 0.18rem rgba(108, 179, 63, 0.24);
        }}
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        textarea,
        input {{
            border-color: var(--airio-border) !important;
            border-radius: 10px !important;
        }}
        div[data-baseweb="select"] > div:focus-within,
        div[data-baseweb="input"] > div:focus-within,
        textarea:focus,
        input:focus {{
            border-color: var(--airio-green) !important;
            box-shadow: 0 0 0 0.14rem rgba(108, 179, 63, 0.18) !important;
        }}
        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {{
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 22px rgba(10, 31, 51, 0.05);
            border: 1px solid var(--airio-border);
        }}
        div[data-testid="stDataFrame"] [role="columnheader"],
        div[data-testid="stDataFrame"] thead tr th,
        div[data-testid="stTable"] thead tr th {{
            background: var(--airio-primary-navy) !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            letter-spacing: 0.01em;
        }}
        /* Zebra + row hover for HTML-based tables (st.table / styled frames). */
        div[data-testid="stTable"] tbody tr:nth-child(even) {{
            background: var(--airio-soft-blue);
        }}
        div[data-testid="stTable"] tbody tr:hover {{
            background: rgba(108, 179, 63, 0.10);
        }}
        div[data-testid="stTable"] td,
        div[data-testid="stTable"] th {{
            padding: 0.55rem 0.7rem !important;
            border-color: var(--airio-border) !important;
        }}
        div[data-testid="stAlert"] {{
            border-radius: 12px;
            border: 1px solid var(--airio-border);
            box-shadow: 0 6px 16px rgba(10, 31, 51, 0.04);
        }}
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #EAF1F7 0%, #F5F8FB 100%);
            border-right: 1px solid var(--airio-border);
        }}
        section[data-testid="stSidebar"] a,
        section[data-testid="stSidebar"] button {{
            border-radius: 10px;
        }}
        /* Nav links: subtle hover affordance + clear active state. */
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {{
            transition: background 0.15s ease, color 0.15s ease;
            font-weight: 650;
        }}
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {{
            background: rgba(24, 63, 95, 0.08);
            color: var(--airio-primary-navy);
        }}
        section[data-testid="stSidebar"] [aria-current="page"],
        section[data-testid="stSidebar"] a[aria-current="page"] {{
            background: var(--airio-primary-navy) !important;
            color: #ffffff !important;
            border-left: 4px solid var(--airio-green);
            box-shadow: 0 6px 14px rgba(24, 63, 95, 0.18);
        }}
        div[data-testid="stChatMessage"] {{
            border-radius: 14px;
            border: 1px solid var(--airio-border);
            background: var(--airio-card);
            box-shadow: 0 8px 20px rgba(10, 31, 51, 0.045);
        }}
        div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
            background: var(--airio-primary-navy);
            color: #ffffff;
            border-color: var(--airio-primary-navy);
        }}
        div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {{
            background: var(--airio-card);
            border-left: 4px solid var(--airio-green);
        }}
        .airio-page-header {{
            background: var(--airio-card);
            color: var(--airio-deep-navy);
            border: 1px solid var(--airio-border);
            border-top: 6px solid var(--airio-primary-navy);
            border-radius: 14px;
            padding: 1.15rem 1.25rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 8px 20px rgba(10,31,51,0.06);
            position: relative;
        }}
        .airio-page-header::after {{
            content: "";
            display: block;
            width: 72px;
            height: 3px;
            border-radius: 999px;
            background: var(--airio-green);
            margin-top: 0.8rem;
        }}
        .airio-page-header-title {{
            font-size: 1.72rem;
            line-height: 1.12;
            font-weight: 800;
            color: var(--airio-primary-navy);
        }}
        .airio-page-header-subtitle {{
            margin-top: 0.4rem;
            color: rgba(10,31,51,0.68);
            font-size: 0.94rem;
            line-height: 1.45;
        }}
        .airio-section-band {{
            background: var(--airio-card);
            border: 1px solid var(--airio-border);
            border-top: 4px solid var(--airio-primary-navy);
            border-radius: 14px;
            box-shadow: 0 8px 18px rgba(10,31,51,0.05);
        }}
        .airio-green-action .stButton > button,
        .airio-green-action button {{
            background: var(--airio-green) !important;
            border-color: var(--airio-green) !important;
            color: #ffffff !important;
        }}
        .airio-badge {{
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.24rem 0.58rem;
            background: rgba(108, 179, 63, 0.14);
            color: var(--airio-deep-navy);
            border: 1px solid rgba(108, 179, 63, 0.22);
            font-weight: 750;
            font-size: 0.75rem;
        }}
        /* Reusable AI insight panel (branded replacement for bare st.info loops). */
        .airio-insight-panel {{
            background: linear-gradient(180deg, #FFFFFF 0%, #F7FBF3 100%);
            border: 1px solid var(--airio-border);
            border-left: 4px solid var(--airio-green);
            border-radius: 14px;
            padding: 0.9rem 1.05rem;
            margin-bottom: 0.6rem;
            box-shadow: 0 8px 18px rgba(10, 31, 51, 0.05);
            transition: transform 0.16s ease, box-shadow 0.16s ease;
        }}
        .airio-insight-panel:hover {{
            transform: translateY(-2px);
            box-shadow: 0 16px 30px rgba(10, 31, 51, 0.1);
        }}
        .airio-insight-kicker {{
            display: flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-weight: 800;
            color: var(--airio-green);
            margin-bottom: 0.5rem;
        }}
        .airio-insight-item {{
            display: flex;
            gap: 0.55rem;
            align-items: flex-start;
            color: rgba(10, 31, 51, 0.86);
            font-size: 0.92rem;
            line-height: 1.5;
            padding: 0.28rem 0;
        }}
        .airio-insight-item + .airio-insight-item {{
            border-top: 1px dashed var(--airio-border);
        }}
        .airio-insight-dot {{
            flex: 0 0 auto;
            width: 7px;
            height: 7px;
            margin-top: 0.5rem;
            border-radius: 999px;
            background: var(--airio-green);
        }}
        /* Custom scrollbar for a more finished feel. */
        ::-webkit-scrollbar {{ width: 11px; height: 11px; }}
        ::-webkit-scrollbar-thumb {{
            background: rgba(24, 63, 95, 0.26);
            border-radius: 999px;
            border: 3px solid transparent;
            background-clip: content-box;
        }}
        ::-webkit-scrollbar-thumb:hover {{ background: rgba(24, 63, 95, 0.4); background-clip: content-box; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
