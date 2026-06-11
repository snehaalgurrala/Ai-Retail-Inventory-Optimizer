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
                linear-gradient(180deg, rgba(245,248,251,0.98), rgba(245,248,251,0.94)),
                var(--airio-bg);
            color: var(--airio-deep-navy);
        }}
        .block-container {{
            padding-top: 1.65rem;
            padding-bottom: 2.5rem;
        }}
        h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
            color: var(--airio-deep-navy);
            letter-spacing: 0;
        }}
        h2, h3, .stMarkdown h2, .stMarkdown h3 {{
            border-left: 4px solid var(--airio-primary-navy);
            padding-left: 0.65rem;
            line-height: 1.15;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"],
        div[data-testid="stExpander"],
        div[data-testid="stMetric"],
        div[data-testid="stDataFrame"] {{
            border-color: var(--airio-border) !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: var(--airio-card);
            border-radius: 14px;
            border-top: 3px solid var(--airio-primary-navy);
            box-shadow: 0 8px 18px rgba(10, 31, 51, 0.055);
        }}
        div[data-testid="stMetric"] {{
            background: var(--airio-card);
            border: 1px solid var(--airio-border);
            border-radius: 14px;
            padding: 0.85rem 0.95rem;
            border-top: 3px solid var(--airio-primary-navy);
            box-shadow: 0 8px 18px rgba(10, 31, 51, 0.045);
            min-height: 98px;
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
            background: var(--airio-primary-navy);
            color: #ffffff;
            border: 1px solid var(--airio-primary-navy);
            border-radius: 10px;
            box-shadow: 0 6px 14px rgba(24, 63, 95, 0.16);
            font-weight: 750;
        }}
        .stButton > button:hover,
        .stDownloadButton > button:hover {{
            background: var(--airio-navy-hover);
            color: #ffffff;
            border-color: var(--airio-navy-hover);
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
        div[data-testid="stDataFrame"] {{
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 22px rgba(10, 31, 51, 0.05);
        }}
        div[data-testid="stDataFrame"] [role="columnheader"],
        div[data-testid="stDataFrame"] thead tr th {{
            background: var(--airio-primary-navy) !important;
            color: #ffffff !important;
            font-weight: 750 !important;
        }}
        div[data-testid="stAlert"] {{
            border-radius: 12px;
            border: 1px solid var(--airio-border);
        }}
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #EAF1F7, #F5F8FB);
            border-right: 1px solid var(--airio-border);
        }}
        section[data-testid="stSidebar"] a,
        section[data-testid="stSidebar"] button {{
            border-radius: 10px;
        }}
        section[data-testid="stSidebar"] [aria-current="page"],
        section[data-testid="stSidebar"] a[aria-current="page"] {{
            background: var(--airio-primary-navy) !important;
            color: #ffffff !important;
            border-left: 4px solid var(--airio-green);
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
        @media (prefers-color-scheme: dark) {{
            .stApp {{
                background: #0b1621;
            }}
            div[data-testid="stVerticalBlockBorderWrapper"],
            div[data-testid="stMetric"] {{
                background: color-mix(in srgb, var(--background-color) 70%, var(--secondary-background-color) 30%);
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
