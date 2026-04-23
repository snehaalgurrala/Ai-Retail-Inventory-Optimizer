import streamlit as st


def render_summary_card(
    title: str,
    icon: str,
    summary: str,
    insights: list[str],
    accent: str,
    button_key: str,
) -> bool:
    """Render one modern recommendation summary card."""
    insight_items = "".join(
        f"<li>{insight}</li>" for insight in insights[:3] if insight
    )

    st.markdown(
        f"""
        <div class="summary-card summary-card-{accent}">
            <div class="summary-card-top">
                <div>
                    <div class="summary-card-title">{title}</div>
                    <div class="summary-card-summary">{summary}</div>
                </div>
                <div class="summary-card-icon">{icon}</div>
            </div>
            <ul class="summary-card-insights">
                {insight_items}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return st.button("Review", key=button_key, use_container_width=True)
