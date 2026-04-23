from frontend.components.ui_components import render_recommendation_summary


def render_summary_card(
    title: str,
    icon: str,
    summary: str,
    insights: list[str],
    accent: str,
    button_key: str,
) -> bool:
    """Compatibility wrapper for recommendation summary cards."""
    return render_recommendation_summary(
        title=title,
        summary=summary,
        insights=insights,
        icon=icon,
        button_key=button_key,
    )
