import sys
import importlib
from datetime import date, datetime
from pathlib import Path
from time import perf_counter

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.agents.orchestrator_agent import run_agent_graph  # noqa: E402
from backend.services.depletion_formatter import (  # noqa: E402
    depletion_urgency_label,
    exact_depletion_tooltip,
    format_depletion_window,
)
from backend.services.low_stock_service import get_low_stock_items  # noqa: E402
from backend.services.stock_alternative_service import (  # noqa: E402
    get_alternative_availability_for_low_stock,
    get_surplus_stock_items,
)
from backend.services import (  # noqa: E402
    abnormal_order_report,
    agent_summary_service,
    depletion_formatter,
    email_service,
    report_service,
)
from backend.utils.data_loader import load_all_data  # noqa: E402
from frontend.components.ui_components import (  # noqa: E402
    apply_command_center_styles,
    render_agent_command_card,
    render_command_center_orchestrator_card,
    render_kpi_card,
    render_low_stock_alert_card,
    render_table,
)
from frontend.utils.page_helpers import (  # noqa: E402
    apply_page_style,
    render_chart_card,
    style_donut_chart,
    style_sales_trend_chart,
)


st.set_page_config(
    page_title="AI Retail Inventory Optimizer",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_dashboard_data() -> dict[str, pd.DataFrame]:
    return load_all_data()


def _get_agent_summary_service():
    """Reload the summary service safely during Streamlit hot reloads."""
    return importlib.reload(agent_summary_service)


def _get_email_service():
    """Reload email helpers safely during Streamlit hot reloads.

    ``importlib.reload`` does not reload a module's dependencies, so we refresh
    the upstream modules ``email_service`` imports from first. Otherwise a
    reloaded ``email_service`` re-binds against a stale cached
    ``depletion_formatter``/``report_service`` in ``sys.modules`` and raises
    ``ImportError`` for symbols added since that cached version was loaded.
    """
    importlib.reload(depletion_formatter)
    importlib.reload(report_service)
    return importlib.reload(email_service)


def _get_report_service():
    """Reload report helpers safely during Streamlit hot reloads."""
    return importlib.reload(report_service)


def processed_file_path(filename: str) -> Path:
    return PROJECT_ROOT / "data" / "processed" / filename


def load_processed_output(filename: str) -> pd.DataFrame:
    file_path = processed_file_path(filename)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def processed_files_exist(filenames: list[str]) -> bool:
    return all(processed_file_path(filename).exists() for filename in filenames)


@st.cache_data
def load_agent_dashboard_outputs() -> dict[str, pd.DataFrame]:
    if processed_files_exist(["agent_outputs.csv", "orchestrator_summary.csv"]):
        service = _get_agent_summary_service()
        ensure_fn = getattr(service, "ensure_agent_card_summaries", None)
        if callable(ensure_fn):
            ensure_fn()
    return {
        "agent_outputs": load_processed_output("agent_outputs.csv"),
        "agent_card_summaries": load_processed_output("agent_card_summaries.csv"),
        "orchestrator_summary": load_processed_output("orchestrator_summary.csv"),
        "recommendations": load_processed_output("recommendations.csv"),
    }


def safe_sum(df: pd.DataFrame, column: str) -> int:
    if df.empty or column not in df.columns:
        return 0
    return int(pd.to_numeric(df[column], errors="coerce").fillna(0).sum())


def processed_row_count(filename: str) -> int:
    file_path = processed_file_path(filename)
    if not file_path.exists():
        return 0
    try:
        return len(pd.read_csv(file_path))
    except Exception:
        return 0


def latest_timestamp_from_files(filenames: list[str]) -> str:
    latest_time = None
    for filename in filenames:
        file_path = processed_file_path(filename)
        if file_path.exists():
            modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            if latest_time is None or modified_time > latest_time:
                latest_time = modified_time
    if latest_time is None:
        return "No agent run recorded yet"
    return latest_time.strftime("%d %b %Y, %I:%M %p")


def latest_timestamp_iso_from_files(filenames: list[str]) -> str:
    latest_time = None
    for filename in filenames:
        file_path = processed_file_path(filename)
        if file_path.exists():
            modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            if latest_time is None or modified_time > latest_time:
                latest_time = modified_time
    if latest_time is None:
        return ""
    return latest_time.isoformat(timespec="seconds")


def database_health_summary() -> str:
    required_files = [
        PROJECT_ROOT / "data" / "raw" / "inventory.csv",
        PROJECT_ROOT / "data" / "raw" / "products.csv",
        PROJECT_ROOT / "data" / "raw" / "sales.csv",
        PROJECT_ROOT / "data" / "raw" / "stores.csv",
        PROJECT_ROOT / "data" / "raw" / "suppliers.csv",
        PROJECT_ROOT / "data" / "raw" / "transactions.csv",
        processed_file_path("recommendations.csv"),
        processed_file_path("agent_outputs.csv"),
        processed_file_path("orchestrator_summary.csv"),
    ]
    available_count = sum(path.exists() for path in required_files)
    total_count = len(required_files)
    if available_count == total_count:
        return f"Healthy ({available_count}/{total_count})"
    if available_count >= total_count - 2:
        return f"Watch ({available_count}/{total_count})"
    return f"Needs Attention ({available_count}/{total_count})"


def get_current_inventory_quantity(
    inventory: pd.DataFrame,
    transactions: pd.DataFrame,
) -> tuple[int, str]:
    if not inventory.empty and "stock_level" in inventory.columns:
        return safe_sum(inventory, "stock_level"), "inventory.csv stock snapshot"

    if transactions.empty or "transaction_type" not in transactions.columns:
        return 0, "No usable inventory source"

    snapshot_rows = transactions[
        transactions["transaction_type"].eq("inventory_snapshot")
    ]
    if not snapshot_rows.empty and "quantity" in snapshot_rows.columns:
        return safe_sum(snapshot_rows, "quantity"), "transactions.csv inventory snapshots"

    quantity = pd.to_numeric(transactions.get("quantity"), errors="coerce").fillna(0)
    transaction_type = transactions["transaction_type"].fillna("")
    signed_quantity = quantity.copy()
    signed_quantity[transaction_type.isin(["sale", "transfer_out"])] *= -1
    return int(signed_quantity.sum()), "transactions.csv movement history"


def build_sales_trend_chart(sales: pd.DataFrame):
    if sales.empty or not {"date", "quantity_sold"}.issubset(sales.columns):
        return None

    sales_trend = (
        sales.assign(
            quantity_sold=pd.to_numeric(
                sales["quantity_sold"],
                errors="coerce",
            ).fillna(0)
        )
        .groupby("date", as_index=False)["quantity_sold"]
        .sum()
        .sort_values("date")
    )
    chart = px.line(
        sales_trend,
        x="date",
        y="quantity_sold",
        markers=True,
        labels={"date": "Date", "quantity_sold": "Quantity Sold"},
    )
    return style_sales_trend_chart(chart)


def build_inventory_distribution_chart(
    inventory: pd.DataFrame,
    products: pd.DataFrame,
):
    if inventory.empty or not {"product_id", "stock_level"}.issubset(inventory.columns):
        return None

    inventory_view = inventory.copy()
    if {"product_id", "category"}.issubset(products.columns):
        inventory_view = inventory_view.merge(
            products[["product_id", "category"]],
            on="product_id",
            how="left",
        )

    if "category" not in inventory_view.columns:
        return None

    stock_by_category = (
        inventory_view.dropna(subset=["category"])
        .assign(
            stock_level=pd.to_numeric(
                inventory_view["stock_level"],
                errors="coerce",
            ).fillna(0)
        )
        .groupby("category", as_index=False)["stock_level"]
        .sum()
        .sort_values("stock_level", ascending=False)
    )
    if stock_by_category.empty:
        return None

    chart = px.pie(
        stock_by_category,
        names="category",
        values="stock_level",
    )
    return style_donut_chart(chart)


def latest_recommendations_table(recommendations: pd.DataFrame) -> pd.DataFrame:
    if recommendations.empty:
        return pd.DataFrame()

    latest = recommendations.copy()
    if "priority" in latest.columns:
        latest["_priority_rank"] = (
            latest["priority"]
            .fillna("")
            .astype(str)
            .str.lower()
            .map({"high": 0, "medium": 1, "low": 2})
            .fillna(3)
        )
        latest = latest.sort_values(["_priority_rank", "recommendation_id"])

    # Fill display gaps so the table never shows raw Python ``None`` values.
    def _blank_mask(series: pd.Series) -> pd.Series:
        text = series.astype(str).str.strip().str.lower()
        return series.isna() | text.isin(["", "none", "nan"])

    # ``store_id`` is missing for store-agnostic recommendation types such as
    # supplier_risk_alert. Show a readable placeholder instead of ``None``.
    if "store_id" in latest.columns:
        store = latest["store_id"].astype("object")
        # Drop a trailing ``.0`` that float store ids pick up from the CSV.
        store = store.map(
            lambda value: str(value).strip()[:-2]
            if isinstance(value, str) and str(value).strip().endswith(".0")
            else value
        )
        store_numeric = pd.to_numeric(latest["store_id"], errors="coerce")
        store = store.mask(store_numeric.notna(), store_numeric.astype("Int64").astype("object"))
        latest["store_id"] = store.mask(_blank_mask(latest["store_id"]), "N/A").astype(str)

    # ``urgency_label`` / ``depletion_window`` only apply to recommendations that
    # carry a depletion prediction. Derive them when a prediction exists,
    # otherwise show "N/A" rather than ``None``.
    days = pd.to_numeric(latest.get("predicted_days_remaining"), errors="coerce")
    if "depletion_window" in latest.columns:
        derived_window = days.map(
            lambda value: format_depletion_window(value) if pd.notna(value) else "N/A"
        )
        latest["depletion_window"] = latest["depletion_window"].mask(
            _blank_mask(latest["depletion_window"]), derived_window
        )
    if "urgency_label" in latest.columns:
        derived_urgency = days.map(
            lambda value: depletion_urgency_label(value) if pd.notna(value) else "N/A"
        )
        latest["urgency_label"] = latest["urgency_label"].mask(
            _blank_mask(latest["urgency_label"]), derived_urgency
        )

    columns = [
        column
        for column in [
            "recommendation_id",
            "recommendation_type",
            "product_name",
            "store_id",
            "priority",
            "urgency_label",
            "depletion_window",
            "action",
            "reason",
            "source_agent",
            "status",
        ]
        if column in latest.columns
    ]
    display = latest.head(10)[columns].copy()
    # Final safety net: never surface raw ``None``/``NaN`` in any display column.
    for column in display.columns:
        display[column] = display[column].where(~_blank_mask(display[column]), "-")
    return display


def format_alert_display_table(alerts: pd.DataFrame) -> pd.DataFrame:
    display = alerts.copy()
    days = pd.to_numeric(display.get("predicted_days_remaining"), errors="coerce")
    if "depletion_window" not in display.columns:
        display["depletion_window"] = days.map(format_depletion_window)
    else:
        display["depletion_window"] = display["depletion_window"].fillna(days.map(format_depletion_window))
    if "urgency_label" not in display.columns:
        display["urgency_label"] = days.map(depletion_urgency_label)
    else:
        display["urgency_label"] = display["urgency_label"].fillna(days.map(depletion_urgency_label))
    display["exact_estimate"] = days.map(exact_depletion_tooltip)
    return display


def available_report_dates(sales_df: pd.DataFrame, inventory_df: pd.DataFrame) -> tuple[date, date]:
    date_values = []
    if not sales_df.empty and "date" in sales_df.columns:
        date_values.append(pd.to_datetime(sales_df["date"], errors="coerce"))
    if not inventory_df.empty and "last_updated" in inventory_df.columns:
        date_values.append(pd.to_datetime(inventory_df["last_updated"], errors="coerce"))
    if not date_values:
        today = datetime.now().date()
        return today, today
    combined = pd.concat(date_values).dropna()
    if combined.empty:
        today = datetime.now().date()
        return today, today
    return combined.min().date(), combined.max().date()


def report_send_signature(report_type: str, branch_filter: str, start_date, end_date) -> str:
    return "|".join([str(report_type), str(branch_filter), str(start_date), str(end_date)])


def send_dashboard_report(report_type: str, branch_filter: str, start_date, end_date) -> None:
    if start_date > end_date:
        st.session_state["report_email_error"] = "Start date must be before or equal to end date."
        return

    signature = report_send_signature(report_type, branch_filter, start_date, end_date)
    if st.session_state.get("last_report_send_signature") == signature:
        st.session_state["report_email_warning"] = "This same report was already sent in this session."
        return

    reports = _get_report_service()
    emails = _get_email_service()
    if report_type == "inventory":
        report = reports.generate_inventory_report(branch_filter, start_date, end_date)
        success_message = (
            "Inventory report PDF and CSV sent successfully to manager."
            if report.get("pdf_path")
            else "Inventory report CSV sent successfully to manager."
        )
        subject = f"Inventory Intelligence Report - {report.get('branch_label', branch_filter)}"
    else:
        report = reports.generate_sales_report(branch_filter, start_date, end_date)
        success_message = "Sales report sent successfully to manager."
        subject = f"Sales Report - {report.get('branch_label', branch_filter)}"

    if not report.get("success"):
        st.session_state["report_email_error"] = str(report.get("message", "No report data found."))
        return

    email_result = emails.send_report_email(
        subject=subject,
        html_body=report.get("email_html", report.get("html", "")),
        attachment_path=report.get("attachment_path"),
        attachment_paths=report.get("attachment_paths"),
    )
    if email_result.get("success"):
        st.session_state["last_report_send_signature"] = signature
        st.session_state["report_email_success"] = success_message
        warnings = [
            str(report.get("pdf_warning", "") or ""),
            str(email_result.get("warning", "") or ""),
        ]
        warning_text = " ".join(warning for warning in warnings if warning)
        if warning_text:
            st.session_state["report_email_warning"] = warning_text
    elif email_result.get("warning"):
        st.session_state["report_email_warning"] = str(email_result.get("warning"))
    else:
        st.session_state["report_email_error"] = str(email_result.get("message", "Report email could not be sent."))


def send_abnormal_order_report(period: str, target_date: date | None = None) -> None:
    """Build and email the executive Abnormal Order Intelligence Report.

    Reuses the shared abnormal-order pipeline (same risk bands, scores, AI
    narrative and inventory scope as the Customer Intelligence page) for the
    requested day, then routes the outcome through the existing report-email
    status banners.
    """
    result = abnormal_order_report.send_abnormal_order_report_email(period, target_date)
    message = str(result.get("message", "") or "")
    if result.get("success"):
        st.session_state["report_email_success"] = message or "Abnormal Order Intelligence Report sent."
    elif result.get("email_sent") is False and "No abnormal orders" in message:
        st.session_state["report_email_warning"] = message
    else:
        st.session_state["report_email_error"] = message or "Abnormal Order Intelligence Report could not be sent."


def render_abnormal_order_report_section() -> None:
    """New section under the Report Email Center for executive abnormal-order reports."""
    st.markdown('<div style="height:0.4rem"></div>', unsafe_allow_html=True)
    st.divider()
    st.markdown(
        '<div class="report-center-title">🚨 Abnormal Order Intelligence Reports</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="report-center-subtitle">Send a premium executive report of the abnormal '
        'customer orders detected on a single day — same risk classifications, AI investigation '
        'and network inventory scope as the Customer Intelligence page.</div>',
        unsafe_allow_html=True,
    )

    abn_cols = st.columns([1.5, 1.25, 1.25, 1.35], gap="small")
    with abn_cols[0]:
        selected_abnormal_date = st.date_input(
            "Report Date",
            value=datetime.now().date(),
            max_value=datetime.now().date(),
            key="abnormal_report_date",
            help="Used by the 'Send Selected Date' button. Today / Yesterday ignore this field.",
        )
    with abn_cols[1]:
        st.write("")
        send_today = st.button(
            "Send Today's Abnormal Orders Report",
            use_container_width=True,
            key="send_abnormal_today",
        )
    with abn_cols[2]:
        st.write("")
        send_yesterday = st.button(
            "Send Yesterday's Abnormal Orders Report",
            use_container_width=True,
            key="send_abnormal_yesterday",
        )
    with abn_cols[3]:
        st.write("")
        send_selected = st.button(
            "Send Selected Date Report",
            use_container_width=True,
            key="send_abnormal_selected",
        )

    if send_today:
        send_abnormal_order_report("today")
        st.rerun()
    if send_yesterday:
        send_abnormal_order_report("yesterday")
        st.rerun()
    if send_selected:
        send_abnormal_order_report("date", selected_abnormal_date)
        st.rerun()


def render_report_email_center(
    sales_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    stores_df: pd.DataFrame,
) -> None:
    st.markdown(
        """
        <style>
        .report-center-title {
            font-size: 1.18rem;
            font-weight: 800;
            color: var(--airio-deep-navy, #0A1F33);
            margin-bottom: 0.15rem;
        }
        .report-center-subtitle {
            color: rgba(10, 31, 51, 0.68);
            font-size: 0.9rem;
            margin-bottom: 0.75rem;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.report-center-title) {
            border-color: var(--airio-border, #D8E2EC);
            border-top: 4px solid var(--airio-primary-navy, #183F5F);
            box-shadow: 0 12px 30px rgba(10, 31, 51, 0.08);
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.report-center-title) .stButton > button {
            background: var(--airio-green, #6CB33F);
            border-color: var(--airio-green, #6CB33F);
            color: #ffffff;
            box-shadow: 0 6px 14px rgba(108, 179, 63, 0.18);
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.report-center-title) .stButton > button:hover {
            background: #5a9b35;
            border-color: #5a9b35;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown('<div class="report-center-title">📩 Report Email Center</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="report-center-subtitle">Send branch-wise inventory or sales reports directly to the manager.</div>',
            unsafe_allow_html=True,
        )

        min_report_date, max_report_date = available_report_dates(sales_df, inventory_df)
        branch_options = ["All Branches"]
        branch_label_to_id = {"All Branches": "All Branches"}
        if not stores_df.empty and {"store_id", "store_name"}.issubset(stores_df.columns):
            for _, store_row in stores_df.sort_values("store_name").iterrows():
                store_id = str(store_row.get("store_id", "")).strip()
                store_name = str(store_row.get("store_name", store_id)).strip()
                city = str(store_row.get("city", "") or "").strip()
                label = f"{store_name} ({store_id})"
                if city:
                    label = f"{store_name} ({store_id}, {city})"
                branch_options.append(label)
                branch_label_to_id[label] = store_id

        report_cols = st.columns([2.2, 1.15, 1.15, 1.35, 1.2], gap="small")
        with report_cols[0]:
            selected_branch_label = st.selectbox(
                "Branch Selector",
                branch_options,
                key="report_branch_selector",
            )
        with report_cols[1]:
            selected_start_date = st.date_input(
                "Start Date",
                value=min_report_date,
                min_value=min_report_date,
                max_value=max_report_date,
                key="report_start_date",
            )
        with report_cols[2]:
            selected_end_date = st.date_input(
                "End Date",
                value=max_report_date,
                min_value=min_report_date,
                max_value=max_report_date,
                key="report_end_date",
            )

        branch_filter = branch_label_to_id.get(selected_branch_label, "All Branches")
        with report_cols[3]:
            st.write("")
            send_inventory_report = st.button(
                "Send Inventory Report",
                use_container_width=True,
                key="send_inventory_report",
            )
            st.caption("Inventory report uses latest stock snapshot. Date range applies only if inventory date exists.")
        with report_cols[4]:
            st.write("")
            send_sales_report = st.button(
                "Send Sales Report",
                use_container_width=True,
                key="send_sales_report",
            )

        if send_inventory_report:
            send_dashboard_report("inventory", branch_filter, selected_start_date, selected_end_date)
            st.rerun()
        if send_sales_report:
            send_dashboard_report("sales", branch_filter, selected_start_date, selected_end_date)
            st.rerun()

        render_abnormal_order_report_section()


apply_page_style()
apply_command_center_styles()

refresh_message = st.session_state.pop("dashboard_refresh_message", "")
refresh_email_message = st.session_state.pop("dashboard_email_message", "")
refresh_email_warning = st.session_state.pop("dashboard_email_warning", "")
refresh_timing = st.session_state.pop("dashboard_refresh_timing", {})
report_email_success = st.session_state.pop("report_email_success", "")
report_email_warning = st.session_state.pop("report_email_warning", "")
report_email_error = st.session_state.pop("report_email_error", "")
agent_output_files = [
    "agent_outputs.csv",
    "agent_card_summaries.csv",
    "orchestrator_summary.csv",
    "recommendations.csv",
]

if refresh_message:
    st.success(refresh_message)
if refresh_email_message:
    st.info(refresh_email_message)
if refresh_email_warning:
    st.warning(refresh_email_warning)
if refresh_timing:
    st.caption(
        "Refresh timing: "
        f"data analysis {refresh_timing.get('data_analysis_seconds', 0)}s | "
        f"LLM {refresh_timing.get('llm_seconds', refresh_timing.get('llm_summary_seconds', 0))}s | "
        f"email {refresh_timing.get('email_seconds', 0)}s | "
        f"file save {refresh_timing.get('file_save_seconds', 0)}s | "
        f"total {refresh_timing.get('total_seconds', refresh_timing.get('total_refresh_seconds', 0))}s"
    )
if report_email_success:
    st.success(report_email_success)
if report_email_warning:
    st.warning(report_email_warning)
if report_email_error:
    st.error(report_email_error)

try:
    data = load_dashboard_data()
except Exception as error:
    st.error("Could not load the dashboard data.")
    st.exception(error)
    st.stop()

products = data["products"]
sales = data["sales"]
inventory = data["inventory"]
transactions = data["transactions"]
stores = data["stores"]

agent_output_state = load_agent_dashboard_outputs()
agent_outputs = agent_output_state["agent_outputs"]
agent_card_summaries = agent_output_state["agent_card_summaries"]
orchestrator_summary_df = agent_output_state["orchestrator_summary"]
recommendations = agent_output_state["recommendations"]
low_stock_alerts = get_low_stock_items(save_output=True)
surplus_stock_items = get_surplus_stock_items(inventory, products, data["stores"])
alternative_availability_alerts = get_alternative_availability_for_low_stock(
    inventory,
    products,
    data["stores"],
    low_stock_alerts,
)
summary_service = _get_agent_summary_service()
build_low_stock_alert_text = getattr(
    summary_service,
    "build_low_stock_alert_text",
    lambda df: "No critical low-stock alerts right now.",
)

has_agent_run = processed_files_exist(
    ["agent_outputs.csv", "orchestrator_summary.csv", "recommendations.csv"]
)

if has_agent_run and agent_card_summaries.empty:
    generate_fn = getattr(summary_service, "generate_agent_card_summaries", None)
    if callable(generate_fn):
        agent_card_summaries, orchestrator_summary_df = generate_fn(
            agent_outputs_df=agent_outputs,
            recommendations_df=recommendations,
            orchestrator_summary_df=orchestrator_summary_df,
            save_output=True,
        )

current_inventory_quantity, inventory_source = get_current_inventory_quantity(
    inventory,
    transactions,
)
total_sales_quantity = safe_sum(sales, "quantity_sold")
dead_stock_count = processed_row_count("dead_stock_candidates.csv")
last_agent_run_time = latest_timestamp_from_files(agent_output_files)
last_updated_timestamp = latest_timestamp_iso_from_files(agent_output_files)

low_stock_count = int(len(low_stock_alerts))
current_low_stock_alert_text = build_low_stock_alert_text(low_stock_alerts)

header_left, header_right = st.columns([4.8, 1.2], gap="large")
with header_left:
    st.markdown(
        """
        <div class="home-command-header">
          <div class="command-header-title">Agent Command Center</div>
          <div class="command-header-subtitle">A premium operations view of orchestrator health, specialist agent updates, and the latest actions worth taking.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with header_right:
    st.markdown(
        f'<div class="command-meta">Last run: {last_agent_run_time}</div>',
        unsafe_allow_html=True,
    )
    send_email_alert = st.checkbox(
        "Send low-stock email alert",
        value=False,
        help="Queues email after dashboard outputs are saved. Duplicate alerts are skipped.",
    )
    if st.button("Run / Refresh Agents", use_container_width=True):
        try:
            email_result = {
                "warning": "",
                "message": "Low-stock email alert was not requested.",
            }
            with st.status("Refreshing all agents on the latest data...", expanded=True) as status:
                st.write("Step 1: Loading latest CSVs")
                load_dashboard_data.clear()
                load_agent_dashboard_outputs.clear()
                st.cache_data.clear()

                st.write("Step 2: Running analysis")
                final_state = run_agent_graph(save_output=True)

                st.write("Step 3: Generating summaries")
                timing_log = final_state.get("timing_log", {})

                st.write("Step 4: Saving outputs")
                refreshed_low_stock_df = get_low_stock_items(save_output=True)

                if send_email_alert and not refreshed_low_stock_df.empty:
                    st.write("Step 5: Email alert queued/sent")
                    email_started = perf_counter()
                    email_helpers = _get_email_service()
                    queue_email_fn = getattr(email_helpers, "queue_low_stock_alert_email", None)
                    if callable(queue_email_fn):
                        email_result = queue_email_fn(refreshed_low_stock_df)
                    else:
                        send_email_fn = getattr(email_helpers, "send_low_stock_alert_email")
                        email_result = send_email_fn(refreshed_low_stock_df)
                    timing_log["email_seconds"] = round(perf_counter() - email_started, 3)
                elif refreshed_low_stock_df.empty:
                    st.write("Step 5: Email alert skipped; no low-stock rows")
                    timing_log["email_seconds"] = 0
                    email_result = {
                        "warning": "",
                        "message": "No low-stock items found.",
                    }
                else:
                    st.write("Step 5: Email alert skipped by user")
                    timing_log["email_seconds"] = 0

                load_dashboard_data.clear()
                load_agent_dashboard_outputs.clear()
                st.cache_data.clear()
                status.update(label="Agent refresh complete.", state="complete")
            refreshed_count = len(final_state.get("unified_recommendations", []))
            refreshed_time = final_state.get("combined_output", {}).get(
                "run_time",
                "just now",
            )
            timing_log = final_state.get("timing_log", {})
            timing_log["email_seconds"] = timing_log.get("email_seconds", 0)
            st.session_state["dashboard_refresh_message"] = (
                f"Agents refreshed successfully. {refreshed_count:,} recommendations generated at {refreshed_time}."
            )
            st.session_state["dashboard_refresh_timing"] = timing_log
            if email_result.get("warning"):
                st.session_state["dashboard_email_warning"] = str(
                    email_result.get("warning", "")
                )
            elif email_result.get("message"):
                st.session_state["dashboard_email_message"] = str(
                    email_result.get("message", "")
                )
            st.rerun()
        except Exception as error:
            st.error("Could not refresh agents.")
            st.exception(error)

if has_agent_run and not orchestrator_summary_df.empty:
    orchestrator_row = orchestrator_summary_df.iloc[0]
    render_command_center_orchestrator_card(
        database_health=str(
            orchestrator_row.get("database_health", database_health_summary())
        ),
        total_recommendations=int(
            pd.to_numeric(
                orchestrator_row.get("total_recommendations", 0),
                errors="coerce",
            )
            or 0
        ),
        high_priority_alerts=int(
            pd.to_numeric(
                orchestrator_row.get("high_priority_alerts", 0),
                errors="coerce",
            )
            or 0
        ),
        last_run_time=str(
            orchestrator_row.get("last_agent_run_time", last_agent_run_time)
        ),
        low_stock_alert=current_low_stock_alert_text,
        surplus_alternative_alert=(
            f"{len(surplus_stock_items):,} surplus items or "
            f"{len(alternative_availability_alerts):,} transfer alternatives detected."
        ),
        top_risk=str(
            orchestrator_row.get(
                "top_risk",
                "No major risk stands out in the latest run.",
            )
        ),
        top_opportunity=str(
            orchestrator_row.get(
                "top_opportunity",
                "No standout commercial opportunity is available yet.",
            )
        ),
        executive_summary=str(
            orchestrator_row.get(
                "executive_summary",
                orchestrator_row.get(
                    "summary",
                    "Run agents to generate latest analysis.",
                ),
            )
        ),
        executive_recommendation=str(
            orchestrator_row.get(
                "executive_recommendation",
                "Review the highest-priority risks first, then move on the strongest commercial opportunity.",
            )
        ),
        summary_source=str(orchestrator_row.get("summary_source", "")),
    )
else:
    st.info('No agent run found. Click Run / Refresh Agents.')

if last_updated_timestamp:
    st.caption(f"Last updated: {last_updated_timestamp}")

st.caption(
    "Each specialist card shows the latest summarized insight, urgency, and next action from the newest agent run."
)

agent_order = [
    "inventory_agent",
    "pricing_agent",
    "transfer_agent",
    "risk_agent",
    "procurement_agent",
]
agent_lookup: dict[str, pd.Series] = {}
if not agent_card_summaries.empty and "agent_name" in agent_card_summaries.columns:
    for _, row in agent_card_summaries.iterrows():
        agent_lookup[str(row.get("agent_name", ""))] = row

agent_columns = st.columns(5, gap="medium")
for index, agent_key in enumerate(agent_order):
    with agent_columns[index]:
        row = agent_lookup.get(agent_key)
        if row is None:
            render_agent_command_card(
                agent_name=agent_key.replace("_", " ").title(),
                role_label="Agent summary",
                finding_count=0,
                priority_level="Info",
                summary="Run agents to generate the latest analysis.",
                recommended_action="Refresh the agents to populate this card.",
                accent="blue",
            )
        else:
            render_agent_command_card(
                agent_name=str(row.get("display_name", agent_key.replace("_", " ").title())),
                role_label=str(row.get("role_label", "Agent summary")),
                finding_count=int(
                    pd.to_numeric(row.get("finding_count", 0), errors="coerce") or 0
                ),
                priority_level=str(row.get("priority_level", "Info")),
                summary=str(
                    row.get("summary", "Run agents to generate the latest analysis.")
                ),
                recommended_action=str(
                    row.get("recommended_action", "Review the latest rows first.")
                ),
                accent=str(row.get("accent", "blue")),
            )

st.divider()

st.subheader("🚨 Low Stock Alerts")
if low_stock_alerts.empty:
    st.success("No critical low-stock alerts right now.")
else:
    render_low_stock_alert_card(current_low_stock_alert_text)
    with st.container(border=True):
        display_alerts = format_alert_display_table(low_stock_alerts)
        preview_df = display_alerts[
            [
                column
                for column in [
                    "product_name",
                    "store_name",
                    "city",
                    "current_quantity",
                    "urgency_label",
                    "depletion_window",
                    "exact_estimate",
                    "avg_daily_sales",
                    "demand_trend",
                    "risk_score",
                    "confidence_level",
                    "suggested_reorder_quantity",
                    "suggested_transfer_branch",
                    "priority",
                ]
                if column in display_alerts.columns
            ]
        ].head(5).rename(
            columns={
                "urgency_label": "Urgency",
                "depletion_window": "Depletion Window",
                "exact_estimate": "Exact Estimate",
            }
        )
        render_table(preview_df, max_height=320)
        if "risk_score" in low_stock_alerts.columns:
            st.caption("Predictive risk indicators")
            for _, row in display_alerts.head(3).iterrows():
                risk_score = int(pd.to_numeric(row.get("risk_score", 0), errors="coerce") or 0)
                product_label = row.get("product_name", row.get("product_id", "Product"))
                store_label = row.get("store_name", row.get("store_id", "Branch"))
                risk_label = row.get("urgency_label", row.get("risk_category", row.get("priority", "Risk")))
                badge_colors = {
                    "Critical": ("#991b1b", "#fee2e2"),
                    "High": ("#9a3412", "#ffedd5"),
                    "Medium": ("#92400e", "#fef3c7"),
                    "Healthy": ("#166534", "#dcfce7"),
                }
                color, background = badge_colors.get(str(risk_label), ("#166534", "#dcfce7"))
                st.markdown(
                    (
                        f"<span style='display:inline-block;padding:0.18rem 0.55rem;border-radius:999px;"
                        f"font-size:0.78rem;font-weight:700;color:{color};background:{background};'>"
                        f"{risk_label}</span> "
                        f"<span style='font-size:0.86rem;color:#0A1F33;'>{product_label} at {store_label}</span>"
                    ),
                    unsafe_allow_html=True,
                )
                st.progress(
                    min(max(risk_score, 0), 100),
                    text=f"{risk_label}: {product_label} at {store_label} - {row.get('depletion_window', '')}",
                )

st.divider()
render_report_email_center(sales, inventory, stores)

st.divider()

st.subheader("Latest Recommendations")
st.caption("The newest recommendation queue from the latest orchestrator run.")

if not has_agent_run or recommendations.empty:
    st.info('No agent run found. Click Run / Refresh Agents.')
else:
    with st.container(border=True):
        render_table(latest_recommendations_table(recommendations))

st.divider()

st.subheader("Business Overview")
st.caption("Core performance indicators from inventory, sales, and analyzer outputs.")

kpi_columns = st.columns(4, gap="medium")
with kpi_columns[0]:
    render_kpi_card(
        "Total Sales",
        f"{total_sales_quantity:,}",
        f"{len(sales):,} sales rows in the current dataset",
        "blue",
        icon="💰",
    )
with kpi_columns[1]:
    render_kpi_card(
        "Total Inventory",
        f"{current_inventory_quantity:,}",
        inventory_source,
        "purple",
        icon="📦",
    )
with kpi_columns[2]:
    render_kpi_card(
        "Low Stock",
        f"{low_stock_count:,}",
        "Predictive depletion alerts",
        "orange",
        icon="⚠️",
        support="Review reorders" if low_stock_count else "All healthy",
    )
with kpi_columns[3]:
    render_kpi_card(
        "Dead Stock",
        f"{dead_stock_count:,}",
        "Candidates from processed analyzer output",
        "red",
        icon="🛑",
    )

st.caption(f"Current inventory source: {inventory_source}")
st.divider()

st.subheader("Performance Trends")
st.caption("Sales movement and inventory composition in a balanced two-column view.")

middle_left, middle_right = st.columns(2, gap="large")
with middle_left:
    render_chart_card(
        "Sales Trend",
        "Daily sales movement from the available sales history.",
        build_sales_trend_chart(sales),
        "No date-based sales trend data is available.",
    )

with middle_right:
    render_chart_card(
        "Inventory Distribution",
        "Current stock units distributed across product categories.",
        build_inventory_distribution_chart(inventory, products),
        "No category inventory data is available.",
    )
