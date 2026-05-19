from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _text(value: Any, fallback: str = "") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return fallback
    text = str(value).strip()
    return text or fallback


def _table_rows(df: pd.DataFrame, columns: list[str], limit: int = 8) -> list[list[str]]:
    if df.empty:
        return [["No rows available."]]
    visible_columns = [column for column in columns if column in df.columns]
    if not visible_columns:
        return [["No rows available."]]
    rows = [[column.replace("_", " ").title() for column in visible_columns]]
    for _, row in df.head(limit).iterrows():
        rows.append([_text(row.get(column)) for column in visible_columns])
    return rows


def _style_table(table, colors, header_bg="#e0f2fe"):
    from reportlab.platypus import TableStyle

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )


def _section_title(text: str, styles):
    from reportlab.platypus import Paragraph, Spacer

    return [Spacer(1, 10), Paragraph(text, styles["SectionTitle"]), Spacer(1, 5)]


def _metric_card(label: str, value: Any, colors, accent: str = "#2563eb", background: str = "#f8fafc"):
    from reportlab.platypus import Paragraph, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    styles = getSampleStyleSheet()
    label_style = ParagraphStyle(
        "MetricLabel",
        parent=styles["Normal"],
        fontSize=7,
        textColor=colors.HexColor("#64748b"),
        leading=9,
    )
    value_style = ParagraphStyle(
        "MetricValue",
        parent=styles["Normal"],
        fontSize=13,
        textColor=colors.HexColor("#0f172a"),
        leading=16,
        fontName="Helvetica-Bold",
    )
    card = Table(
        [[Paragraph(_text(label).upper(), label_style)], [Paragraph(_text(value), value_style)]],
        colWidths=[120],
    )
    card.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(background)),
                ("LINEBEFORE", (0, 0), (0, -1), 4, colors.HexColor(accent)),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#d8e2ef")),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return card


def _bar_chart(title: str, df: pd.DataFrame, label_column: str, value_column: str, colors):
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.shapes import Drawing, String

    drawing = Drawing(250, 145)
    drawing.add(String(4, 130, title, fontSize=8, fillColor=colors.HexColor("#0f172a")))
    chart = VerticalBarChart()
    chart.x = 25
    chart.y = 25
    chart.height = 90
    chart.width = 205
    if df.empty or label_column not in df.columns or value_column not in df.columns:
        data = [0]
        labels = ["None"]
    else:
        view = df.head(6).copy()
        data = pd.to_numeric(view[value_column], errors="coerce").fillna(0).astype(float).tolist()
        labels = [_text(value)[:12] for value in view[label_column].tolist()]
    chart.data = [data]
    chart.categoryAxis.categoryNames = labels
    chart.categoryAxis.labels.fontSize = 5.8
    chart.valueAxis.labels.fontSize = 6
    chart.bars[0].fillColor = colors.HexColor("#2563eb")
    drawing.add(chart)
    return drawing


def _pie_chart(title: str, labels: list[str], values: list[int], colors):
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.shapes import Drawing, String

    drawing = Drawing(250, 145)
    drawing.add(String(4, 130, title, fontSize=8, fillColor=colors.HexColor("#0f172a")))
    pie = Pie()
    pie.x = 62
    pie.y = 18
    pie.width = 105
    pie.height = 105
    pie.data = values if any(values) else [1]
    pie.labels = labels if any(values) else ["No alerts"]
    palette = ["#22c55e", "#ef4444", "#f59e0b", "#2563eb"]
    for index, color in enumerate(palette[: len(pie.data)]):
        pie.slices[index].fillColor = colors.HexColor(color)
    pie.slices.strokeColor = colors.white
    drawing.add(pie)
    return drawing


def generate_inventory_pdf_report(report: dict[str, Any], output_path: str | Path) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "TitleBlue",
            parent=styles["Title"],
            fontSize=21,
            leading=26,
            textColor=colors.HexColor("#0f766e"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            "Muted",
            parent=styles["Normal"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#475569"),
        )
    )
    styles.add(
        ParagraphStyle(
            "SectionTitle",
            parent=styles["Heading2"],
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=6,
            spaceAfter=2,
        )
    )

    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=28,
        leftMargin=28,
        topMargin=26,
        bottomMargin=26,
    )
    header = Table(
        [
            [Paragraph("AI Retail Inventory Intelligence Report", styles["TitleBlue"])],
            [
                Paragraph(
                    f"{_text(report.get('branch_label'))} | Generated {_text(report.get('generated_at'))} | Range {_text(report.get('date_range_label'))}",
                    styles["Muted"],
                )
            ],
            [Paragraph(_text(report.get("snapshot_note")), styles["Muted"])],
        ],
        colWidths=[520],
    )
    from reportlab.platypus import TableStyle

    header.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#dbeafe")),
                ("LINEBEFORE", (0, 0), (0, -1), 6, colors.HexColor("#0f766e")),
                ("LEFTPADDING", (0, 0), (-1, -1), 14),
                ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story = [header, Spacer(1, 12)]

    metrics = [
        _metric_card("Total Products", f"{int(report.get('total_products', 0)):,}", colors, "#2563eb", "#eff6ff"),
        _metric_card("Total Stock", f"{int(report.get('total_inventory_quantity', 0)):,}", colors, "#0f766e", "#ecfdf5"),
        _metric_card("Low Stock", f"{int(report.get('low_stock_count', 0)):,}", colors, "#dc2626", "#fff7f7"),
        _metric_card("Overstock", f"{int(report.get('overstock_count', 0)):,}", colors, "#d97706", "#fffbeb"),
        _metric_card("Transfers", f"{int(report.get('transfer_opportunity_count', 0)):,}", colors, "#0891b2", "#ecfeff"),
        _metric_card("High Priority", f"{int(report.get('high_priority_recommendations_count', 0)):,}", colors, "#7c3aed", "#f5f3ff"),
    ]
    metric_grid = Table([metrics[:3], metrics[3:]], colWidths=[168, 168, 168])
    story.extend([metric_grid, Spacer(1, 10)])

    chart_grid = Table(
        [
            [
                _bar_chart(
                    "Category-wise Inventory",
                    report.get("category_summary", pd.DataFrame()),
                    "category",
                    "stock_level",
                    colors,
                ),
                _bar_chart(
                    "Low-stock Count by Branch",
                    report.get("low_stock_by_branch", pd.DataFrame()),
                    "store_name",
                    "low_stock_count",
                    colors,
                ),
            ],
            [
                _bar_chart(
                    "Overstock Count by Branch",
                    report.get("overstock_by_branch", pd.DataFrame()),
                    "store_name",
                    "overstock_count",
                    colors,
                ),
                _pie_chart(
                    "Stock Status Distribution",
                    ["Healthy", "Low", "Overstock"],
                    [
                        int(report.get("healthy_stock_count", 0)),
                        int(report.get("low_stock_count", 0)),
                        int(report.get("overstock_count", 0)),
                    ],
                    colors,
                ),
            ],
        ],
        colWidths=[260, 260],
    )
    story.extend([chart_grid])

    sections = [
        (
            "Low Stock Items",
            report.get("low_stock_table", pd.DataFrame()),
            [
                "product_name",
                "store_name",
                "stock_level",
                "reorder_threshold",
                "shortage_quantity",
                "suggested_reorder_quantity",
                "priority",
                "ai_recommendation",
            ],
        ),
        (
            "Overstock Items",
            report.get("overstock_table", pd.DataFrame()),
            ["product_name", "store_name", "stock_level", "reorder_threshold", "surplus_quantity", "suggested_action"],
        ),
        (
            "Transfer / Alternative Availability",
            report.get("transfer_opportunities", pd.DataFrame()),
            [
                "product",
                "source_branch",
                "target_branch",
                "suggested_transfer_quantity",
                "alternative_product",
                "reason",
            ],
        ),
        (
            "Branch-wise Recommendations",
            report.get("branch_recommendations", pd.DataFrame()),
            ["store_name", "source_agent", "priority", "recommendation_type", "product_name", "action"],
        ),
        (
            "5 Agent Suggestions per Branch",
            report.get("agent_suggestions", pd.DataFrame()),
            ["store_name", "agent", "suggestion", "source"],
        ),
    ]

    for title, df, columns in sections:
        story.extend(_section_title(title, styles))
        table = Table(_table_rows(df, columns), repeatRows=1)
        _style_table(table, colors)
        story.append(table)

    story.extend([Spacer(1, 12), Paragraph("AI Retail Inventory Optimizer", styles["Muted"])])
    doc.build(story)
    return path
