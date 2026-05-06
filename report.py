#!/usr/bin/env python3
"""
`report.py`

Build a colour-coded PDF report from scored results.
The colours code a status / meaning as follows:
- GREEN. As robustly verified as the system gets. Probably all good, lowest priority for any manual checks.
- AMBER. Partial match. Worth checking manually
- RED. Not found. higher likelihood of error
- GREY. No API match at all
- PURPLE. API process skipped due to missing required fields.
- PINK. Unexpected API / scoring error.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

logger = logging.getLogger(__name__)


__author__ = "Mark Gotham"


# -----------------------------------------------------------------------------

# Palette


GREEN  = colors.HexColor("#d4edda")
AMBER  = colors.HexColor("#fff3cd")
RED    = colors.HexColor("#f8d7da")
GREY   = colors.HexColor("#e2e3e5")
PURPLE = colors.HexColor("#e8d5f5")
PINK   = colors.HexColor("#fce4ec")

GREEN_DARK  = colors.HexColor("#155724")
AMBER_DARK  = colors.HexColor("#856404")
RED_DARK    = colors.HexColor("#721c24")
GREY_DARK   = colors.HexColor("#383d41")
PURPLE_DARK = colors.HexColor("#4b1a6e")
PINK_DARK   = colors.HexColor("#880e4f")

# Labels in descending priority order (used for sorting and summary counts).
# TODO: if a new status is added to status(), add it here too.
STATUS_PRIORITY = ["FLAG", "ERROR", "INCOMPLETE", "CHECK", "OK", "NOT FOUND"]


def status(
        score, amber: float, red: float, skipped: bool = False, error: bool = False
) -> tuple:
    """
    Given one result (score),
    return status in the form of a
    (label, bg_colour, fg_colour) tuple.
    """
    if skipped:        return "INCOMPLETE", PURPLE, PURPLE_DARK
    if error:          return "ERROR",      PINK,   PINK_DARK
    if score is None:  return "NOT FOUND",  GREY,   GREY_DARK
    if score >= amber: return "OK",         GREEN,  GREEN_DARK
    if score >= red:   return "CHECK",      AMBER,  AMBER_DARK
    return                    "FLAG",       RED,    RED_DARK


# -----------------------------------------------------------------------------

# Helpers


def para(text, style) -> Paragraph:
    """Wrap *text* in a ReportLab `Paragraph` object, escaping HTML special characters."""
    safe = escape(str(text)) if text else "—"
    return Paragraph(safe, style)


def score_bar(score) -> str:
    """Simple text sparkline for the score (0–10 blocks)."""
    if score is None:
        return "n/a"
    filled = min(round(score * 10), 10)   # clamp to [0, 10]
    return "█" * filled + "░" * (10 - filled) + f"  {score:.2f}"


def _result_status(r: dict, amber: float, red: float) -> tuple:
    """Compute status for a result dict exactly once."""
    return status(
        r["score"],
        amber,
        red,
        skipped=r.get("skipped", False),
        error="error" in r,
    )


def _venue(entry: dict) -> str:
    return entry.get("journal") or entry.get("booktitle") or "—"


def _format_authors(authors: list[str], max_shown: int = 4) -> str:
    if not authors:
        return "—"
    shown = authors[:max_shown]
    suffix = " …" if len(authors) > max_shown else ""
    return "; ".join(shown) + suffix


# -----------------------------------------------------------------------------

# Main builder

def build_report(results: list[dict], out_path: Path, amber: float, red: float) -> None:
    """
    Main report builder.
    Design choices include the colours (as reported elsewhere) and
    deriving column widths from the document's live text width.
    This way, widths stay correct if margins or page size change.
    """
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm,
    )

    text_w = doc.width   # A4 minus margins

    #  Styles 
    mono   = ParagraphStyle(
        "mono",   fontName="Courier",           fontSize=8,  leading=11)
    small  = ParagraphStyle(
        "small",  fontName="Helvetica",         fontSize=8,  leading=11)
    smallB = ParagraphStyle(
        "smallB", fontName="Helvetica-Bold",    fontSize=8,  leading=11)
    title  = ParagraphStyle(
        "title",  fontName="Helvetica-Bold",    fontSize=16, leading=20, spaceAfter=4)
    sub    = ParagraphStyle(
        "sub",    fontName="Helvetica",         fontSize=10, leading=14, textColor=colors.HexColor("#555555")
    )
    warn   = ParagraphStyle(
        "warn",   fontName="Helvetica-Oblique", fontSize=7.5, leading=10, textColor=PURPLE_DARK
    )

    story = []

    #  Header 
    story.append(para("Reference Quality Report", title))
    story.append(para(
        f"Generated: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  |  "
        f"Thresholds: amber ≥ {amber}  red &lt; {red}  "
        f"(score = weighted avg of DOI·0.4 + title·0.3 + authors·0.15 + year·0.1 + venue·0.05)",
        sub,
    ))
    story.append(Spacer(1, 0.4*cm))

    #  Incomplete-entry summary block 
    incomplete = [r for r in results if r.get("missing_fields")]
    if incomplete:
        lines = ["<b>Entries with missing required fields:</b>"]
        for r in incomplete:
            key = escape(r["entry"].get("ID", "?"))
            missing = r.get("missing_fields", [])
            missing_str = escape(", ".join(missing)) if missing else "unknown"
            skipped_note = " [API skipped]" if r.get("skipped") else ""
            lines.append(f"  • {key}: missing {missing_str}{skipped_note}")

        story.append(Paragraph("<br/>".join(lines), warn))
        story.append(Spacer(1, 0.3*cm))

    # Summary counts (compute status once per result to avoid repeated calls).
    statuses = [_result_status(r, amber, red) for r in results]
    counts   = {label: 0 for label in STATUS_PRIORITY}
    for label, _, _ in statuses:
        counts[label] += 1

    summary_data = [
        # ... With pretty icons no less ... ;)
        ["Total", "✓ OK", "⚠ CHECK", "✗ FLAG", "? NOT FOUND", "⊘ INCOMPLETE", "! ERROR"],
        [
            str(len(results)),
            str(counts["OK"]),
            str(counts["CHECK"]),
            str(counts["FLAG"]),
            str(counts["NOT FOUND"]),
            str(counts["INCOMPLETE"]),
            str(counts["ERROR"]),
        ],
    ]
    col_w = [text_w / 7] * 7
    summary_table = Table(summary_data, colWidths=col_w)
    summary_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#343a40")),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("BACKGROUND",  (1, 1), (1, 1),  GREEN),
        ("BACKGROUND",  (2, 1), (2, 1),  AMBER),
        ("BACKGROUND",  (3, 1), (3, 1),  RED),
        ("BACKGROUND",  (4, 1), (4, 1),  GREY),
        ("BACKGROUND",  (5, 1), (5, 1),  PURPLE),
        ("BACKGROUND",  (6, 1), (6, 1),  PINK),
        ("FONTNAME",    (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("BOX",         (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID",   (0, 0), (-1, -1), 0.3, colors.lightgrey),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.6*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    story.append(Spacer(1, 0.3*cm))

    # Per-reference cards
    # Sort order as per `STATUS_PRIORITY` = ["FLAG", "ERROR", "INCOMPLETE", "CHECK", "OK", "NOT FOUND"]
    # Status already computed in `statuses` (zip to preserve pairing).
    priority = {label: i for i, label in enumerate(STATUS_PRIORITY)}
    sorted_pairs = sorted(
        zip(statuses, results),
        key=lambda pair: priority[pair[0][0]],
    )

    for (label, bg, fg), r in sorted_pairs:
        entry   = r["entry"]
        score   = r["score"]
        hit     = r["best_hit"]
        comps   = r["components"]
        skipped = r.get("skipped", False)
        missing = r.get("missing_fields", [])
        err_msg = r.get("error")

        key     = entry.get("ID",     "?")
        e_title = entry.get("title",  "—")
        e_auth  = entry.get("author", "—")
        e_year  = entry.get("year",   "—")
        e_venue = _venue(entry)
        e_doi   = entry.get("doi",    "")

        if skipped:
            bar_text = "skipped (incomplete)"
        elif err_msg:
            bar_text = "error"
        else:
            bar_text = score_bar(score)

        # Card header
        header = Table(
            [[para(f"[{label}]  {key}", smallB), para(bar_text, mono)]],
            colWidths=[text_w * 0.76, text_w * 0.24],
        )
        header.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, -1), bg),
            ("TEXTCOLOR",    (0, 0), (-1, -1), fg),
            ("LEFTPADDING",  (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING",   (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ]))
        story.append(header)

        # Skipped entries: compact notice, no comparison grid
        if skipped:
            missing_str = ", ".join(missing) if missing else "unknown"
            notice = Table(
                [[para(
                    f"Missing fields: {missing_str}  — API check skipped. "
                    "Treat as FLAG until fields are supplied.",
                    warn,
                )]],
                colWidths=[text_w],
            )
            notice.setStyle(TableStyle([
                ("BACKGROUND",   (0, 0), (-1, -1), colors.HexColor("#f3e8ff")),
                ("LEFTPADDING",  (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING",   (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
                ("BOX",          (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            story.append(notice)
            story.append(Spacer(1, 0.35*cm))
            continue

        # Error entries: show error message, no comparison grid
        if err_msg:
            notice = Table(
                [[para(f"Scoring error: {err_msg}", warn)]],
                colWidths=[text_w],
            )
            notice.setStyle(TableStyle([
                ("BACKGROUND",   (0, 0), (-1, -1), colors.HexColor("#fce4ec")),
                ("LEFTPADDING",  (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING",   (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
                ("BOX",          (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            story.append(notice)
            story.append(Spacer(1, 0.35*cm))
            continue

        # Detail grid: bib vs best API hit
        def _row(field_label: str, bib_val: str, api_val, comp_key: str = "") -> list:
            comp_str = f"  [{comps[comp_key]:.2f}]" if comp_key and comp_key in comps else ""
            return [
                para(field_label, smallB),
                para(bib_val or "—", small),
                para((api_val or "—") + comp_str, small),
            ]

        # Truncate both sides of authors to the same limit for a fair visual.
        bib_authors_display = _format_authors(
            [a.strip() for a in e_auth.replace(" and ", ";").split(";") if a.strip()]
        )
        api_authors_display = _format_authors(hit["authors"]) if hit else "—"

        detail_data = [
            [para("Field", smallB), para("In .bib", smallB), para("Best API hit", smallB)],
            _row("Title", e_title, hit["title"] if hit else None, "title_sim"),
            [para("Author", smallB),
             para(bib_authors_display, small),
             para(api_authors_display +
                  (f"  [{comps['author_sim']:.2f}]" if "author_sim" in comps else ""), small)],
            _row("Year", e_year, hit["year"] if hit else None, "year_match"),
            _row("Venue", e_venue, hit["venue"] if hit else None, "journal_sim"),
            _row("DOI", e_doi or "none", hit.get("source", "") if hit else None),
        ]
        if missing:
            missing_str = ", ".join(missing)
            detail_data.append([
                para("⚠ Missing", smallB),
                para(missing_str, warn),
                para("—", small),
            ])
        col_w_detail = [text_w * 0.147, text_w * 0.441, text_w * 0.412]
        detail = Table(detail_data, colWidths=col_w_detail)
        detail.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("LEADING", (0, 0), (-1, -1), 10),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
        ]))
        story.append(detail)
        story.append(Spacer(1, 0.35 * cm))

    try:
        doc.build(story)
    except Exception as exc:
        raise RuntimeError(f"Failed to write PDF to {out_path}: {exc}") from exc
