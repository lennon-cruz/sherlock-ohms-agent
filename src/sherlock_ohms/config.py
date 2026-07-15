"""Corpus sources for the two report collections. Both are openly licensed for reuse."""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

MODEL = "gpt-4o-mini"

CORPORA = {
    "fintech": {
        "collection_name": "fintech_report",
        "pdf_path": DATA_DIR / "fintech_report.pdf",
        "url": (
            "https://www.greenfinanceplatform.org/sites/default/files/downloads/resource/"
            "Fintech%20and%20the%20Future%20of%20Finance_World%20Bank.pdf"
        ),
        "title": "Fintech and the Future of Finance: Market and Policy Implications (World Bank)",
        "license": "CC BY 3.0 IGO",
    },
    "telecom": {
        "collection_name": "telecom_report",
        "pdf_path": DATA_DIR / "telecom_report.pdf",
        "url": (
            "https://www.itu.int/itu-d/reports/statistics/wp-content/uploads/sites/5/2024/11/"
            "2402588_1e_Measuring-digital-development-Facts-and-Figures-2024_v4.pdf"
        ),
        "title": "Measuring Digital Development: Facts and Figures 2024 (ITU)",
        "license": "CC BY-NC-SA 3.0 IGO",
    },
}
