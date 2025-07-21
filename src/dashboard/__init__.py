
# src/dashboard/__init__.py
"""
Dashboard package for the Agentic AI Platform.
Contains utilities for:
- Streamlit-based AG-UI dashboard (query input, chart rendering, report generation)
- Reusable components for interactive visualizations and user interaction
"""

from .app import run_dashboard
from .components.query_form import QueryForm
from .components.chart_renderer import ChartRenderer
from .components.report_generator import ReportGenerator

__all__ = [
    "run_dashboard",
    "QueryForm",
    "ChartRenderer",
    "ReportGenerator",
]
