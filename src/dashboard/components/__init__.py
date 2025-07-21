
# src/dashboard/components/__init__.py
"""
Components package for the AG-UI dashboard in the Agentic AI Platform.
Contains reusable Streamlit components for:
- Query input form for user interactions
- Chart rendering for interactive visualizations
- Report generation for stakeholder-specific outputs
"""

from .query_form import QueryForm
from .chart_renderer import ChartRenderer
from .report_generator import ReportGenerator

__all__ = [
    "QueryForm",
    "ChartRenderer",
    "ReportGenerator",
]
