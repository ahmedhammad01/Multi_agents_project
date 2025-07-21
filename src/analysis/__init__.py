
# src/analysis/__init__.py
"""
Analysis package for the Agentic AI Platform.
Contains utilities for:
- Statistical analysis (SciPy, SHAP for pathway evaluation)
- Predictive modeling (scikit-learn for risk forecasting)
- Fairness metrics (AIF360 for bias detection)
"""

from .stats_utils import StatsUtils
from .predictive_models import PredictiveModels
from .fairness_metrics import FairnessMetrics

__all__ = [
    "StatsUtils",
    "PredictiveModels",
    "FairnessMetrics",
]
