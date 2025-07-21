
# src/data/__init__.py
"""
Data package for the Agentic AI Platform.
Contains utilities for:
- Connecting to MIMIC-IV (BigQuery)
- Cleaning data (missing values, standardization, outliers)
- Extracting PROMs (NLP with spaCy)
- Validating data (thresholds, fairness)
"""

from .mimic_connector import MIMICDataConnector
from .data_cleaner import DataCleaner
from .prom_extractor import PROMExtractor
from .validator import DataValidator

__all__ = [
    "MIMICDataConnector",
    "DataCleaner",
    "PROMExtractor",
    "DataValidator",
]
