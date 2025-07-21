
# src/__init__.py
"""
Agentic AI Platform for Evaluating Healthcare Pathways
This package contains the core implementation of the multi-agent system for
processing MIMIC-IV data, building knowledge graphs, analyzing pathways, and
generating explanations.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Ensure spaCy model is available
try:
    import spacy
    spacy.load("en_core_web_sm")
except ImportError:
    import logging
    logging.warning("⚠️ spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
