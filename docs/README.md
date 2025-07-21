# Agentic AI Platform for Evaluating Healthcare Pathways

## Overview

This project is an agentic AI platform designed to evaluate treatment pathways for chronic conditions (diabetes and COPD) using the MIMIC-IV dataset. It employs a multi-agent architecture with four primary agents: Data Processing, Graph Building, Pathway Analysis, and Explanation. The system processes ~10,000-15,000 patient records, builds a Neo4j knowledge graph, performs statistical analysis, and generates plain-language recommendations, ensuring fairness and PROM (Patient-Reported Outcome Measures) integration. Built for an MSc project (15-week timeline), it uses open-source tools (LangChain, Neo4j, spaCy, AIF360, Streamlit) and supports <15 min preprocessing and <10s query responses.

## Features

- **Data Processing Agent**: Ingests, cleans, extracts PROMs, and validates MIMIC-IV data using LangGraph for dynamic workflows.
- **Graph Building Agent**: Constructs a Neo4j knowledge graph with GraphRAG indexing.
- **Pathway Analysis Agent**: Performs EDA, hypothesis testing, predictive modeling, and fairness checks.
- **Explanation Agent**: Generates plain-language explanations, visualizations, and stakeholder-specific reports.
- **AG-UI Dashboard**: Streamlit-based interface for natural language queries and interactive results.
- **Ethical Compliance**: PII masking, bias detection (AIF360), and audit trails.

## Requirements

- Python 3.8+
- Dependencies: See `requirements.txt`
- BigQuery access: `gcloud auth application-default login` for MIMIC-IV
- Neo4j Community Edition
- spaCy model: `python -m spacy download en_core_web_sm`

## Setup Instructions

1. **Clone Repository**:
   ```bash
   git clone <repo_url>
   cd healthcare_pathway_project


Create Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt
python -m spacy download en_core_web_sm


Configure Environment:

Copy config/.env.example to config/.env
Update config/.env with:
GOOGLE_CLOUD_PROJECT: Your BigQuery project ID
NEO4J_PASSWORD: Your Neo4j password
LLM_API_KEY: Your Grok/OpenAI API key




Run Preprocessing:
python main.py --preprocess


Launch Dashboard:
streamlit run src/dashboard/app.py


Run Tests:
./scripts/run_tests.sh



Usage

Preprocessing: Run python main.py --preprocess to ingest MIMIC-IV data, clean it, extract PROMs, validate, and build the graph.
Reset Data: Use python main.py --reset to clear and re-run preprocessing.
Querying: Access the Streamlit dashboard (http://localhost:8501) to submit queries (e.g., "What are the main risk factors for diabetes complications?").
Refinements: Add refinements (e.g., "Focus on patients over 60") for multi-turn interactions.
Reports: Download clinician/admin reports from the dashboard.

Project Structure

src/: Source code (agents, data, graph, analysis, visualization, dashboard, utils)
data/: Raw and processed data (excluded from git)
config/: Configuration files (config.yaml, .env)
tests/: Pytest unit/integration tests
docs/: Documentation (README, user guide, architecture)
scripts/: Automation scripts (setup, tests)

Testing
Run pytest for >80% coverage:
pytest tests/ --cov=src --cov-report=html

Documentation

docs/user_guide.md: Guide for clinicians using the dashboard
docs/architecture.md: System design and workflows
docs/api_reference.md: Auto-generated code reference

License
MIT License```