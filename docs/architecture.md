# System Architecture for the Agentic AI Platform

## Overview

The Agentic AI Platform for Evaluating Healthcare Pathways is a multi-agent system designed to analyze treatment pathways for chronic conditions (diabetes and COPD) using the MIMIC-IV dataset. It processes ~10,000-15,000 patient records, builds a Neo4j knowledge graph, performs statistical and predictive analysis, and generates plain-language recommendations. The system uses open-source tools (LangChain, Neo4j, spaCy, AIF360, Streamlit) to ensure modularity, scalability (<15 min preprocessing, <10s queries), and ethical compliance (PII masking, fairness). This document describes the architecture, agents, workflows, and integration points, aligning with requirements for maintainability (3.3.4) and usability (3.3.1).

## System Components

The platform consists of four primary agents, orchestrated by a Workflow Coordinator, and supported by utility modules and a Streamlit-based AG-UI dashboard.

### 1. Data Processing Agent
- **Role**: Ingests, cleans, extracts PROMs, and validates MIMIC-IV data.
- **Implementation**: Uses LangGraph for a dynamic workflow with nodes for ingestion, issue detection, cleaning, PROM extraction, and validation.
- **Key Features**:
  - Ingests ~10,000-15,000 records via BigQuery (`gcloud` auth).
  - Cleans data (missing values, outliers, standardization).
  - Extracts PROMs using spaCy (en_core_web_sm) with rule-based adjustments.
  - Validates against thresholds (quality ≥0.8, PROM confidence ≥0.7).
  - Retries, fallbacks, and loops for robustness.
- **Dependencies**: `pandas`, `spacy`, `aif360`, `langgraph`.

### 2. Graph Building Agent
- **Role**: Constructs and indexes a Neo4j knowledge graph.
- **Implementation**: Builds nodes (Patient, Treatment, Outcome) and relationships (e.g., RECEIVED_TREATMENT) using `neo4j-driver`; enhances with GraphRAG for contextual queries.
- **Key Features**:
  - Creates indexed nodes/edges; generates summaries.
  - Validates graph readiness (node_count ≥15,000, freshness <7 days).
  - Retries for Neo4j connections.
- **Dependencies**: `neo4j`, `langchain`.

### 3. Pathway Analysis Agent
- **Role**: Query-triggered analysis of treatment pathways.
- **Implementation**: Parses queries to Cypher; performs EDA, hypothesis testing, predictive modeling, and fairness checks.
- **Key Features**:
  - Uses GraphRAG for retrieval; SciPy for stats (t-tests, ANOVA).
  - Predicts risks (e.g., readmission via logistic regression).
  - Ensures fairness with AIF360 (disparate impact ≥0.8).
  - Supports multi-turn queries.
- **Dependencies**: `scipy`, `aif360`, `sklearn`, `langchain`.

### 4. Explanation Agent
- **Role**: Generates plain-language explanations and reports.
- **Implementation**: Uses LangChain LLM for narratives; Plotly for visualizations.
- **Key Features**:
  - Produces PROM-aligned recommendations and stakeholder reports (clinician/admin).
  - Includes confidence scores and bias warnings.
  - Supports multi-turn refinements.
- **Dependencies**: `langchain`, `plotly`.

### 5. Workflow Coordinator
- **Role**: Orchestrates agents via MCP and A2A.
- **Implementation**: Manages preprocessing (stages 1-5) and query modes (stages 6-7); uses JSON payloads with jsonschema validation.
- **Key Features**:
  - Mode switching based on graph readiness.
  - Handles reset and preprocessing triggers.
- **Dependencies**: `langchain`, `jsonschema`.

### 6. AG-UI Dashboard
- **Role**: Provides interactive query interface and visualization.
- **Implementation**: Streamlit app with components for query input, chart rendering, and report generation.
- **Key Features**:
  - Natural language query input with example prompts.
  - Interactive charts (bar, line, heatmap, network).
  - Downloadable reports.
- **Dependencies**: `streamlit`, `plotly`.

## Workflow

The system operates in two modes: **Preprocess Mode** (stages 1-5) and **Query Mode** (stages 6-7), managed by the Workflow Coordinator.

### Preprocess Mode
1. **Data Ingestion (Data Processing Agent)**:
   - Fetch ~10,000-15,000 records from MIMIC-IV (BigQuery, `gcloud` auth).
   - Save raw data to `data/raw/mimic_raw.json`.
2. **Issue Detection and Cleaning (Data Processing Agent)**:
   - Detect missing values, outliers, bias; clean with rules (e.g., dropna, standardize).
   - Save cleaned data to `data/processed/mimic_clean.json`.
3. **PROM Extraction (Data Processing Agent)**:
   - Extract QoL/pain/mobility from notes (spaCy, rules).
   - Adjust with labs/treatments; ensure confidence ≥0.7.
4. **Validation (Data Processing Agent)**:
   - Check thresholds (records ≥10,000, quality ≥0.8, lab coverage ≥0.05).
   - Loop (max 3) if failed; flag partial success.
5. **Graph Building (Graph Building Agent)**:
   - Build Neo4j nodes/relationships; index with GraphRAG.
   - Validate readiness (nodes ≥15,000).

### Query Mode
6. **Pathway Analysis (Pathway Analysis Agent)**:
   - Parse query to Cypher; retrieve data with GraphRAG.
   - Perform EDA, hypothesis testing, predictions, fairness checks.
   - Output JSON insights (<3s).
7. **Explanation (Explanation Agent)**:
   - Generate NL explanations, visualizations, reports.
   - Support multi-turn refinements (<5s).

### Orchestration
- **MCP**: Sequences tasks (Data → Graph → Analysis → Explanation).
- **A2A**: JSON payloads for inter-agent communication (validated with jsonschema).
- **Mode Switching**: Readiness check (Neo4j counts/freshness); triggers preprocessing if needed.

## Integration Points

- **Data Flow**:
  - MIMIC-IV → [LangGraph: Ingest/Detect/Clean/PROM/Validate] → Neo4j → [Query Parse/EDA/Analysis] → [Explanation/Viz/Reports] → AG-UI.
- **A2A Payloads**:
  - Data Processing → Graph Building: Cleaned data JSON.
  - Graph Building → Analysis: Graph metadata.
  - Analysis → Explanation: Insights JSON.
- **External Services**:
  - BigQuery: Data ingestion (gcloud auth).
  - Neo4j: Graph storage/querying.
  - LLM (Grok/OpenAI): Query parsing, explanations.

## Ethical Considerations

- **PII Masking**: Hashed IDs in data processing.
- **Fairness**: AIF360 bias checks; warnings in reports.
- **Auditability**: JSON logs (`logs/app.log`) for transparency.
- **Consent**: Notes in explanations (consult physician).

## Performance

- **Preprocessing**: <15 min on 16GB RAM (stages 1-5).
- **Queries**: <10s (Analysis <3s, Explanation <5s).
- **Scalability**: Supports up to 200,000 records via batching/caching.

## Dependencies

- See `requirements.txt` for full list (LangChain, Neo4j, spaCy, AIF360, Plotly, Streamlit).
- BigQuery: Requires `gcloud auth application-default login`.
- spaCy: `en_core_web_sm` model installed separately.

## Testing

- Pytest suite in `tests/` (>80% coverage).
- Run: `pytest tests/ --cov=src --cov-report=html`.

## Version

- **Version**: 1.0.0
- **Last Updated**: July 20, 2025
