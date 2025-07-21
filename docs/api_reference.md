# API Reference for the Agentic AI Platform

## Overview

This document provides a reference for the key classes, functions, and methods in the Agentic AI Platform for Evaluating Healthcare Pathways. The system processes MIMIC-IV data, builds a Neo4j knowledge graph, analyzes treatment pathways, and generates explanations using a multi-agent architecture. This reference is intended for developers maintaining or extending the platform, supporting modularity and supportability (3.3.4).

*Note*: This is a simplified reference. Use tools like `pdoc` or `sphinx` to generate detailed documentation.

## Modules

### src.agents.data_agent

**Class: `DataProcessingAgent`**

- **Description**: Manages data ingestion, cleaning, PROM extraction, and validation using LangGraph.
- **Key Methods**:
  - `__init__(config: Dict)`: Initializes with config; sets up LangGraph workflow.
  - `_ingest_node(state: DataState) -> DataState`: Fetches MIMIC-IV data with retries.
  - `_detect_issues_node(state: DataState) -> DataState`: Identifies missing values, outliers, bias.
  - `_clean_node(state: DataState) -> DataState`: Cleans data with rule-based adjustments.
  - `_prom_extract_node(state: DataState) -> DataState`: Extracts PROMs using spaCy.
  - `_validate_node(state: DataState) -> DataState`: Validates against thresholds.
  - `run(initial_state: DataState) -> Dict`: Executes LangGraph workflow.

**Class: `DataState` (Pydantic)**

- **Description**: Shared state for LangGraph workflow.
- **Fields**:
  - `raw_data: Dict[str, Any]`: Raw MIMIC-IV DataFrames.
  - `cleaned_data: Dict[str, Any]`: Cleaned DataFrames.
  - `prom_scores: List`: PROM results.
  - `validation_results: Dict`: Validation outcomes.
  - `issues_detected: List`: Detected issues.
  - `quality_score: float`: Data quality score.
  - `attempts: Dict`: Retry counters.
  - `flags: List`: Status flags (e.g., partial_success).

### src.agents.graph_agent

**Class: `GraphBuildingAgent`**

- **Description**: Constructs and indexes Neo4j knowledge graph with GraphRAG.
- **Key Methods**:
  - `__init__(config: Dict)`: Initializes with Neo4j and LLM config.
  - `connect() -> bool`: Connects to Neo4j with retries.
  - `build_graph(validated_data: Dict) -> Dict`: Builds nodes/relationships; indexes with GraphRAG.
  - `close()`: Closes Neo4j connection.

### src.agents.analysis_agent

**Class: `PathwayAnalysisAgent`**

- **Description**: Analyzes treatment pathways with EDA, hypothesis testing, and predictions.
- **Key Methods**:
  - `__init__(config: Dict)`: Initializes with Neo4j and LLM config.
  - `connect() -> bool`: Connects to Neo4j.
  - `analyze_pathways(query: str, session_state: Dict) -> Dict`: Performs EDA, stats, predictions, fairness.
  - `close()`: Closes Neo4j connection.

### src.agents.explanation_agent

**Class: `ExplanationAgent`**

- **Description**: Generates plain-language explanations, visualizations, and reports.
- **Key Methods**:
  - `__init__(config: Dict)`: Initializes with LLM config.
  - `generate_explanation(query: str, analysis_results: Dict, session_state: Dict) -> Dict`: Produces explanations and charts.
  - `handle_refinement(query: str, analysis_results: Dict, refinement: str) -> Dict`: Handles multi-turn queries.
  - `_calculate_confidence(analysis_results: Dict) -> float`: Computes confidence score.

### src.agents.coordinator

**Class: `WorkflowCoordinator`**

- **Description**: Orchestrates agents via MCP and A2A.
- **Key Methods**:
  - `__init__(config: Dict)`: Initializes all agents.
  - `check_graph_readiness() -> bool`: Checks Neo4j graph readiness.
  - `reset_data()`: Clears data directories.
  - `run_preprocessing() -> Dict`: Runs stages 1-5 (Data Processing, Graph Building).
  - `run_query(query: str, session_state: Dict) -> Dict`: Runs stages 6-7 (Analysis, Explanation).

**Class: `QueryState` (Pydantic)**

- **Description**: State for query mode.
- **Fields**:
  - `query: str`: User query.
  - `session_state: Dict`: Multi-turn session data.
  - `analysis_results: Dict`: Analysis outputs.
  - `explanation_results: Dict`: Explanation outputs.

### src.data.mimic_connector

**Class: `MIMICDataConnector`**

- **Description**: Fetches MIMIC-IV data via BigQuery (gcloud auth).
- **Key Methods**:
  - `__init__(config: Dict)`: Initializes with BigQuery config.
  - `connect() -> bool`: Connects to BigQuery.
  - `get_diabetes_patients(limit: int) -> pd.DataFrame`: Fetches diabetes patients.
  - `get_treatment_data(subject_ids: List) -> pd.DataFrame`: Fetches treatments.
  - `get_lab_results(subject_ids: List) -> pd.DataFrame`: Fetches labs.
  - `get_discharge_notes(subject_ids: List, limit_per_patient: int) -> pd.DataFrame`: Fetches notes.
  - `get_procedures(subject_ids: List) -> pd.DataFrame`: Fetches procedures.
  - `save_raw_data(data_dict: Dict, filepath: str) -> bool`: Saves to JSON.

### src.data.data_cleaner

**Class: `DataCleaner`**

- **Description**: Cleans MIMIC-IV data.
- **Key Methods**:
  - `__init__(config: Dict)`: Initializes with thresholds.
  - `clean_patient_data(patients: pd.DataFrame) -> pd.DataFrame`: Cleans patient data.
  - `clean_treatment_data(treatments: pd.DataFrame) -> pd.DataFrame`: Cleans treatments.
  - `clean_lab_results(labs: pd.DataFrame) -> pd.DataFrame`: Cleans labs.
  - `detect_bias(patients: pd.DataFrame) -> Dict`: Detects demographic bias.
  - `calculate_quality_score(patients: pd.DataFrame) -> float`: Computes quality score.

### src.data.prom_extractor

**Class: `PROMExtractor`**

- **Description**: Extracts PROMs from clinical notes.
- **Key Methods**:
  - `__init__(config: Dict)`: Initializes with spaCy.
  - `extract_proms(patients, notes, labs, treatments: pd.DataFrame) -> List[Dict]`: Extracts QoL/pain/mobility.

### src.data.validator

**Class: `DataValidator`**

- **Description**: Validates data against thresholds.
- **Key Methods**:
  - `__init__(config: Dict)`: Initializes with thresholds.
  - `validate_data(patients, cleaned_data, prom_scores) -> Dict`: Checks quality, coverage, fairness.

### src.graph.neo4j_driver

**Class: `Neo4jDriver`**

- **Description**: Manages Neo4j connections and queries.
- **Key Methods**:
  - `__init__(config: Dict)`: Initializes with Neo4j config.
  - `connect() -> bool`: Connects with retries.
  - `execute_query(query: str, params: Dict) -> List[Dict]`: Runs Cypher queries.
  - `create_index(label: str, property: str) -> bool`: Creates indexes.
  - `check_graph_readiness() -> Dict`: Validates graph readiness.
  - `close()`: Closes connection.

### src.graph.graphrag_indexer

**Class: `GraphRAGIndexer`**

- **Description**: Enhances graph with GraphRAG.
- **Key Methods**:
  - `__init__(config: Dict)`: Initializes with Neo4j/LLM.
  - `index_graph(graph_data: Dict) -> Dict`: Indexes entities/relationships.
  - `close()`: Closes Neo4j connection.

### src.analysis.stats_utils

**Class: `StatsUtils`**

- **Description**: Provides statistical analysis for pathways.
- **Key Methods**:
  - `__init__(config: Dict)`: Initializes with thresholds.
  - `calculate_correlations(df: pd.DataFrame, target_col: str, features: List[str]) -> Dict`: Computes correlations.
  - `perform_ttest(df, group_col, value_col, groups) -> List[Dict]`: Runs t-tests.
  - `perform_anova(df, group_col, value_col) -> Dict`: Runs ANOVA.
  - `calculate_shap_values(df, target_col, features) -> Dict`: Computes SHAP values.
  - `analyze_pathways(df, group_col, value_col) -> Dict`: Combines analyses.

### src.analysis.predictive_models

**Class: `PredictiveModels`**

- **Description**: Provides predictive modeling.
- **Key Methods**:
  - `__init__(config: Dict)`: Initializes with thresholds.
  - `predict_readmission_risk(df, features, target_col) -> Dict`: Predicts readmission risk.
  - `predict_treatment_response(df, treatment_col, outcome_col) -> Dict`: Predicts treatment response.

### src.analysis.fairness_metrics

**Class: `FairnessMetrics`**

- **Description**: Detects and mitigates bias.
- **Key Methods**:
  - `__init__(config: Dict)`: Initializes with bias fields.
  - `calculate_fairness_metrics(df, outcome_col) -> Dict`: Computes fairness metrics.
  - `reweight_data(df, field, outcome_col) -> pd.DataFrame`: Reweights data.

### src.visualization.viz_utils

**Class: `VisualizationUtils`**

- **Description**: Generates interactive charts.
- **Key Methods**:
  - `__init__(config: Dict)`: Initializes with config.
  - `create_chart(data, chart_type, title, annotations) -> str`: Generates Plotly chart.
  - `generate_dashboard_charts(analysis_results, query) -> List[str]`: Creates multiple charts.

### src.dashboard.app

**Function: `run_dashboard()`**

- **Description**: Runs the Streamlit AG-UI dashboard.

### src.dashboard.components.query_form

**Class: `QueryForm`**

- **Description**: Renders query input form.
- **Key Methods**:
  - `__init__()`: Initializes with example queries.
  - `render() -> Tuple[str, str]`: Renders query/refinement inputs.

### src.dashboard.components.chart_renderer

**Class: `ChartRenderer`**

- **Description**: Renders Plotly charts in Streamlit.
- **Key Methods**:
  - `__init__()`: Initializes.
  - `render(chart_html: str)`: Renders single chart.
  - `render_multiple(charts: List[str])`: Renders multiple charts.

### src.dashboard.components.report_generator

**Class: `ReportGenerator`**

- **Description**: Renders reports in Streamlit.
- **Key Methods**:
  - `__init__()`: Initializes.
  - `render(report_content: str, stakeholder: str)`: Renders single report.
  - `render_multiple(reports: Dict)`: Renders multiple reports.

### src.utils.logging_config

**Function: `setup_logging(config: Dict)`**

- **Description**: Configures JSON logging.

**Class: `JSONFormatter`**

- **Description**: Formats logs as JSON.

### src.utils.config_loader

**Function: `load_config(config_path: str) -> Dict`**

- **Description**: Loads YAML and .env configs.

### src.utils.json_validator

**Function: `validate_payload(payload: Dict, schema: Dict) -> bool`**

- **Description**: Validates A2A JSON payloads.

### src.utils.clinical_knowledge

**Class: `ClinicalKnowledge`**

- **Description**: Integrates clinical guidelines.
- **Key Methods**:
  - `__init__(config: Dict)`: Initializes with guidelines.
  - `validate_analysis(analysis_results, condition) -> Dict`: Validates against guidelines.
  - `enrich_explanation(explanation, analysis_results, condition) -> str`: Adds clinical context.

## Version

- **Version**: 1.0.0
- **Last Updated**: July 20, 2025
