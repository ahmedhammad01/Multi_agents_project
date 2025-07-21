
import pytest
from unittest.mock import patch, MagicMock
from src.agents.coordinator import WorkflowCoordinator, QueryState
from src.utils.config_loader import load_config
import os
from datetime import datetime

# Mock data for testing
@pytest.fixture
def config():
    """Load test configuration"""
    config = load_config()
    config["neo4j"]["node_count_min"] = 10
    config["neo4j"]["edge_count_min"] = 5
    config["data_processing"]["min_records"] = 50
    config["data_processing"]["quality_score_min"] = 0.7
    return config

@pytest.fixture
def mock_data_result():
    """Mock Data Processing Agent result"""
    return {
        "status": "success",
        "quality_score": 0.85,
        "issues": [],
        "record_counts": {"patients": 100},
        "cleaned_data": {
            "patients": [{"subject_id": 1, "age": 50, "gender": "Male"}],
            "prom_scores": [{"subject_id": 1, "quality_of_life_score": 70.0, "confidence": 0.7}]
        }
    }

@pytest.fixture
def mock_graph_result():
    """Mock Graph Building Agent result"""
    return {
        "status": "success",
        "nodes": 15,
        "relationships": 10
    }

@pytest.fixture
def mock_analysis_result():
    """Mock Pathway Analysis Agent result"""
    return {
        "status": "success",
        "query": "Test query",
        "hypotheses": ["Hypothesis 1"],
        "pathways": [{"treatment": "insulin", "impact": 0.15, "p_value": 0.01}],
        "predictions": [{"risk_prediction": 0.23}],
        "fairness_metrics": {"gender": {"disparate_impact": 0.85}}
    }

@pytest.fixture
def mock_explanation_result():
    """Mock Explanation Agent result"""
    return {
        "status": "success",
        "query": "Test query",
        "explanation": "Insulin effective.",
        "confidence": 85,
        "visualization": "<html>chart</html>",
        "reports": {"clinician": "Clinical summary", "admin": "Admin summary"}
    }

@pytest.fixture
def coordinator(config):
    """Initialize Workflow Coordinator"""
    return WorkflowCoordinator(config)

def test_check_graph_readiness(coordinator):
    """Test graph readiness check"""
    with patch("neo4j.GraphDatabase.driver") as mock_driver:
        mock_session = MagicMock()
        mock_driver.return_value.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value.single.return_value = {
            "node_count": 15,
            "rel_count": 10,
            "last_updated": datetime.now()
        }
        assert coordinator.check_graph_readiness()
        mock_session.run.assert_called_once()

def test_check_graph_readiness_failure(coordinator):
    """Test graph readiness with insufficient nodes"""
    with patch("neo4j.GraphDatabase.driver") as mock_driver:
        mock_session = MagicMock()
        mock_driver.return_value.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value.single.return_value = {
            "node_count": 5,
            "rel_count": 2,
            "last_updated": datetime.now()
        }
        assert not coordinator.check_graph_readiness()

def test_reset_data(coordinator):
    """Test data reset"""
    with patch("shutil.rmtree") as mock_rmtree, patch("os.makedirs") as mock_makedirs:
        coordinator.reset_data()
        mock_rmtree.assert_called()
        mock_makedirs.assert_called()

def test_run_preprocessing(coordinator, mock_data_result, mock_graph_result):
    """Test preprocessing pipeline"""
    with patch.object(coordinator.data_agent, "run", return_value=mock_data_result), \
         patch.object(coordinator.graph_agent, "build_graph", return_value=mock_graph_result):
        result = coordinator.run_preprocessing()
        assert result["status"] == "success"
        assert result["quality_score"] == 0.85
        assert result["graph_nodes"] == 15
        assert result["graph_relationships"] == 10

def test_run_preprocessing_failure(coordinator):
    """Test preprocessing with failure"""
    with patch.object(coordinator.data_agent, "run", return_value={"status": "failed", "error": "Data error"}):
        result = coordinator.run_preprocessing()
        assert result["status"] == "failed"
        assert "Data error" in result["error"]

def test_run_query(coordinator, mock_analysis_result, mock_explanation_result):
    """Test query pipeline"""
    with patch.object(coordinator, "check_graph_readiness", return_value=True), \
         patch.object(coordinator.analysis_agent, "analyze_pathways", return_value=mock_analysis_result), \
         patch.object(coordinator.explanation_agent, "generate_explanation", return_value=mock_explanation_result):
        result = coordinator.run_query("Test query")
        assert result["status"] == "success"
        assert result["query"] == "Test query"
        assert result["analysis"] == mock_analysis_result
        assert result["explanation"] == mock_explanation_result

def test_run_query_preprocess_needed(coordinator, mock_data_result, mock_graph_result, mock_analysis_result, mock_explanation_result):
    """Test query with preprocessing needed"""
    with patch.object(coordinator, "check_graph_readiness", return_value=False), \
         patch.object(coordinator.data_agent, "run", return_value=mock_data_result), \
         patch.object(coordinator.graph_agent, "build_graph", return_value=mock_graph_result), \
         patch.object(coordinator.analysis_agent, "analyze_pathways", return_value=mock_analysis_result), \
         patch.object(coordinator.explanation_agent, "generate_explanation", return_value=mock_explanation_result):
        result = coordinator.run_query("Test query")
        assert result["status"] == "success"
        assert result["query"] == "Test query"
