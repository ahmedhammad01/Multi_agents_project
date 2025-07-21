
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.agents.analysis_agent import PathwayAnalysisAgent
from src.utils.config_loader import load_config

# Mock data for testing
@pytest.fixture
def config():
    """Load test configuration"""
    config = load_config()
    config["llm"]["model"] = "gpt-4o"  # Test model
    return config

@pytest.fixture
def mock_query_data():
    """Mock query results from Neo4j"""
    return [
        {"patient_id": 1, "treatment_type": "insulin", "qol_score": 70.0, "age": 50, "gender": "Male"},
        {"patient_id": 2, "treatment_type": "metformin", "qol_score": 65.0, "age": 60, "gender": "Female"},
        {"patient_id": 3, "treatment_type": "insulin", "qol_score": 75.0, "age": 55, "gender": "Male"}
    ]

@pytest.fixture
def analysis_agent(config):
    """Initialize Pathway Analysis Agent"""
    return PathwayAnalysisAgent(config)

def test_connect(analysis_agent):
    """Test Neo4j connection"""
    with patch("neo4j.GraphDatabase.driver") as mock_driver:
        mock_session = MagicMock()
        mock_driver.return_value.session.return_value.__enter__.return_value = mock_session
        assert analysis_agent.connect()
        mock_driver.assert_called_with(
            analysis_agent.neo4j_uri,
            auth=(analysis_agent.neo4j_user, analysis_agent.neo4j_password),
            max_connection_retries=analysis_agent.config["neo4j"]["max_retries"]
        )
        mock_session.run.assert_called_with("MATCH (n) RETURN n LIMIT 1")

def test_analyze_pathways(analysis_agent, mock_query_data):
    """Test pathway analysis"""
    with patch("neo4j.GraphDatabase.driver") as mock_driver:
        mock_session = MagicMock()
        mock_driver.return_value.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value.data.return_value = mock_query_data
        
        with patch.object(analysis_agent.llm, "__call__", side_effect=[
            "MATCH (p:Patient)-[:RECEIVED_TREATMENT]->(t:Treatment) RETURN p, t",  # Cypher query
            "Hypothesis 1\nHypothesis 2"  # Hypotheses
        ]):
            result = analysis_agent.analyze_pathways("Test query")
        
        assert result["status"] == "success"
        assert len(result["hypotheses"]) == 2
        assert len(result["pathways"]) == 1  # Significant pathways (insulin vs. metformin)
        assert "fairness_metrics" in result
        assert "correlations" in result
        assert len(result["predictions"]) == 1  # One prediction

def test_analyze_pathways_empty_query(analysis_agent):
    """Test analysis with empty query result"""
    with patch("neo4j.GraphDatabase.driver") as mock_driver:
        mock_session = MagicMock()
        mock_driver.return_value.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value.data.return_value = []
        
        result = analysis_agent.analyze_pathways("Empty query")
        
        assert result["status"] == "partial_success"
        assert "Empty query result" in result["error"]

def test_analyze_pathways_neo4j_failure(analysis_agent):
    """Test analysis with Neo4j connection failure"""
    with patch.object(analysis_agent, "connect", return_value=False):
        result = analysis_agent.analyze_pathways("Test query")
        assert result["status"] == "partial_success"
        assert "Neo4j connection failed" in result["error"]

def test_close(analysis_agent):
    """Test closing Neo4j connection"""
    with patch("neo4j.GraphDatabase.driver") as mock_driver:
        mock_driver_instance = MagicMock()
        analysis_agent.driver = mock_driver_instance
        analysis_agent.close()
        mock_driver_instance.close.assert_called_once()
        assert analysis_agent.driver is None
