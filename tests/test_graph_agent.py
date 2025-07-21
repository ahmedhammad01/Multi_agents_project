
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.agents.graph_agent import GraphBuildingAgent
from src.utils.config_loader import load_config

# Mock data for testing
@pytest.fixture
def config():
    """Load test configuration"""
    config = load_config()
    config["neo4j"]["node_count_min"] = 10  # Small threshold for testing
    config["neo4j"]["edge_count_min"] = 5
    config["neo4j"]["max_retries"] = 2
    return config

@pytest.fixture
def mock_validated_data():
    """Mock validated data for graph building"""
    return {
        "patients": pd.DataFrame({
            "subject_id": [1, 2],
            "gender": ["Male", "Female"],
            "age": [50, 60],
            "condition_type": ["Type 1 Diabetes", "Type 2 Diabetes"]
        }),
        "treatments": pd.DataFrame({
            "subject_id": [1, 2],
            "drug": ["insulin", "metformin"],
            "treatment_category": ["medication", "medication"],
            "starttime": ["2023-01-01", "2023-01-02"]
        }),
        "prom_scores": [
            {"subject_id": 1, "quality_of_life_score": 70.0, "confidence": 0.7},
            {"subject_id": 2, "quality_of_life_score": 65.0, "confidence": 0.65}
        ],
        "labs": pd.DataFrame({
            "subject_id": [1],
            "lab_test": ["Glucose"],
            "valuenum": [150],
            "charttime": ["2023-01-01"]
        }),
        "procedures": pd.DataFrame({
            "subject_id": [1],
            "procedure_name": ["glucose monitoring"]
        })
    }

@pytest.fixture
def graph_agent(config):
    """Initialize Graph Building Agent"""
    return GraphBuildingAgent(config)

def test_connect(graph_agent):
    """Test Neo4j connection"""
    with patch("neo4j.GraphDatabase.driver") as mock_driver:
        mock_session = MagicMock()
        mock_driver.return_value.session.return_value.__enter__.return_value = mock_session
        assert graph_agent.connect()
        mock_driver.assert_called_with(
            graph_agent.neo4j_uri,
            auth=(graph_agent.neo4j_user, graph_agent.neo4j_password),
            max_connection_retries=graph_agent.config["neo4j"]["max_retries"]
        )
        mock_session.run.assert_called_with("MATCH (n) RETURN n LIMIT 1")

def test_build_graph(graph_agent, mock_validated_data):
    """Test graph building"""
    with patch("neo4j.GraphDatabase.driver") as mock_driver:
        mock_session = MagicMock()
        mock_driver.return_value.session.return_value.__enter__.return_value = mock_session
        mock_session.run.side_effect = [
            None,  # Patient node
            None,  # Treatment node
            None,  # Treatment relationship
            None,  # Outcome node
            None,  # Outcome relationship
            None,  # GraphRAG command
            None,  # Index 1
            None,  # Index 2
            None,  # Index 3
            MagicMock(single=lambda: {"node_count": 15, "rel_count": 10})  # Validation query
        ]
        
        with patch.object(graph_agent.llm, "__call__", return_value='[{"type": "Summary", "level": "global"}]'):
            result = graph_agent.build_graph(mock_validated_data)
        
        assert result["status"] == "success"
        assert result["nodes"] == 15
        assert result["relationships"] == 10
        assert mock_session.run.call_count >= 5  # At least patients, treatments, outcomes

def test_build_graph_empty_data(graph_agent):
    """Test graph building with empty data"""
    result = graph_agent.build_graph({})
    assert result["status"] == "failed"
    assert "Empty patients" in result["error"]

def test_build_graph_neo4j_failure(graph_agent, mock_validated_data):
    """Test graph building with Neo4j connection failure"""
    with patch.object(graph_agent, "connect", return_value=False):
        result = graph_agent.build_graph(mock_validated_data)
        assert result["status"] == "partial_success"
        assert "Neo4j connection failed" in result["error"]

def test_close(graph_agent):
    """Test closing Neo4j connection"""
    with patch("neo4j.GraphDatabase.driver") as mock_driver:
        mock_driver_instance = MagicMock()
        graph_agent.driver = mock_driver_instance
        graph_agent.close()
        mock_driver_instance.close.assert_called_once()
        assert graph_agent.driver is None
