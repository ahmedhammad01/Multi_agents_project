
import pytest
import pandas as pd
import os
import json
from src.agents.data_agent import DataProcessingAgent, DataState
from src.utils.config_loader import load_config
from unittest.mock import patch, MagicMock

# Mock data for testing
@pytest.fixture
def config():
    """Load test configuration"""
    config = load_config()
    config["bigquery"]["batch_size"] = 100  # Small batch for testing
    config["data_processing"]["min_records"] = 50
    config["data_processing"]["quality_score_min"] = 0.7
    config["data_processing"]["prom_confidence_min"] = 0.6
    return config

@pytest.fixture
def mock_data():
    """Mock MIMIC-IV data"""
    patients = pd.DataFrame({
        "subject_id": [1, 2, 3],
        "gender": ["M", "F", None],
        "age": [50, 60, 130],
        "condition_type": ["Type 1 Diabetes", "Type 2 Diabetes", "COPD"]
    })
    treatments = pd.DataFrame({
        "subject_id": [1, 2],
        "drug": ["insulin", "metformin"],
        "treatment_category": ["medication", "medication"],
        "starttime": ["2023-01-01", "2023-01-02"]
    })
    labs = pd.DataFrame({
        "subject_id": [1],
        "lab_test": ["Glucose"],
        "valuenum": [150],
        "charttime": ["2023-01-01"]
    })
    notes = pd.DataFrame({
        "subject_id": [1, 2],
        "text": ["Stable condition", "Poorly controlled"]
    })
    return {
        "patients": patients,
        "treatments": treatments,
        "labs": labs,
        "notes": notes,
        "ingestion_timestamp": "2025-07-20T20:00:00",
        "record_counts": {
            "patients": 3,
            "treatments": 2,
            "labs": 1,
            "notes": 2
        }
    }

@pytest.fixture
def data_agent(config):
    """Initialize Data Processing Agent"""
    return DataProcessingAgent(config)

def test_ingest_node(data_agent, mock_data):
    """Test ingestion node"""
    with patch.object(data_agent.mimic_connector, 'connect', return_value=True), \
         patch.object(data_agent.mimic_connector, 'get_diabetes_patients', return_value=mock_data["patients"]), \
         patch.object(data_agent.mimic_connector, 'get_treatment_data', return_value=mock_data["treatments"]), \
         patch.object(data_agent.mimic_connector, 'get_lab_results', return_value=mock_data["labs"]), \
         patch.object(data_agent.mimic_connector, 'get_discharge_notes', return_value=mock_data["notes"]), \
         patch.object(data_agent.mimic_connector, 'get_procedures', return_value=pd.DataFrame()), \
         patch.object(data_agent.mimic_connector, 'save_raw_data', return_value=True):
        
        state = DataState()
        result_state = data_agent._ingest_node(state)
        
        assert result_state.raw_data["patients"].equals(mock_data["patients"])
        assert result_state.raw_data["record_counts"]["patients"] == 3
        assert not result_state.issues_detected  # No issues with mock data

def test_detect_issues_node(data_agent, mock_data):
    """Test issue detection node"""
    state = DataState(raw_data=mock_data)
    result_state = data_agent._detect_issues_node(state)
    
    assert "high_missing_values" in result_state.issues_detected  # Due to None in gender
    assert "outliers_age" in result_state.issues_detected  # Due to age=130
    assert "bias_gender" not in result_state.issues_detected  # Small sample, no severe bias

def test_clean_node(data_agent, mock_data):
    """Test cleaning node"""
    state = DataState(raw_data=mock_data)
    result_state = data_agent._clean_node(state)
    
    cleaned_patients = result_state.cleaned_data["patients"]
    assert len(cleaned_patients) == 2  # Drops None gender and age=130
    assert cleaned_patients["age"].max() <= 120
    assert all(cleaned_patients["gender"].isin(["Male", "Female"]))

def test_prom_extract_node(data_agent, mock_data):
    """Test PROM extraction node"""
    state = DataState(cleaned_data=mock_data)
    result_state = data_agent._prom_extract_node(state)
    
    assert len(result_state.prom_scores) == 2  # Two patients with notes
    assert all(p["confidence"] >= 0.6 for p in result_state.prom_scores)
    assert all(15 <= p["quality_of_life_score"] <= 95 for p in result_state.prom_scores)

def test_validate_node(data_agent, mock_data):
    """Test validation node"""
    state = DataState(cleaned_data=mock_data, prom_scores=[
        {"subject_id": 1, "confidence": 0.7, "quality_of_life_score": 70.0},
        {"subject_id": 2, "confidence": 0.65, "quality_of_life_score": 60.0}
    ])
    result_state = data_agent._validate_node(state)
    
    assert not result_state.validation_results["validations"]["record_count"]  # 3 < 50
    assert result_state.validation_results["validations"]["prom_confidence"]  # Avg 0.675 >= 0.6
    assert "validation_failed" in result_state.issues_detected

def test_run_workflow(data_agent, mock_data):
    """Test full Data Processing Agent workflow"""
    with patch.object(data_agent.mimic_connector, 'connect', return_value=True), \
         patch.object(data_agent.mimic_connector, 'get_diabetes_patients', return_value=mock_data["patients"]), \
         patch.object(data_agent.mimic_connector, 'get_treatment_data', return_value=mock_data["treatments"]), \
         patch.object(data_agent.mimic_connector, 'get_lab_results', return_value=mock_data["labs"]), \
         patch.object(data_agent.mimic_connector, 'get_discharge_notes', return_value=mock_data["notes"]), \
         patch.object(data_agent.mimic_connector, 'get_procedures', return_value=pd.DataFrame()), \
         patch.object(data_agent.mimic_connector, 'save_raw_data', return_value=True):
        
        initial_state = DataState()
        result = data_agent.run(initial_state)
        
        assert result["status"] == "partial_success"  # Due to low record count
        assert result["quality_score"] >= 0.7
        assert "validation_failed" in result["issues"]
