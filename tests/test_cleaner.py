
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data.data_cleaner import DataCleaner
from src.utils.config_loader import load_config

# Mock data for testing
@pytest.fixture
def config():
    """Load test configuration"""
    config = load_config()
    config["data_processing"]["min_records"] = 2
    config["data_processing"]["max_error_rate"] = 0.01
    config["data_processing"]["quality_score_min"] = 0.7
    config["ethical"]["bias_check_fields"] = ["age", "gender"]
    return config

@pytest.fixture
def mock_patients():
    """Mock patient data"""
    return pd.DataFrame({
        "subject_id": [1, 2, 3, 1],  # Duplicate
        "gender": ["M", "F", None, "M"],
        "age": [50, 60, 130, 50],
        "condition_type": ["Type 1 Diabetes", "Type 2 Diabetes", "COPD", "Type 1 Diabetes"]
    })

@pytest.fixture
def mock_treatments():
    """Mock treatment data"""
    return pd.DataFrame({
        "subject_id": [1, 2, None],
        "drug": ["Insulin", "Metformin", "Unknown"],
        "dose_val_rx": ["100", "500", None],
        "starttime": ["2023-01-01", "invalid", "2023-01-02"]
    })

@pytest.fixture
def mock_labs():
    """Mock lab data"""
    return pd.DataFrame({
        "subject_id": [1, 2],
        "lab_test": ["Glucose", "HbA1c"],
        "valuenum": [150, -1],  # Invalid value
        "charttime": ["2023-01-01", "invalid"]
    })

@pytest.fixture
def data_cleaner(config):
    """Initialize DataCleaner"""
    return DataCleaner(config)

def test_clean_patient_data(data_cleaner, mock_patients):
    """Test cleaning patient data"""
    result = data_cleaner.clean_patient_data(mock_patients)
    assert len(result) == 2  # Drops duplicate and None gender
    assert result["age"].max() <= 120  # Drops age=130
    assert all(result["gender"].isin(["Male", "Female"]))
    assert "age_group" in result.columns
    assert all(result["age_group"].isin(["18-29", "30-49", "50-64", "65-79", "80+"]))

def test_clean_treatment_data(data_cleaner, mock_treatments):
    """Test cleaning treatment data"""
    result = data_cleaner.clean_treatment_data(mock_treatments)
    assert len(result) == 2  # Drops None subject_id
    assert result["drug"].str.islower().all()
    assert pd.api.types.is_datetime64_any_dtype(result["starttime"])
    assert pd.api.types.is_numeric_dtype(result["dose_val_rx"])

def test_clean_lab_data(data_cleaner, mock_labs):
    """Test cleaning lab data"""
    result = data_cleaner.clean_lab_results(mock_labs)
    assert len(result) == 1  # Drops invalid valuenum
    assert result["valuenum"].min() > 0
    assert pd.api.types.is_datetime64_any_dtype(result["charttime"])

def test_detect_bias(data_cleaner, mock_patients):
    """Test bias detection"""
    with patch("aif360.metrics.BinaryLabelDatasetMetric") as mock_metric:
        mock_metric.return_value.disparate_impact.return_value = 0.75
        result = data_cleaner.detect_bias(mock_patients)
        assert result["gender"]["bias_detected"]
        assert result["disparate_impact"]["disparate_impact"] == 0.75
        assert result["overall"]["bias_detected"]

def test_calculate_quality_score(data_cleaner, mock_patients):
    """Test quality score calculation"""
    result = data_cleaner.calculate_quality_score(mock_patients)
    assert 0.0 <= result <= 1.0
    assert result < 0.7  # Due to missing gender and duplicates

def test_calculate_quality_score_empty(data_cleaner):
    """Test quality score with empty data"""
    result = data_cleaner.calculate_quality_score(pd.DataFrame())
    assert result == 0.0
