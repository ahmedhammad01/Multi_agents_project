
import logging
import os
import json
from datetime import datetime
from typing import Dict, Any
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
import spacy
from tenacity import retry, stop_after_attempt, wait_exponential
import pandas as pd
import numpy as np
from src.data.mimic_connector import MIMICDataConnector
from src.data.data_cleaner import DataCleaner
from src.data.prom_extractor import PROMExtractor
from src.data.validator import DataValidator

logger = logging.getLogger(__name__)

# Define shared state
from pydantic import BaseModel
from pydantic import ConfigDict

class DataState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    raw_data: Dict[str, Any] = {}  # DataFrames for patients, treatments, labs, notes, procedures
    cleaned_data: Dict[str, Any] = {}
    prom_scores: list = []
    validation_results: Dict = {}
    issues_detected: list = []
    quality_score: float = 0.0
    attempts: Dict[str, int] = {}  # Track retries per node
    flags: list = []  # e.g., 'partial_success'
    cumulative_patients: pd.DataFrame = pd.DataFrame()  # Accumulate cleaned patients across batches
    cumulative_treatments: pd.DataFrame = pd.DataFrame()
    cumulative_labs: pd.DataFrame = pd.DataFrame()
    cumulative_notes: pd.DataFrame = pd.DataFrame()
    cumulative_procedures: pd.DataFrame = pd.DataFrame()
    cumulative_prom_scores: list = []

class DataProcessingAgent:
    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = config["project"]["data_dir"]
        self.raw_data_file = config["project"]["raw_data_file"]
        self.clean_data_file = config["project"]["clean_data_file"]
        self.mimic_connector = MIMICDataConnector(config)
        self.data_cleaner = DataCleaner(config)
        self.prom_extractor = PROMExtractor(config)
        self.validator = DataValidator(config)
        self.nlp = None
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load spaCy model: {e}")
        self.workflow = self._build_workflow()
        self.batch_size = 5000  # Batch size for incremental fetching
        self.target_cleaned = 10000  # Target number of cleaned patients
        self.offset = 0  # Initial offset for querying

    def _build_workflow(self):
        """Build LangGraph workflow for data processing"""
        workflow = StateGraph(DataState)
        workflow.add_node("ingest", self._ingest_node)
        workflow.add_node("detect_issues", self._detect_issues_node)
        workflow.add_node("clean", self._clean_node)
        workflow.add_node("prom_extract", self._prom_extract_node)
        workflow.add_node("validate", self._validate_node)
        
        workflow.set_entry_point("ingest")
        workflow.add_edge("ingest", "detect_issues")
        workflow.add_conditional_edges(
            "detect_issues",
            lambda state: "clean" if state.issues_detected else "prom_extract"
        )
        workflow.add_edge("clean", "prom_extract")
        workflow.add_edge("prom_extract", "validate")
        workflow.add_conditional_edges(
            "validate",
            lambda state: "detect_issues" if "validation_failed" in state.issues_detected and state.attempts.get("validate", 0) < self.config["data_processing"]["max_validation_iterations"] else END
        )
        return workflow.compile()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10))
    def _ingest_node(self, state: DataState) -> DataState:
        """Ingest data from MIMIC-IV"""
        logger.info("🔄 Stage 1: Data Ingestion")
        try:
            if not self.mimic_connector.connect():
                state.issues_detected.append("connection_failed")
                raise ValueError("BigQuery connection failed")
            
            batch_size = self.config["bigquery"]["batch_size"]
            logger.info(f"📊 Querying MIMIC-IV for up to {batch_size} patients...")
            patients_df = self.mimic_connector.get_diabetes_patients(limit=batch_size)
            
            if len(patients_df) < batch_size * 0.7:
                self.config["bigquery"]["batch_size"] //= 2
                raise ValueError("Partial fetch; retrying smaller batch")
            if patients_df.empty:
                state.issues_detected.append("no_data")
                raise ValueError("No patient data retrieved")
            
            subject_ids = patients_df['subject_id'].unique().tolist()
            treatments_df = self.mimic_connector.get_treatment_data(subject_ids)
            labs_df = self.mimic_connector.get_lab_results(subject_ids)
            notes_df = self.mimic_connector.get_discharge_notes(subject_ids, limit_per_patient=3)
            procedures_df = self.mimic_connector.get_procedures(subject_ids)
            
            # Integrity check
            error_rate = (patients_df['subject_id'].duplicated().mean() + patients_df[['subject_id', 'age']].isnull().mean().mean()) * 100
            if error_rate > self.config["data_processing"]["max_error_rate"] * 100:
                state.issues_detected.append("high_error_rate")
            
            state.raw_data = {
                "patients": patients_df,
                "treatments": treatments_df,
                "labs": labs_df,
                "notes": notes_df,
                "procedures": procedures_df,
                "ingestion_timestamp": datetime.now().isoformat(),
                "record_counts": {
                    "patients": len(patients_df),
                    "treatments": len(treatments_df),
                    "labs": len(labs_df),
                    "notes": len(notes_df),
                    "procedures": len(procedures_df)
                }
            }
            
            os.makedirs(os.path.dirname(self.raw_data_file), exist_ok=True)
            if self.mimic_connector.save_raw_data(state.raw_data, self.raw_data_file):
                logger.info(f"✅ Ingested {len(patients_df)} patients")
            else:
                state.issues_detected.append("save_failed")
        except Exception as e:
            state.issues_detected.append(f"ingestion_error: {e}")
            if os.path.exists(self.raw_data_file):
                with open(self.raw_data_file, 'r') as f:
                    state.raw_data = json.load(f)
            else:
                raise
        return state

    def _detect_issues_node(self, state: DataState) -> DataState:
        """Detect data issues (missing, outliers, bias)"""
        logger.info("🔍 Stage 2: Issue Detection")
        try:
            patients = state.raw_data.get("patients", pd.DataFrame())
            if patients.empty:
                state.issues_detected.append("empty_data")
                return state
            
            missing_rate = patients[['age', 'gender', 'condition_type']].isnull().mean().mean() * 100
            if missing_rate > 5:
                state.issues_detected.append("high_missing_values")
            
            if 'age' in patients and (patients['age'] > 120).any():
                state.issues_detected.append("outliers_age")
            
            gender_balance = min(patients['gender'].value_counts(normalize=True)) if 'gender' in patients else 0
            if gender_balance < 0.2:
                state.issues_detected.append("bias_gender")
            
            logger.info(f"🔍 Detected issues: {state.issues_detected}")
        except Exception as e:
            state.issues_detected.append(f"detection_error: {e}")
        return state

    def _clean_node(self, state: DataState) -> DataState:
        """Clean data based on detected issues"""
        logger.info("🧹 Stage 3: Data Cleaning")
        state.attempts['clean'] = state.attempts.get('clean', 0) + 1
        try:
            patients = state.raw_data.get("patients", pd.DataFrame())
            treatments = state.raw_data.get("treatments", pd.DataFrame())
            labs = state.raw_data.get("labs", pd.DataFrame())
            notes = state.raw_data.get("notes", pd.DataFrame())
            procedures = state.raw_data.get("procedures", pd.DataFrame())
            
            if patients.empty:
                state.issues_detected.append("empty_data")
                return state
            
            if state.attempts['clean'] == 1:
                patients = self.data_cleaner.clean_patient_data(patients)
                treatments = self.data_cleaner.clean_treatment_data(treatments)
                labs = self.data_cleaner.clean_lab_results(labs)
            else:
                # Softer cleaning on retry
                patients['age'] = patients['age'].fillna(patients['age'].median())
                treatments = treatments.dropna(subset=['subject_id', 'drug'])
                labs = labs[labs['valuenum'] > 0]
            
            state.quality_score = self.data_cleaner.calculate_quality_score(patients)
            if state.quality_score < self.config["data_processing"]["quality_score_min"]:
                state.issues_detected.append("low_quality")
            
            state.cleaned_data = {
                "patients": patients,
                "treatments": treatments,
                "labs": labs,
                "notes": notes,
                "procedures": procedures,
                "cleaning_timestamp": datetime.now().isoformat()
            }
            
            os.makedirs(os.path.dirname(self.clean_data_file), exist_ok=True)
            with open(self.clean_data_file, 'w') as f:
                json.dump({k: v.to_dict('records') if isinstance(v, pd.DataFrame) else v for k, v in state.cleaned_data.items()}, f, default=str, indent=2)
            
            logger.info(f"✅ Cleaning complete: Quality score {state.quality_score:.3f}")
        except Exception as e:
            state.issues_detected.append(f"clean_error: {e}")
        return state

    def _prom_extract_node(self, state: DataState) -> DataState:
        """Extract PROMs from clinical notes"""
        logger.info("🔍 Stage 4: PROM Extraction")
        try:
            patients = state.cleaned_data.get("patients", pd.DataFrame())
            notes = state.cleaned_data.get("notes", pd.DataFrame())
            labs = state.cleaned_data.get("labs", pd.DataFrame())
            treatments = state.cleaned_data.get("treatments", pd.DataFrame())
            
            if patients.empty or notes.empty:
                state.issues_detected.append("no_data_for_prom")
                return state
            
            prom_scores = self.prom_extractor.extract_proms(patients, notes, labs, treatments)
            state.prom_scores = prom_scores
            avg_confidence = sum(p['confidence'] for p in prom_scores) / len(prom_scores) if prom_scores else 0
            
            state.cleaned_data["prom_scores"] = prom_scores
            state.cleaned_data["prom_summary"] = {
                "average_qol": sum(p['quality_of_life_score'] for p in prom_scores) / len(prom_scores) if prom_scores else 0,
                "average_confidence": avg_confidence,
                "patients_with_notes": sum(p['data_completeness']['has_notes'] for p in prom_scores)
            }
            
            with open(self.clean_data_file, 'w') as f:
                json.dump({k: v.to_dict('records') if isinstance(v, pd.DataFrame) else v for k, v in state.cleaned_data.items()}, f, default=str, indent=2)
            
            if avg_confidence < self.config["data_processing"]["prom_confidence_min"]:
                state.issues_detected.append("low_prom_confidence")
            
            logger.info(f"✅ PROM extraction complete: {len(prom_scores)} scores, avg confidence {avg_confidence:.3f}")
        except Exception as e:
            state.issues_detected.append(f"prom_error: {e}")
        return state

    def _validate_node(self, state: DataState) -> DataState:
        """Validate processed data"""
        logger.info("✅ Stage 5: Data Validation")
        state.attempts['validate'] = state.attempts.get('validate', 0) + 1
        try:
            patients = state.cleaned_data.get("patients", pd.DataFrame())
            prom_scores = state.prom_scores
            validations = self.validator.validate_data(patients, state.cleaned_data, prom_scores)
            
            state.validation_results = validations
            state.validation_results["validation_passed"] = all(validations.values())
            state.validation_results["validation_timestamp"] = datetime.now().isoformat()
            
            if not state.validation_results["validation_passed"]:
                state.issues_detected.append("validation_failed")
            
            with open(self.clean_data_file, 'w') as f:
                json.dump({k: v.to_dict('records') if isinstance(v, pd.DataFrame) else v for k, v in state.cleaned_data.items()}, f, default=str, indent=2)
            
            logger.info(f"✅ Validation complete: Passed={state.validation_results['validation_passed']}")
        except Exception as e:
            state.issues_detected.append(f"validation_error: {e}")
            state.flags.append("partial_success")
        return state

    def run(self, initial_state: DataState) -> Dict:
            """Execute the LangGraph workflow with incremental batch processing"""
            try:
                while len(initial_state.cumulative_patients) < self.target_cleaned:
                    logger.info(f"📊 Processing batch with offset {self.offset}")
                    batch_state = DataState()
                    batch_state = self.workflow.invoke(batch_state)
                    
                    # Append cleaned batch to cumulative
                    batch_cleaned = batch_state["cleaned_data"].get("patients", pd.DataFrame())
                    initial_state.cumulative_patients = pd.concat([initial_state.cumulative_patients, batch_cleaned], ignore_index=True)
                    
                    # Similarly accumulate other data
                    initial_state.cumulative_treatments = pd.concat([initial_state.cumulative_treatments, batch_state["cleaned_data"].get("treatments", pd.DataFrame())], ignore_index=True)
                    initial_state.cumulative_labs = pd.concat([initial_state.cumulative_labs, batch_state["cleaned_data"].get("labs", pd.DataFrame())], ignore_index=True)
                    initial_state.cumulative_notes = pd.concat([initial_state.cumulative_notes, batch_state["cleaned_data"].get("notes", pd.DataFrame())], ignore_index=True)
                    initial_state.cumulative_procedures = pd.concat([initial_state.cumulative_procedures, batch_state["cleaned_data"].get("procedures", pd.DataFrame())], ignore_index=True)
                    initial_state.cumulative_prom_scores.extend(batch_state["prom_scores"])
                    
                    # Save cumulative cleaned data
                    os.makedirs(os.path.dirname(self.clean_data_file), exist_ok=True)
                    cumulative_data = {
                        "patients": initial_state.cumulative_patients.to_dict('records'),
                        "treatments": initial_state.cumulative_treatments.to_dict('records'),
                        "labs": initial_state.cumulative_labs.to_dict('records'),
                        "notes": initial_state.cumulative_notes.to_dict('records'),
                        "procedures": initial_state.cumulative_procedures.to_dict('records'),
                        "prom_scores": initial_state.cumulative_prom_scores,
                        "cumulative_timestamp": datetime.now().isoformat()
                    }
                    with open(self.clean_data_file, 'w') as f:
                        json.dump(cumulative_data, f, default=str, indent=2)
                    
                    logger.info(f"✅ Accumulated {len(initial_state.cumulative_patients)} cleaned patients")
                    
                    if batch_state["issues_detected"]:
                        logger.warning(f"⚠️ Batch issues: {batch_state['issues_detected']}")
                        break  # Stop if major issues
                    
                    self.offset += self.batch_size
                
                final_result = {
                    "status": "success" if len(initial_state.cumulative_patients) >= self.target_cleaned else "partial_success",
                    "quality_score": batch_state["quality_score"],
                    "issues": batch_state["issues_detected"],
                    "record_counts": {
                        "patients": len(initial_state.cumulative_patients),
                        "treatments": len(initial_state.cumulative_treatments),
                        "labs": len(initial_state.cumulative_labs),
                        "notes": len(initial_state.cumulative_notes),
                        "procedures": len(initial_state.cumulative_procedures)
                    }
                }
                return final_result
            except Exception as e:
                logger.error(f"❌ Workflow failed: {e}")
                return {"status": "failed", "error": str(e)}
