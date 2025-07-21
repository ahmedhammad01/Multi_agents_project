
import logging
import pandas as pd
import numpy as np
from aif360.metrics import BinaryLabelDatasetMetric
from typing import Dict, Any

logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self, config: Dict):
        self.config = config
        self.min_records = config["data_processing"]["min_records"]
        self.max_error_rate = config["data_processing"]["max_error_rate"]
        self.quality_score_min = config["data_processing"]["quality_score_min"]
        self.bias_check_fields = config["ethical"]["bias_check_fields"]

    def clean_patient_data(self, patients: pd.DataFrame) -> pd.DataFrame:
        """Clean patient data (age, gender, condition_type)"""
        logger.info("🧹 Cleaning patient data...")
        try:
            if patients.empty:
                logger.warning("⚠️ Empty patient data")
                return patients
            
            # Avoid SettingWithCopyWarning by using copy
            cleaned = patients.copy()
            
            # Softer duplicate removal
            cleaned = cleaned.drop_duplicates(subset=['subject_id', 'hadm_id'], keep='first')
            
            # Fill missing values
            cleaned.loc[:, 'gender'] = cleaned['gender'].fillna('Unknown').map({'M': 'Male', 'F': 'Female', None: 'Unknown'})
            cleaned.loc[:, 'age'] = cleaned['age'].fillna(cleaned['age'].median())
            cleaned.loc[:, 'condition_type'] = cleaned['condition_type'].astype(str).fillna('Unknown')
            
            # Clip outliers
            cleaned.loc[:, 'age'] = cleaned['age'].clip(18, 120)
            
            # Add age groups
            cleaned.loc[:, 'age_group'] = pd.cut(
                cleaned['age'],
                bins=[18, 30, 50, 65, 80, 120],
                labels=['18-29', '30-49', '50-64', '65-79', '80+']
            )
            
            # Minimal filtering to reduce drop rate
            cleaned = cleaned[cleaned['subject_id'].notna()]
            
            logger.info(f"✅ Patient data cleaned: {len(patients)} → {len(cleaned)} ({(1 - len(cleaned)/len(patients))*100:.1f}% dropped)")
            return cleaned
        except Exception as e:
            logger.error(f"❌ Patient cleaning failed: {e}")
            return patients

    def clean_treatment_data(self, treatments: pd.DataFrame) -> pd.DataFrame:
        """Clean treatment data (drug, dose)"""
        logger.info("🧹 Cleaning treatment data...")
        try:
            if treatments.empty:
                logger.warning("⚠️ Empty treatment data")
                return treatments
            
            cleaned = treatments.copy()
            cleaned.loc[:, 'subject_id'] = cleaned['subject_id'].fillna(-1).astype(str)
            cleaned.loc[:, 'drug'] = cleaned['drug'].str.lower().str.strip().fillna('unknown')
            cleaned.loc[:, 'dose_val_rx'] = pd.to_numeric(cleaned['dose_val_rx'], errors='coerce').fillna(0)
            cleaned.loc[:, 'starttime'] = pd.to_datetime(cleaned['starttime'], errors='coerce').fillna(pd.Timestamp('1900-01-01'))
            cleaned.loc[:, 'treatment_category'] = cleaned['treatment_category'].fillna('unknown')
            
            logger.info(f"✅ Treatment data cleaned: {len(treatments)} → {len(cleaned)} ({(1 - len(cleaned)/len(treatments))*100:.1f}% dropped)")
            return cleaned
        except Exception as e:
            logger.error(f"❌ Treatment cleaning failed: {e}")
            return treatments

    def clean_lab_results(self, labs: pd.DataFrame) -> pd.DataFrame:
        """Clean lab results (glucose, HbA1c)"""
        logger.info("🧹 Cleaning lab results...")
        try:
            if labs.empty:
                logger.warning("⚠️ Empty lab data")
                return labs
            
            cleaned = labs.copy()
            cleaned.loc[:, 'subject_id'] = cleaned['subject_id'].fillna(-1).astype(str)
            cleaned.loc[:, 'lab_test'] = cleaned['lab_test'].fillna('unknown')
            cleaned.loc[:, 'valuenum'] = pd.to_numeric(cleaned['valuenum'], errors='coerce').clip(lower=0)
            cleaned.loc[:, 'charttime'] = pd.to_datetime(cleaned['charttime'], errors='coerce').fillna(pd.Timestamp('1900-01-01'))
            cleaned = cleaned[cleaned['valuenum'].notna()]
            
            logger.info(f"✅ Lab data cleaned: {len(labs)} → {len(cleaned)} ({(1 - len(cleaned)/len(labs))*100:.1f}% dropped)")
            return cleaned
        except Exception as e:
            logger.error(f"❌ Lab cleaning failed: {e}")
            return labs

    def detect_bias(self, patients: pd.DataFrame) -> Dict:
        """Detect demographic bias using AIF360"""
        logger.info("⚖️ Detecting potential bias...")
        bias_report = {}
        try:
            if patients.empty:
                logger.warning("⚠️ No patients for bias detection")
                return {"overall": {"bias_detected": True, "message": "No data"}}

            for field in self.bias_check_fields:
                if field in patients:
                    balance = patients[field].value_counts(normalize=True)
                    min_representation = balance.min() if not balance.empty else 0
                    bias_detected = min_representation < 0.2
                    bias_report[field] = {
                        "bias_detected": bias_detected,
                        "message": f"Min representation: {min_representation:.2f}"
                    }
                    if bias_detected:
                        logger.warning(f"⚠️ Bias detected in {field}: min representation {min_representation:.2f}")

            if 'gender' in patients:
                dataset = pd.DataFrame({
                    'label': (patients['age'] > patients['age'].median()).astype(int),
                    'gender': patients['gender'].map({'Female': 1, 'Male': 0, 'Unknown': 0})
                })
                metric = BinaryLabelDatasetMetric(
                    dataset, privileged_groups=[{'gender': 0}], unprivileged_groups=[{'gender': 1}]
                )
                bias_report['disparate_impact'] = {
                    "bias_detected": metric.disparate_impact() < 0.8,
                    "message": f"Disparate impact: {metric.disparate_impact():.2f}"
                }

            overall_bias = any(report["bias_detected"] for report in bias_report.values())
            bias_report["overall"] = {
                "bias_detected": overall_bias,
                "message": "Bias check completed"
            }
            logger.info("✅ Bias detection completed")
            return bias_report
        except Exception as e:
            logger.error(f"❌ Bias detection failed: {e}")
            return {"overall": {"bias_detected": True, "message": str(e)}}

    def calculate_quality_score(self, patients: pd.DataFrame) -> float:
        """Calculate data quality score"""
        try:
            if patients.empty:
                logger.warning("⚠️ Empty data for quality score")
                return 0.0
            
            required_fields = ['subject_id', 'age', 'gender', 'condition_type']
            completeness = patients[required_fields].notna().all(axis=1).mean()
            age_reasonable = ((patients['age'] >= 18) & (patients['age'] <= 120)).mean()
            gender_balance = min(patients['gender'].value_counts(normalize=True)) if 'gender' in patients else 0
            quality_score = (completeness * 0.5 + age_reasonable * 0.3 + gender_balance * 0.2)
            
            logger.info(f"📊 Data quality score: {quality_score:.2f}")
            return quality_score
        except Exception as e:
            logger.error(f"❌ Quality score calculation failed: {e}")
            return 0.0
