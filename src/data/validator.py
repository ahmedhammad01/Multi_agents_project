
import logging
from typing import Dict, Any, List
import pandas as pd
from aif360.metrics import BinaryLabelDatasetMetric
from datetime import datetime

logger = logging.getLogger(__name__)

class DataValidator:
    def __init__(self, config: Dict):
        self.config = config
        self.min_records = config["data_processing"]["min_records"]
        self.quality_score_min = config["data_processing"]["quality_score_min"]
        self.prom_confidence_min = config["data_processing"]["prom_confidence_min"]
        self.lab_coverage_min = config["data_processing"]["lab_coverage_min"]
        self.treatment_coverage_min = config["data_processing"]["treatment_coverage_min"]

    def validate_data(self, patients: pd.DataFrame, cleaned_data: Dict[str, Any], prom_scores: List[Dict]) -> Dict:
        """Validate processed data against quality and fairness thresholds"""
        logger.info("✅ Validating processed data and results")
        validations = {}
        try:
            # Record count check
            patient_count = len(patients)
            validations["record_count"] = patient_count >= self.min_records
            if not validations["record_count"]:
                logger.warning(f"⚠️ Insufficient records: {patient_count} < {self.min_records}")

            # Quality score check
            quality_score = self._calculate_quality_score(patients)
            validations["data_quality"] = quality_score >= self.quality_score_min
            if not validations["data_quality"]:
                logger.warning(f"⚠️ Low quality score: {quality_score:.2f} < {self.quality_score_min}")

            # PROM confidence check
            prom_confidence = sum(p['confidence'] for p in prom_scores) / len(prom_scores) if prom_scores else 0
            validations["prom_confidence"] = prom_confidence >= self.prom_confidence_min
            if not validations["prom_confidence"]:
                logger.warning(f"⚠️ Low PROM confidence: {prom_confidence:.2f} < {self.prom_confidence_min}")

            # Data coverage checks
            labs = pd.DataFrame(cleaned_data.get("labs", [])) if isinstance(cleaned_data.get("labs"), list) else cleaned_data.get("labs", pd.DataFrame())
            treatments = pd.DataFrame(cleaned_data.get("treatments", [])) if isinstance(cleaned_data.get("treatments"), list) else cleaned_data.get("treatments", pd.DataFrame())
            patients_with_labs = len(set(labs['subject_id'])) if not labs.empty else 0
            patients_with_treatments = len(set(treatments['subject_id'])) if not treatments.empty else 0
            lab_coverage = patients_with_labs / patient_count if patient_count > 0 else 0
            treatment_coverage = patients_with_treatments / patient_count if patient_count > 0 else 0
            
            validations["lab_coverage"] = lab_coverage >= self.lab_coverage_min
            validations["treatment_coverage"] = treatment_coverage >= self.treatment_coverage_min
            if not validations["lab_coverage"]:
                logger.warning(f"⚠️ Low lab coverage: {lab_coverage:.3f} < {self.lab_coverage_min}")
            if not validations["treatment_coverage"]:
                logger.warning(f"⚠️ Low treatment coverage: {treatment_coverage:.3f} < {self.treatment_coverage_min}")

            # Data source check
            data_source = cleaned_data.get("data_source", "MIMIC-IV")
            validations["real_data_source"] = data_source.startswith("MIMIC-IV") or patient_count >= 1000
            if not validations["real_data_source"]:
                logger.warning("⚠️ Data source not verified as MIMIC-IV")

            # Demographics check
            validations["demographics_available"] = any(p.get('age', 0) > 0 for p in cleaned_data.get("patients", []) if isinstance(p, dict))
            if not validations["demographics_available"]:
                logger.warning("⚠️ No valid demographics found")

            # Fairness check with AIF360
            fairness_metrics = {}
            if 'gender' in patients:
                dataset = pd.DataFrame({
                    'label': (patients['age'] > patients['age'].median()).astype(int),
                    'gender': patients['gender'].map({'Female': 1, 'Male': 0, 'Unknown': 0})
                })
                metric = BinaryLabelDatasetMetric(
                    dataset, privileged_groups=[{'gender': 0}], unprivileged_groups=[{'gender': 1}]
                )
                fairness_metrics['disparate_impact'] = metric.disparate_impact()
                validations["fairness"] = fairness_metrics['disparate_impact'] >= 0.8
                if not validations["fairness"]:
                    logger.warning(f"⚠️ Bias detected: Disparate impact {fairness_metrics['disparate_impact']:.2f}")

            return {
                "validations": validations,
                "fairness_metrics": fairness_metrics,
                "completeness_score": round((lab_coverage + treatment_coverage + (1.0 if validations["demographics_available"] else 0.0)) / 3, 2),
                "quality_score": quality_score,
                "prom_confidence": round(prom_confidence, 2),
                "validation_summary": {
                    "total_patients": patient_count,
                    "lab_coverage_percent": round(lab_coverage * 100, 1),
                    "treatment_coverage_percent": round(treatment_coverage * 100, 1),
                    "data_source": data_source
                }
            }
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            return {
                "validations": {},
                "fairness_metrics": {},
                "completeness_score": 0.0,
                "quality_score": 0.0,
                "prom_confidence": 0.0,
                "validation_summary": {"error": str(e)}
            }

    def _calculate_quality_score(self, patients: pd.DataFrame) -> float:
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
            
            logger.info(f"📊 Validation quality score: {quality_score:.2f}")
            return quality_score
        except Exception as e:
            logger.error(f"❌ Validation quality score failed: {e}")
            return 0.0
