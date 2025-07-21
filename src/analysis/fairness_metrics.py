
import logging
import pandas as pd
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import BinaryLabelDataset
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class FairnessMetrics:
    def __init__(self, config: Dict):
        self.config = config
        self.bias_check_fields = config["ethical"]["bias_check_fields"]
        self.disparate_impact_threshold = 0.8  # Minimum acceptable disparate impact ratio

    def calculate_fairness_metrics(self, df: pd.DataFrame, outcome_col: str = "qol_score") -> Dict:
        """Calculate fairness metrics for demographic groups"""
        logger.info(f"⚖️ Calculating fairness metrics for {outcome_col}")
        fairness_metrics = {}
        try:
            if df.empty or outcome_col not in df:
                logger.warning("⚠️ Invalid data or outcome column")
                return {"status": "failed", "error": "Invalid input"}

            # Prepare binary label (e.g., above/below median QoL)
            df['label'] = (df[outcome_col] > df[outcome_col].median()).astype(int)

            for field in self.bias_check_fields:
                if field in df:
                    # Create AIF360 dataset
                    dataset = BinaryLabelDataset(
                        df=df[[field, 'label']],
                        label_names=['label'],
                        protected_attribute_names=[field]
                    )
                    
                    # Define privileged/unprivileged groups
                    privileged = {field: df[field].value_counts().idxmax()}  # Most common value
                    unprivileged = {field: v for v in df[field].unique() if v != privileged[field]}
                    
                    if unprivileged:
                        metric = BinaryLabelDatasetMetric(
                            dataset,
                            privileged_groups=[privileged],
                            unprivileged_groups=[unprivileged]
                        )
                        disparate_impact = metric.disparate_impact()
                        fairness_metrics[field] = {
                            "disparate_impact": float(disparate_impact),
                            "bias_detected": disparate_impact < self.disparate_impact_threshold,
                            "message": f"Disparate impact for {field}: {disparate_impact:.2f}"
                        }
                        if fairness_metrics[field]["bias_detected"]:
                            logger.warning(f"⚠️ Bias detected in {field}: Disparate impact {disparate_impact:.2f}")

            # Overall fairness
            overall_bias = any(m["bias_detected"] for m in fairness_metrics.values())
            fairness_metrics["overall"] = {
                "bias_detected": overall_bias,
                "message": "Fairness check completed"
            }
            
            logger.info("✅ Fairness metrics calculated")
            return {
                "status": "success",
                "fairness_metrics": fairness_metrics
            }
        except Exception as e:
            logger.error(f"❌ Fairness metrics calculation failed: {e}")
            return {"status": "failed", "error": str(e)}

    def reweight_data(self, df: pd.DataFrame, field: str, outcome_col: str = "qol_score") -> pd.DataFrame:
        """Reweight data to mitigate detected bias"""
        logger.info(f"⚖️ Reweighting data for {field}")
        try:
            if df.empty or field not in df or outcome_col not in df:
                logger.warning("⚠️ Invalid data or columns")
                return df

            from aif360.algorithms.preprocessing import Reweighing
            dataset = BinaryLabelDataset(
                df=df[[field, outcome_col]],
                label_names=[outcome_col],
                protected_attribute_names=[field]
            )
            privileged = {field: df[field].value_counts().idxmax()}
            unprivileged = {field: v for v in df[field].unique() if v != privileged[field]}
            
            if unprivileged:
                reweigher = Reweighing(unprivileged_groups=[unprivileged], privileged_groups=[privileged])
                reweighted_dataset = reweigher.fit_transform(dataset)
                reweighted_df = reweighted_dataset.convert_to_dataframe()[0]
                logger.info(f"✅ Data reweighted for {field}")
                return reweighted_df
            return df
        except Exception as e:
            logger.error(f"❌ Data reweighting failed: {e}")
            return df
