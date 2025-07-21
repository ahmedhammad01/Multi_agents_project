
import logging
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, f_oneway
import shap
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class StatsUtils:
    def __init__(self, config: Dict):
        self.config = config
        self.min_p_value = 0.05  # Threshold for statistical significance
        self.min_impact = 0.10  # Minimum clinical impact (10%)

    def calculate_correlations(self, df: pd.DataFrame, target_col: str, features: List[str]) -> Dict:
        """Calculate correlations between target and features"""
        logger.info(f"📊 Calculating correlations for {target_col}")
        try:
            correlations = {}
            for feature in features:
                if feature in df and pd.api.types.is_numeric_dtype(df[feature]):
                    corr = df[target_col].corr(df[feature])
                    correlations[feature] = float(corr) if not pd.isna(corr) else 0.0
            logger.info(f"✅ Correlations calculated: {len(correlations)} features")
            return correlations
        except Exception as e:
            logger.error(f"❌ Correlation calculation failed: {e}")
            return {}

    def perform_ttest(self, df: pd.DataFrame, group_col: str, value_col: str, groups: List[str]) -> List[Dict]:
        """Perform t-tests for pathway comparisons"""
        logger.info(f"📊 Performing t-tests for {group_col} on {value_col}")
        results = []
        try:
            for group in groups:
                group1 = df[df[group_col] == group][value_col].dropna()
                group2 = df[df[group_col] != group][value_col].dropna()
                if len(group1) > 1 and len(group2) > 1:
                    t_stat, p_value = ttest_ind(group1, group2, nan_policy='omit')
                    # Ensure t_stat and p_value are scalars
                    if isinstance(t_stat, (np.ndarray, tuple, list)):
                        t_stat = t_stat[0]
                    if isinstance(p_value, (np.ndarray, tuple, list)):
                        p_value = p_value[0]
                    group2_mean = group2.mean()
                    if pd.isna(group2_mean) or group2_mean == 0:
                        impact = 0
                    else:
                        impact = float(group1.mean() - group2_mean) / group2_mean
                    results.append({
                        "group": group,
                        "t_stat": float(np.asarray(t_stat).item()),
                        "p_value": float(np.asarray(p_value).item()),
                        "impact": impact,
                        "significant": float(np.asarray(p_value).item()) < self.min_p_value and abs(impact) >= self.min_impact
                    })
            logger.info(f"✅ T-tests completed: {len(results)} groups")
            return results
        except Exception as e:
            logger.error(f"❌ T-test failed: {e}")
            return []

    def perform_anova(self, df: pd.DataFrame, group_col: str, value_col: str) -> Dict:
        """Perform ANOVA for demographic fairness"""
        logger.info(f"📊 Performing ANOVA for {group_col} on {value_col}")
        try:
            groups = [df[df[group_col] == g][value_col].dropna() for g in df[group_col].unique()]
            if len(groups) > 1 and all(len(g) > 1 for g in groups):
                f_stat, p_value = f_oneway(*groups)
                return {
                    "f_stat": float(f_stat),
                    "p_value": float(p_value),
                    "significant": p_value < self.min_p_value
                }
            logger.warning("⚠️ Insufficient data for ANOVA")
            return {}
        except Exception as e:
            logger.error(f"❌ ANOVA failed: {e}")
            return {}

    def calculate_shap_values(self, df: pd.DataFrame, target_col: str, features: List[str]) -> Dict:
        """Calculate SHAP values for feature importance"""
        logger.info(f"📊 Calculating SHAP values for {target_col}")
        try:
            if target_col not in df or not features:
                logger.warning("⚠️ Invalid target or features")
                return {}
            X = df[features].fillna(df[features].mean())
            y = (df[target_col] > df[target_col].median()).astype(int)
            model = LogisticRegression().fit(X, y)
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)
            result = {
                "features": features,
                "shap_values": {f: float(np.mean(np.abs(shap_values.values[:, i]))) for i, f in enumerate(features)}
            }
            logger.info(f"✅ SHAP values calculated for {len(features)} features")
            return result
        except Exception as e:
            logger.error(f"❌ SHAP calculation failed: {e}")
            return {}

    def analyze_pathways(self, df: pd.DataFrame, group_col: str = "treatment_type", value_col: str = "qol_score") -> Dict:
        """Analyze pathways with statistical tests and SHAP"""
        logger.info(f"🔬 Analyzing pathways for {group_col} on {value_col}")
        try:
            correlations = self.calculate_correlations(df, value_col, ['age', 'qol_score'])
            t_tests = self.perform_ttest(df, group_col, value_col, list(df[group_col].unique()))
            anova = self.perform_anova(df, "gender", value_col)
            shap_values = self.calculate_shap_values(df, value_col, ['age'])
            
            results = {
                "correlations": correlations,
                "t_tests": t_tests,
                "anova": anova,
                "shap_values": shap_values,
                "pathways": [
                    {
                        "treatment": t["group"],
                        "impact": t["impact"],
                        "p_value": t["p_value"],
                        "significant": t["significant"]
                    } for t in t_tests if t["significant"]
                ]
            }
            logger.info(f"✅ Pathway analysis complete: {len(t_tests)} tests")
            return results
        except Exception as e:
            logger.error(f"❌ Pathway analysis failed: {e}")
            return {}
