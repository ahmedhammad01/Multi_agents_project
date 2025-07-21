
import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class PredictiveModels:
    def __init__(self, config: Dict):
        self.config = config
        self.min_sample_size = 100  # Minimum samples for reliable modeling
        self.min_accuracy = 0.7  # Minimum acceptable accuracy

    def predict_readmission_risk(self, df: pd.DataFrame, features: List[str], target_col: str = "readmission") -> Dict:
        """Predict readmission risk using logistic regression"""
        logger.info(f"📈 Predicting readmission risk with features: {features}")
        try:
            if df.empty or target_col not in df or not features:
                logger.warning("⚠️ Invalid data or features for prediction")
                return {"status": "failed", "error": "Invalid input"}

            # Prepare data
            X = df[features].fillna(df[features].mean())
            y = df[target_col].astype(int) if target_col in df else (df['qol_score'] < df['qol_score'].median()).astype(int)
            
            if len(X) < self.min_sample_size:
                logger.warning(f"⚠️ Insufficient samples: {len(X)} < {self.min_sample_size}")
                return {"status": "partial_success", "error": "Insufficient samples"}

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            
            # Evaluate
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            
            if accuracy < self.min_accuracy:
                logger.warning(f"⚠️ Low model accuracy: {accuracy:.2f}")
                return {"status": "partial_success", "error": f"Low accuracy: {accuracy:.2f}"}

            # Example prediction (e.g., for median feature values)
            example_input = np.array(X.median().values).reshape(1, -1)
            risk_prob = float(model.predict_proba(example_input)[0][1])
            
            logger.info(f"✅ Readmission risk prediction complete: Accuracy {accuracy:.2f}, Risk {risk_prob:.2f}")
            return {
                "status": "success",
                "model_type": "logistic_regression",
                "accuracy": accuracy,
                "risk_prediction": risk_prob,
                "features": features
            }
        except Exception as e:
            logger.error(f"❌ Readmission prediction failed: {e}")
            return {"status": "failed", "error": str(e)}

    def predict_treatment_response(self, df: pd.DataFrame, treatment_col: str, outcome_col: str = "qol_score") -> Dict:
        """Predict treatment response (e.g., QoL improvement)"""
        logger.info(f"📈 Predicting treatment response for {treatment_col} on {outcome_col}")
        try:
            if df.empty or treatment_col not in df or outcome_col not in df:
                logger.warning("⚠️ Invalid data or columns")
                return {"status": "failed", "error": "Invalid input"}

            results = []
            for treatment in df[treatment_col].unique():
                subset = df[df[treatment_col] == treatment]
                if len(subset) < self.min_sample_size:
                    continue
                
                X = subset[['age']].fillna(subset['age'].mean())
                y = (subset[outcome_col] > subset[outcome_col].median()).astype(int)
                
                model = LogisticRegression(max_iter=1000)
                model.fit(X, y)
                
                example_input = np.array([[50]])  # Example: age=50
                response_prob = float(model.predict_proba(example_input)[0][1])
                
                results.append({
                    "treatment": treatment,
                    "response_probability": response_prob,
                    "sample_size": len(subset)
                })
            
            if not results:
                logger.warning("⚠️ No valid treatments for prediction")
                return {"status": "partial_success", "error": "No valid results"}

            logger.info(f"✅ Treatment response prediction complete: {len(results)} treatments")
            return {
                "status": "success",
                "model_type": "logistic_regression",
                "predictions": results
            }
        except Exception as e:
            logger.error(f"❌ Treatment response prediction failed: {e}")
            return {"status": "failed", "error": str(e)}
