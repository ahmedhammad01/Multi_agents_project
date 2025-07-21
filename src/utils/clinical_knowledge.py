
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class ClinicalKnowledge:
    def __init__(self, config: Dict):
        self.config = config
        # Predefined clinical guidelines (simplified for demo; extend with API if needed)
        self.guidelines = {
            "diabetes": {
                "HbA1c_target": 7.0,  # Target for good control
                "readmission_risk_factors": ["high HbA1c (>9)", "age >65", "comorbidities"],
                "recommended_treatments": ["insulin", "metformin", "SGLT2 inhibitors"]
            },
            "copd": {
                "qol_target": 50.0,  # Minimum quality-of-life score
                "readmission_risk_factors": ["low FEV1", "frequent exacerbations", "age >70"],
                "recommended_treatments": ["inhaled corticosteroids", "bronchodilators"]
            }
        }

    def validate_analysis(self, analysis_results: Dict, condition: str = "diabetes") -> Dict:
        """Validate analysis results against clinical guidelines"""
        logger.info(f"📚 Validating analysis for {condition}")
        try:
            if not analysis_results:
                logger.warning("⚠️ No analysis results provided")
                return {"status": "failed", "error": "No results"}

            guideline = self.guidelines.get(condition.lower(), {})
            if not guideline:
                logger.warning(f"⚠️ No guidelines for {condition}")
                return {"status": "partial_success", "error": f"No guidelines for {condition}"}

            validation_results = {
                "status": "success",
                "validations": [],
                "warnings": []
            }

            # Validate pathways
            for pathway in analysis_results.get("pathways", []):
                treatment = pathway.get("treatment", "").lower()
                p_value = pathway.get("p_value", 1.0)
                impact = pathway.get("impact", 0.0)
                
                # Check if treatment is recommended
                if treatment not in [t.lower() for t in guideline["recommended_treatments"]]:
                    validation_results["warnings"].append(
                        f"Treatment '{treatment}' not in recommended guidelines for {condition}"
                    )
                
                # Check statistical significance and impact
                if p_value >= 0.05 or abs(impact) < 0.10:
                    validation_results["warnings"].append(
                        f"Treatment '{treatment}' has weak significance (p={p_value:.3f}, impact={impact:.2f})"
                    )

            # Validate predictions
            for prediction in analysis_results.get("predictions", []):
                risk = prediction.get("risk_prediction", 0.0)
                if risk > 0.5 and "high HbA1c" in guideline["readmission_risk_factors"]:
                    validation_results["validations"].append(
                        f"High readmission risk ({risk:.2f}) aligns with guideline risk factors"
                    )

            # Check for fairness issues
            fairness_metrics = analysis_results.get("fairness_metrics", {})
            if fairness_metrics.get("overall", {}).get("bias_detected", False):
                validation_results["warnings"].append(
                    "Demographic bias detected; review fairness metrics"
                )

            logger.info(f"✅ Clinical validation complete: {len(validation_results['validations'])} validations, {len(validation_results['warnings'])} warnings")
            return validation_results
        except Exception as e:
            logger.error(f"❌ Clinical validation failed: {e}")
            return {"status": "failed", "error": str(e)}

    def enrich_explanation(self, explanation: str, analysis_results: Dict, condition: str = "diabetes") -> str:
        """Enrich explanation with clinical context"""
        logger.info(f"📚 Enriching explanation for {condition}")
        try:
            guideline = self.guidelines.get(condition.lower(), {})
            if not guideline:
                logger.warning(f"⚠️ No guidelines for {condition}")
                return explanation

            enriched = explanation
            # Add guideline context
            enriched += f"\n\n**Clinical Context**: According to guidelines for {condition}, "
            if condition == "diabetes":
                enriched += f"target HbA1c is {guideline['HbA1c_target']}%. "
                enriched += f"Key risk factors include: {', '.join(guideline['readmission_risk_factors'])}."
            elif condition == "copd":
                enriched += f"target QoL score is ≥{guideline['qol_target']}. "
                enriched += f"Key risk factors include: {', '.join(guideline['readmission_risk_factors'])}."
            
            # Add treatment recommendations
            enriched += f"\nRecommended treatments: {', '.join(guideline['recommended_treatments'])}."
            
            logger.info("✅ Explanation enriched with clinical context")
            return enriched
        except Exception as e:
            logger.error(f"❌ Explanation enrichment failed: {e}")
            return explanation
