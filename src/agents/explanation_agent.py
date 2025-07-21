
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)

class ExplanationAgent:
    def __init__(self, config: Dict):
        self.config = config
        self.llm = ChatGroq(
            model="llama3-70b-8192",
            api_key=config["llm"]["api_key"],
            max_tokens=config["llm"]["max_tokens"],
            temperature=config["llm"]["temperature"]
        )
        self.explanation_prompt = PromptTemplate.from_template(
            """
            Generate a plain-language clinical explanation for: {query}.
            Analysis results: {analysis}.
            Include:
            - Clear summary of findings (avoid jargon)
            - Actionable recommendations (PROM-aligned, cost-efficient)
            - Bias warnings if detected
            - Confidence score (0-100%)
            """
        )
        self.visualization_prompt = PromptTemplate.from_template(
            """
            Based on analysis: {analysis}, select the best chart type (bar, line, heatmap, network) for: {query}.
            Return JSON: {'chart_type': str, 'data': dict, 'title': str, 'annotations': list}
            """
        )
        self.report_prompt = PromptTemplate.from_template(
            """
            Generate a report for {stakeholder} (clinician/admin) based on: {analysis}.
            Include key findings, recommendations, and clinical context.
            """
        )

    def generate_explanation(self, query: str, analysis_results: Dict, session_state: Optional[Dict] = None) -> Dict:
        """Generate explanations, visualizations, and reports from analysis"""
        logger.info(f"📚 Stage 7: Explanation Generation for query: {query}")
        try:
            # Session state for multi-turn
            session_state = session_state or {}

            # Generate plain-language explanation
            chain = LLMChain(llm=self.llm, prompt=self.explanation_prompt)
            explanation = chain.invoke({"query": query, "analysis": json.dumps(analysis_results)})["text"].strip()

            # Confidence scoring
            confidence = self._calculate_confidence(analysis_results)
            if confidence < 50:
                logger.warning("⚠️ Low confidence in explanation")
                explanation += "\nNote: Low confidence due to sparse data or weak statistical significance."

            # Fairness warnings
            fairness_note = ""
            if analysis_results.get("fairness_metrics", {}).get("disparate_impact", 1.0) < 0.8:
                fairness_note = "Warning: Potential demographic bias detected. Consider balanced interventions."

            # Generate visualization
            viz_chain = LLMChain(llm=self.llm, prompt=self.visualization_prompt)
            viz_result = viz_chain.invoke({"query": query, "analysis": json.dumps(analysis_results)})["text"]
            try:
                viz_data = json.loads(viz_result)
                fig = self._create_plotly_chart(viz_data)
                chart_html = fig.to_html(full_html=False)
            except Exception as e:
                logger.warning(f"⚠️ Visualization generation failed: {e}")
                chart_html = None

            # Generate reports
            clinician_report = self._generate_report("clinician", analysis_results)
            admin_report = self._generate_report("admin", analysis_results)

            # Pre-handover validation
            if not explanation or confidence < 50:
                logger.warning("⚠️ Incomplete explanation")
                return {
                    "status": "partial_success",
                    "error": "Incomplete explanation or low confidence",
                    "query": query,
                    "timestamp": datetime.now().isoformat()
                }

            result = {
                "status": "success",
                "query": query,
                "explanation": explanation,
                "confidence": confidence,
                "fairness_note": fairness_note,
                "visualization": chart_html,
                "reports": {
                    "clinician": clinician_report,
                    "admin": admin_report
                },
                "timestamp": datetime.now().isoformat()
            }
            logger.info(f"✅ Explanation generated: Confidence {confidence}%")
            return result

        except Exception as e:
            logger.error(f"❌ Explanation generation failed: {e}")
            return {"status": "failed", "error": str(e), "query": query}

    def _calculate_confidence(self, analysis_results: Dict) -> float:
        """Calculate confidence score based on analysis results"""
        data_avail = 0.3 if analysis_results.get("record_counts", {}).get("patients", 0) > 1000 else 0.1
        stats_strength = 0.4 if any(r.get("p_value", 1.0) < 0.05 for r in analysis_results.get("pathways", [])) else 0.2
        prom_weight = 0.2 if analysis_results.get("prom_scores", []) else 0.1
        quality = 0.1 if analysis_results.get("quality_score", 0.0) >= self.config["data_processing"]["quality_score_min"] else 0.05
        return min(100, (data_avail + stats_strength + prom_weight + quality) * 100)

    def _create_plotly_chart(self, viz_data: Dict) -> Any:
        """Create Plotly chart from visualization data"""
        chart_type = viz_data.get("chart_type", "bar")
        data = viz_data.get("data", {})
        title = viz_data.get("title", "Analysis Results")
        
        if chart_type == "bar":
            fig = px.bar(data_frame=pd.DataFrame(data), x="x", y="y", title=title)
        elif chart_type == "line":
            fig = px.line(data_frame=pd.DataFrame(data), x="x", y="y", title=title)
        elif chart_type == "heatmap":
            if "z" in data:
                fig = px.density_heatmap(data_frame=pd.DataFrame(data), x="x", y="y", z="z", title=title)
            else:
                logger.warning("⚠️ Heatmap requires z data")
                return go.Figure()
        else:  # Fallback to bar
            fig = px.bar(data_frame=pd.DataFrame(data), x="x", y="y", title=title)
        
        for annotation in viz_data.get("annotations", []):
            fig.add_annotation(text=annotation, x=0.5, y=0.95, showarrow=False)
        return fig

    def _generate_report(self, stakeholder: str, analysis_results: Dict) -> str:
        """Generate stakeholder-specific report"""
        chain = LLMChain(llm=self.llm, prompt=self.report_prompt)
        report = chain.invoke({"stakeholder": stakeholder, "analysis": json.dumps(analysis_results)})["text"].strip()
        return report

    def handle_refinement(self, query: str, analysis_results: Dict, refinement: str) -> Dict:
        """Handle multi-turn query refinements"""
        logger.info(f"🔄 Handling refinement: {refinement}")
        updated_query = f"{query} - Refinement: {refinement}"
        return self.generate_explanation(updated_query, analysis_results)
