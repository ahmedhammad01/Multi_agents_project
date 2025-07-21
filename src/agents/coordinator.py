
import logging
import os
import shutil
from typing import Dict, Any, Optional
from src.utils.config_loader import load_config
from src.agents.data_agent import DataProcessingAgent
from src.agents.graph_agent import GraphBuildingAgent
from src.agents.analysis_agent import PathwayAnalysisAgent
from src.agents.explanation_agent import ExplanationAgent
from src.utils.json_validator import validate_payload
from neo4j import GraphDatabase
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Define state for query mode
class QueryState(BaseModel):
    query: str = ""
    session_state: Dict = {}
    analysis_results: Dict = {}
    explanation_results: Dict = {}

class WorkflowCoordinator:
    def __init__(self, config: Dict):
        self.config = config
        self.data_agent = DataProcessingAgent(config)
        self.graph_agent = GraphBuildingAgent(config)
        self.analysis_agent = PathwayAnalysisAgent(config)
        self.explanation_agent = ExplanationAgent(config)
        self.neo4j_uri = config["neo4j"]["uri"]
        self.neo4j_user = config["neo4j"]["user"]
        self.neo4j_password = config["neo4j"]["password"]

    def check_graph_readiness(self) -> bool:
        """Check if Neo4j graph is ready for queries"""
        try:
            driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
            with driver.session() as session:
                result = session.run(
                    "MATCH (n) RETURN count(n) as node_count, count(()-[]->()) as rel_count, max(n.timestamp) as last_updated"
                )
                data = result.single()
                if data is None:
                    logger.warning("⚠️ No data returned from Neo4j query")
                    return False
                node_count = data["node_count"]
                rel_count = data["rel_count"]
                last_updated = data["last_updated"]
                if node_count < self.config["neo4j"]["node_count_min"]:
                    logger.warning(f"⚠️ Insufficient nodes: {node_count}")
                    return False
                if rel_count < self.config["neo4j"]["edge_count_min"]:
                    logger.warning(f"⚠️ Insufficient relationships: {rel_count}")
                    return False
                if last_updated:
                    from datetime import datetime
                    age_days = (datetime.now().timestamp() - last_updated.timestamp()) / (24 * 3600)
                    if age_days > 7:
                        logger.warning(f"⚠️ Graph stale: {age_days:.1f} days old")
                        return False
            driver.close()
            logger.info("✅ Graph is ready")
            return True
        except Exception as e:
            logger.error(f"❌ Graph readiness check failed: {e}")
            return False

    def reset_data(self):
        """Reset data directories for fresh preprocessing"""
        try:
            for directory in [self.config["project"]["data_dir"] + "/raw", self.config["project"]["data_dir"] + "/processed"]:
                if os.path.exists(directory):
                    shutil.rmtree(directory)
                    logger.info(f"🧹 Cleared directory: {directory}")
            os.makedirs(self.config["project"]["data_dir"] + "/raw", exist_ok=True)
            os.makedirs(self.config["project"]["data_dir"] + "/processed", exist_ok=True)
        except Exception as e:
            logger.error(f"❌ Data reset failed: {e}")

    def run_preprocessing(self) -> Dict:
        """Run preprocessing pipeline (stages 1-5)"""
        logger.info("🚀 Starting preprocessing pipeline")
        try:
            # Initialize state for Data Agent
            from src.agents.data_agent import DataState
            initial_state = DataState(
                raw_data={},
                cleaned_data={},
                prom_scores=[],
                validation_results={},
                issues_detected=[],
                quality_score=0.0,
                attempts={},
                flags=[]
            )
            
            # Run Data Processing Agent
            data_result = self.data_agent.run(initial_state)
            if data_result["status"] != "success":
                logger.warning(f"⚠️ Data processing issues: {data_result['issues']}")
                return data_result
            
            # Validate A2A payload
            schema = {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "quality_score": {"type": "number"},
                    "issues": {"type": "array"},
                    "record_counts": {"type": "object"}
                },
                "required": ["status", "quality_score", "issues"]
            }
            validate_payload(data_result, schema)
            
            # Run Graph Building Agent
            graph_result = self.graph_agent.build_graph(data_result.get("cleaned_data", {}))
            if graph_result["status"] != "success":
                logger.warning(f"⚠️ Graph building issues: {graph_result['error']}")
                return graph_result
            
            logger.info("✅ Preprocessing completed successfully")
            return {
                "status": "success",
                "quality_score": data_result["quality_score"],
                "graph_nodes": graph_result.get("nodes", 0),
                "graph_relationships": graph_result.get("relationships", 0)
            }
        except Exception as e:
            logger.error(f"❌ Preprocessing pipeline failed: {e}")
            return {"status": "failed", "error": str(e)}

    def run_query(self, query: str, session_state: Optional[Dict] = None) -> Dict:
        """Run query pipeline (stages 6-7)"""
        logger.info(f"🚀 Processing query: {query}")
        try:
            # Check graph readiness
            if not self.check_graph_readiness():
                logger.warning("⚠️ Graph not ready, triggering preprocessing")
                preprocess_result = self.run_preprocessing()
                if preprocess_result["status"] != "success":
                    return preprocess_result
            
            # Initialize query state
            state = QueryState(query=query, session_state=session_state or {})
            
            # Run Pathway Analysis Agent
            analysis_result = self.analysis_agent.analyze_pathways(query, state.session_state)
            if analysis_result["status"] != "success":
                logger.warning(f"⚠️ Analysis issues: {analysis_result['error']}")
                return analysis_result
            
            # Validate A2A payload
            schema = {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "query": {"type": "string"},
                    "hypotheses": {"type": "array"},
                    "pathways": {"type": "array"},
                    "predictions": {"type": "array"},
                    "fairness_metrics": {"type": "object"}
                },
                "required": ["status", "query", "hypotheses", "pathways"]
            }
            validate_payload(analysis_result, schema)
            state.analysis_results = analysis_result
            
            # Run Explanation Agent
            explanation_result = self.explanation_agent.generate_explanation(query, analysis_result, state.session_state)
            if explanation_result["status"] != "success":
                logger.warning(f"⚠️ Explanation issues: {explanation_result['error']}")
                return explanation_result
            
            state.explanation_results = explanation_result
            logger.info("✅ Query pipeline completed successfully")
            return {
                "status": "success",
                "query": query,
                "analysis": analysis_result,
                "explanation": explanation_result
            }
        except Exception as e:
            logger.error(f"❌ Query pipeline failed: {e}")
            return {"status": "failed", "error": str(e)}
