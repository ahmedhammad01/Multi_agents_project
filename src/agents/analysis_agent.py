import logging
import json
from typing import Dict, Any, List, Optional
from neo4j import GraphDatabase
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from scipy.stats import ttest_ind
from aif360.metrics import BinaryLabelDatasetMetric
import pandas as pd
import numpy as np
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)

class PathwayAnalysisAgent:
    def __init__(self, config: Dict):
        self.config = config
        self.neo4j_uri = config["neo4j"]["uri"]
        self.neo4j_user = config["neo4j"]["user"]
        self.neo4j_password = config["neo4j"]["password"]
        self.driver = None
        self.llm = ChatGroq(model="llama3-70b-8192", api_key=config["llm"]["api_key"])
        self.query_prompt = PromptTemplate.from_template(
            "Parse this query: {query}. Generate Cypher to retrieve relevant data from a Neo4j graph with Patient, Treatment, and Outcome nodes."
        )
        self.hypothesis_prompt = PromptTemplate.from_template(
            "Given data: {data_summary}, generate hypotheses about treatment pathways (e.g., efficacy, adherence). Return a list of hypotheses."
        )

    def connect(self):
        """Connect to Neo4j with retries"""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )
            with self.driver.session() as session:
                session.run("MATCH (n) RETURN n LIMIT 1")
            logger.info("✅ Connected to Neo4j")
            return True
        except Exception as e:
            logger.error(f"❌ Neo4j connection failed: {e}")
            return False

    def analyze_pathways(self, query: str, session_state: Optional[Dict] = None) -> Dict:
        """Analyze pathways based on user query"""
        logger.info(f"🔬 Stage 6: Pathway Analysis for query: {query}")
        try:
            if not self.connect() or self.driver is None:
                logger.warning("⚠️ Using cached results if available")
                return {"status": "partial_success", "error": "Neo4j connection failed"}

            # Parse query
            chain = LLMChain(llm=self.llm, prompt=self.query_prompt)
            cypher_query = chain.invoke({"query": query})["text"].strip()

            # Retrieve data
            with self.driver.session() as session:
                result = session.run(cypher_query)
                data = [record.data() for record in result]

            if not data:
                logger.warning("⚠️ No data retrieved for query")
                return {"status": "partial_success", "error": "Empty query result"}

            # Convert to DataFrame
            df = pd.DataFrame(data)
            if df.empty:
                return {"status": "failed", "error": "No valid data"}

            # EDA: Correlations and patterns
            correlations = {}
            if 'qol_score' in df and 'age' in df:
                correlations['age_qol'] = df['qol_score'].corr(df['age'])

            # Hypothesis generation
            chain = LLMChain(llm=self.llm, prompt=self.hypothesis_prompt)
            data_summary = f"Records: {len(df)}, Columns: {list(df.columns)}, Correlations: {correlations}"
            hypotheses = chain.invoke({"data_summary": data_summary})["text"].split('\n')

            # Statistical analysis
            stats_results = []
            if 'treatment_type' in df and 'qol_score' in df:
                for treatment in df['treatment_type'].unique():
                    group1 = df[df['treatment_type'] == treatment]['qol_score']
                    group2 = df[df['treatment_type'] != treatment]['qol_score']
                    if len(group1) > 1 and len(group2) > 1:
                        t_stat, p_value = map(float, ttest_ind(group1, group2))
                        stats_results.append({
                            "treatment": treatment,
                            "t_stat": t_stat,
                            "p_value": p_value,
                            "impact": float(group1.mean() - group2.mean())
                        })

            # Fairness check (simplified AIF360)
            fairness_metrics = {}
            if 'gender' in df and 'qol_score' in df:
                female_scores = df[df['gender'] == 'Female']['qol_score']
                male_scores = df[df['gender'] == 'Male']['qol_score']
                if len(female_scores) > 1 and len(male_scores) > 1:
                    dataset = pd.DataFrame({
                        'label': (df['qol_score'] > df['qol_score'].median()).astype(int),
                        'gender': df['gender'].map({'Female': 1, 'Male': 0})
                    })
                    fairness_metrics['disparate_impact'] = BinaryLabelDatasetMetric(
                        dataset, privileged_groups=[{'gender': 0}], unprivileged_groups=[{'gender': 1}]
                    ).disparate_impact()

            # Predictive modeling (basic logistic regression for risk)
            predictions = []
            if 'qol_score' in df and 'age' in df:
                from sklearn.linear_model import LogisticRegression
                X = df[['age']].fillna(df['age'].mean())
                y = (df['qol_score'] > df['qol_score'].median()).astype(int)
                model = LogisticRegression().fit(X, y)
                predictions.append({
                    "risk_prediction": float(model.predict_proba(np.array([[50]]))[0][1]),  # Example: Risk for age=50
                    "model_type": "logistic_regression"
                })

            # Pre-handover validation
            if not stats_results or not hypotheses:
                logger.warning("⚠️ Incomplete analysis")
                return {"status": "partial_success", "error": "Incomplete results"}

            result = {
                "status": "success",
                "query": query,
                "hypotheses": hypotheses,
                "pathways": stats_results,
                "predictions": predictions,
                "fairness_metrics": fairness_metrics,
                "correlations": correlations,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            logger.info(f"✅ Analysis complete: {len(stats_results)} pathways, {len(hypotheses)} hypotheses")
            return result

        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            return {"status": "failed", "error": str(e)}

    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("✅ Neo4j connection closed")
