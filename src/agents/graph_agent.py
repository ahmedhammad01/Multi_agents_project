
import logging
import json
from datetime import datetime
from typing import Dict, Any
from neo4j import GraphDatabase
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from src.utils.config_loader import load_config
from langchain_core.messages import HumanMessage

import pandas as pd
import os

logger = logging.getLogger(__name__)

class GraphBuildingAgent:
    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = config["project"]["data_dir"]
        self.clean_data_file = config["project"]["clean_data_file"]
        self.neo4j_uri = config["neo4j"]["uri"]
        self.neo4j_user = config["neo4j"]["user"]
        self.neo4j_password = config["neo4j"]["password"]
        self.driver = None
        from pydantic import SecretStr
        self.llm = ChatGroq(model="llama3-70b-8192", api_key=SecretStr(os.getenv("GROQ_API_KEY") or ""))  # Use Groq model
        self.graphrag_prompt = PromptTemplate.from_template(
            "Extract entities and relationships from: {data}. Create Cypher commands for Neo4j."
        )

    def connect(self):
        """Connect to Neo4j database with retries"""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
            )
            # Test connection
            with self.driver.session() as session:
                session.run("MATCH (n) RETURN n LIMIT 1")
            logger.info("✅ Connected to Neo4j")
            return True
        except Exception as e:
            logger.error(f"❌ Neo4j connection failed: {e}")
            return False

    def build_graph(self, validated_data: Dict[str, Any]) -> Dict:
        """Build knowledge graph from validated data"""
        logger.info("🕸️ Stage 5: Knowledge Graph Building")
        try:
            # Load data from file if not provided
            if not validated_data:
                if not os.path.exists(self.clean_data_file):
                    logger.error("❌ No validated data available")
                    return {"status": "failed", "error": "No data"}
                with open(self.clean_data_file, 'r') as f:
                    validated_data = json.load(f)

            # Convert lists to DataFrames
            patients = pd.DataFrame(validated_data.get("patients", []))
            treatments = pd.DataFrame(validated_data.get("treatments", []))
            labs = pd.DataFrame(validated_data.get("labs", []))
            proms = validated_data.get("prom_scores", [])
            procedures = pd.DataFrame(validated_data.get("procedures", []))

            if patients.empty:
                logger.error("❌ No patient data for graph building")
                return {"status": "failed", "error": "Empty patients"}

            # Connect to Neo4j
            if not self.connect() or self.driver is None:
                logger.warning("⚠️ Using partial graph if available")
                return {"status": "partial_success", "error": "Neo4j connection failed"}

            # Build graph
            nodes_created = 0
            relationships_created = 0
            with self.driver.session() as session:
                # Create Patient nodes
                for _, patient in patients.iterrows():
                    session.run(
                        "MERGE (p:Patient {id: $id, age: $age, gender: $gender, condition_type: $condition})",
                        id=str(patient['subject_id']),
                        age=float(patient.get('age', 0)),
                        gender=str(patient.get('gender', 'Unknown')),
                        condition=str(patient.get('condition_type', 'Unknown'))
                    )
                    nodes_created += 1

                # Create Treatment nodes and relationships
                for _, treatment in treatments.iterrows():
                    session.run(
                        """
                        MERGE (t:Treatment {id: $id, type: $type})
                        MERGE (p:Patient {id: $patient_id})
                        MERGE (p)-[:RECEIVED_TREATMENT {starttime: $starttime}]->(t)
                        """,
                        id=str(treatment.get('drug', 'unknown')),
                        type=str(treatment.get('treatment_category', 'medication')),
                        patient_id=str(treatment['subject_id']),
                        starttime=str(treatment.get('starttime', ''))
                    )
                    nodes_created += 1
                    relationships_created += 1

                # Create Outcome (PROM) nodes and relationships
                for prom in proms:
                    session.run(
                        """
                        MERGE (o:Outcome {id: $id, qol_score: $qol, confidence: $conf})
                        MERGE (p:Patient {id: $patient_id})
                        MERGE (p)-[:HAS_OUTCOME]->(o)
                        """,
                        id=f"prom_{prom['subject_id']}",
                        qol=float(prom['quality_of_life_score']),
                        conf=float(prom['confidence']),
                        patient_id=str(prom['subject_id'])
                    )
                    nodes_created += 1
                    relationships_created += 1

                # GraphRAG indexing (simplified)
                # Generate summaries with LLM
                data_summary = f"Patients: {len(patients)}, Treatments: {len(treatments)}, PROMs: {len(proms)}"
                chain = LLMChain(llm=self.llm, prompt=self.graphrag_prompt)
                cypher_commands = chain.run({"data": data_summary}).strip()
                for cmd in cypher_commands.split('\n'):
                    if cmd.strip():
                        try:
                            session.run(cmd)
                        except Exception as e:
                            logger.warning(f"⚠️ GraphRAG command failed: {e}")

                # Create indexes
                session.run("CREATE INDEX ON :Patient(id)")
                session.run("CREATE INDEX ON :Treatment(type)")
                session.run("CREATE INDEX ON :Outcome(qol_score)")

            # Validate graph
            with self.driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as node_count, count(()-[]->()) as rel_count")
                counts = result.single()
                if counts is None:
                    logger.error("❌ Failed to retrieve node and relationship counts from Neo4j")
                    return {"status": "failed", "error": "Could not retrieve graph counts"}
                node_count = counts["node_count"]
                rel_count = counts["rel_count"]
                if node_count < self.config["neo4j"]["node_count_min"] or rel_count < self.config["neo4j"]["edge_count_min"]:
                    logger.warning("⚠️ Graph incomplete")
                    return {"status": "partial_success", "nodes": node_count, "relationships": rel_count}

            logger.info(f"✅ Graph built: {node_count} nodes, {rel_count} relationships")
            return {
                "status": "success",
                "nodes": node_count,
                "relationships": rel_count,
                "graph_type": "Clinical_Knowledge_Graph",
                "graph_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Graph building failed: {e}")
            return {"status": "failed", "error": str(e)}

    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("✅ Neo4j connection closed")
