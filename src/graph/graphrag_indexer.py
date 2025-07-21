
import logging
import json
from typing import Dict, Any, List
from neo4j import GraphDatabase
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from src.graph.neo4j_driver import Neo4jDriver

logger = logging.getLogger(__name__)

class GraphRAGIndexer:
    def __init__(self, config: Dict):
        self.config = config
        self.neo4j_driver = Neo4jDriver(config)
        self.llm = ChatGroq(
            model="llama3-70b-8192",
            api_key=config["llm"]["api_key"],
            max_tokens=config["llm"]["max_tokens"],
            temperature=config["llm"]["temperature"]
        )
        self.entity_prompt = PromptTemplate.from_template(
            """
            Analyze data: {data_summary}.
            Extract entities (e.g., patients, treatments, outcomes) and relationships.
            Return JSON: {"entities": [{"type": str, "id": str, "attributes": dict}], "relationships": [{"source": str, "target": str, "type": str}]}
            """
        )
        self.summary_prompt = PromptTemplate.from_template(
            """
            Summarize graph data: {graph_data}.
            Generate hierarchical summaries for clinical insights (e.g., treatment efficacy, patient outcomes).
            Return JSON: {"summaries": [{"level": str, "content": str}]}
            """
        )

    def index_graph(self, graph_data: Dict[str, Any]) -> Dict:
        """Index graph with entities, relationships, and summaries using GraphRAG"""
        logger.info("🕸️ Indexing graph with GraphRAG")
        try:
            if not self.neo4j_driver.connect():
                logger.error("❌ Neo4j connection failed")
                return {"status": "failed", "error": "Neo4j connection failed"}

            # Extract entities and relationships
            data_summary = (
                f"Patients: {len(graph_data.get('patients', []))}, "
                f"Treatments: {len(graph_data.get('treatments', []))}, "
                f"PROMs: {len(graph_data.get('prom_scores', []))}"
            )
            chain = LLMChain(llm=self.llm, prompt=self.entity_prompt)
            entity_result = chain.invoke({"data_summary": data_summary})["text"]

            try:
                entities_rels = json.loads(entity_result)
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ Entity extraction failed: {e}")
                entities_rels = {"entities": [], "relationships": []}

            # Add entities and relationships to Neo4j
            if self.neo4j_driver.driver is None:
                logger.error("❌ Neo4j driver is not initialized")
                return {"status": "failed", "error": "Neo4j driver is not initialized"}
            with self.neo4j_driver.driver.session() as session:
                for entity in entities_rels.get("entities", []):
                    session.run(
                        "MERGE (n:$type {id: $id}) SET n += $attributes",
                        type=entity["type"],
                        id=entity["id"],
                        attributes=entity["attributes"]
                    )

                for rel in entities_rels.get("relationships", []):
                    session.run(
                        """
                        MERGE (source:Node {id: $source})
                        MERGE (target:Node {id: $target})
                        MERGE (source)-[r:{rel_type}]->(target)
                        """,
                        source=rel["source"],
                        target=rel["target"],
                        rel_type=rel["type"]
                    )

            # Generate hierarchical summaries
            chain = LLMChain(llm=self.llm, prompt=self.summary_prompt)
            graph_data_str = json.dumps(graph_data, default=str)
            summary_result = chain.invoke({"graph_data": graph_data_str})["text"]

            try:
                summaries = json.loads(summary_result)
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ Summary extraction failed: {e}")
                summaries = {"summaries": []}
            # Store summaries as Neo4j properties
            if self.neo4j_driver.driver is None:
                logger.error("❌ Neo4j driver is not initialized")
                return {"status": "failed", "error": "Neo4j driver is not initialized"}
            with self.neo4j_driver.driver.session() as session:
                for summary in summaries.get("summaries", []):
                    session.run(
                        """
                        MERGE (n:Summary {level: $level})
                        SET n.content = $content
                        """,
                        level=summary["level"],
                        content=summary["content"]
                    )
            # Validate indexing
            if self.neo4j_driver.driver is None:
                logger.error("❌ Neo4j driver is not initialized")
                return {"status": "failed", "error": "Neo4j driver is not initialized"}
            with self.neo4j_driver.driver.session() as session:
                result = session.run("MATCH (n:Summary) RETURN count(n) as summary_count")
                single_result = result.single()
                summary_count = single_result["summary_count"] if single_result else 0
                if summary_count < len(summaries.get("summaries", [])) * 0.9:
                    logger.warning("⚠️ Incomplete indexing")
                    return {
                        "status": "partial_success",
                        "error": "Incomplete indexing",
                        "summary_count": summary_count
                    }
                    logger.warning("⚠️ Incomplete indexing")
                    return {
                        "status": "partial_success",
                        "error": "Incomplete indexing",
                        "summary_count": summary_count
                    }

            logger.info(f"✅ GraphRAG indexing complete: {summary_count} summaries")
            return {
                "status": "success",
                "entities": len(entities_rels.get("entities", [])),
                "relationships": len(entities_rels.get("relationships", [])),
                "summaries": summary_count
            }

        except Exception as e:
            logger.error(f"❌ GraphRAG indexing failed: {e}")
            return {"status": "failed", "error": str(e)}

    def close(self):
        """Close Neo4j connection"""
        self.neo4j_driver.close()
  