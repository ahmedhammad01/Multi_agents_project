
import logging
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class Neo4jDriver:
    def __init__(self, config: Dict):
        self.config = config
        self.uri = config["neo4j"]["uri"]
        self.user = config["neo4j"]["user"]
        self.password = config["neo4j"]["password"]
        self.max_retries = config["neo4j"]["max_retries"]
        self.driver = None

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=10))
    def connect(self) -> bool:
        """Connect to Neo4j database with retries"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            # Test connection
            with self.driver.session() as session:
                session.run("MATCH (n) RETURN n LIMIT 1")
            logger.info("✅ Connected to Neo4j database")
            return True
        except Exception as e:
            logger.error(f"❌ Neo4j connection failed: {e}")
            return False

    def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute a Cypher query and return results"""
        if not self.driver:
            logger.error("❌ Neo4j driver not connected")
            return []
        
        try:
            with self.driver.session() as session:
                # type: ignore is used here to suppress the type checker error for query argument
                result = session.run(query, params or {})  # type: ignore
                records = [record.data() for record in result]
                logger.info(f"✅ Executed query: {query[:50]}... ({len(records)} records)")
                return records
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            return []

    def create_index(self, label: str, property: str) -> bool:
        """Create an index on a node property"""
        try:
            query = f"CREATE INDEX ON :{label}({property})"
            self.execute_query(query)
            logger.info(f"✅ Created index on {label}.{property}")
            return True
        except Exception as e:
            logger.error(f"❌ Index creation failed: {e}")
            return False

    def check_graph_readiness(self) -> Dict:
        """Check graph readiness (node/edge counts, freshness)"""
        if not self.driver:
            logger.error("❌ Neo4j driver not connected")
            return {"is_ready": False, "error": "Neo4j driver not connected"}
        try:
            with self.driver.session() as session:
                result = session.run(
                    "MATCH (n) RETURN count(n) as node_count, count(()-[]->()) as rel_count, max(n.timestamp) as last_updated"
                )
                data = result.single()
                if data is None:
                    logger.error("❌ No data returned from graph readiness query")
                    return {"is_ready": False, "error": "No data returned from graph readiness query"}
                node_count = data["node_count"]
                rel_count = data["rel_count"]
                last_updated = data["last_updated"]
                
                readiness = {
                    "node_count": node_count,
                    "rel_count": rel_count,
                    "is_ready": (
                        node_count >= self.config["neo4j"]["node_count_min"] and
                        rel_count >= self.config["neo4j"]["edge_count_min"]
                    )
                }
                if last_updated:
                    from datetime import datetime
                    age_days = (datetime.now().timestamp() - last_updated.timestamp()) / (24 * 3600)
                    readiness["age_days"] = age_days
                    readiness["is_ready"] = readiness["is_ready"] and age_days <= 7
                
                logger.info(f"✅ Graph readiness: {node_count} nodes, {rel_count} relationships, ready={readiness['is_ready']}")
                return readiness
        except Exception as e:
            logger.error(f"❌ Graph readiness check failed: {e}")
            return {"is_ready": False, "error": str(e)}

    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("✅ Neo4j connection closed")
            self.driver = None
