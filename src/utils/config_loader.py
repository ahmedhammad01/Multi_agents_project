
import logging
from logging import config
import os
from typing import Dict
import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/config.yaml") -> Dict:
    """Load configuration from YAML file and environment variables"""
    try:
        # Load environment variables from .env
        load_dotenv("config/.env")
        
        # Load YAML config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override with environment variables (e.g., sensitive credentials)
        config["bigquery"]["project_id"] = os.getenv("GOOGLE_CLOUD_PROJECT", config["bigquery"]["project_id"])
        config["neo4j"]["user"] = os.getenv("NEO4J_USER", config["neo4j"]["user"])
        config["neo4j"]["password"] = os.getenv("NEO4J_PASSWORD", config["neo4j"]["password"])
        config["llm"]["api_key"] = os.getenv("GROQ_API_KEY", config["llm"]["api_key"])        
        # Validate critical fields
        if not config["bigquery"]["project_id"]:
            raise ValueError("BigQuery project ID not set")
        if not config["neo4j"]["password"]:
            raise ValueError("Neo4j password not set")
        
        logger.info("✅ Configuration loaded successfully")
        return config
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        raise
