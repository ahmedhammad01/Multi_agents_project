import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from google.cloud import bigquery
from neo4j import GraphDatabase
from langchain_groq import ChatGroq
from pydantic import SecretStr
from langchain_core.messages import HumanMessage
import dotenv
from src.utils.config_loader import load_config
config = load_config("config/config.yaml")

dotenv.load_dotenv("config/.env")

def test_bigquery_connection():
    """Test connection to Google BigQuery"""
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    try:
        client = bigquery.Client(project=project_id)
        datasets = list(client.list_datasets())
        assert datasets is not None
        print(f"BigQuery connection successful. Found {len(datasets)} datasets.")
    except Exception as e:
        pytest.fail(f"BigQuery connection failed: {e}")

def test_neo4j_connection():
    """Test connection to Neo4j"""
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")
    if password is None:
        pytest.fail("Neo4j password is not set in environment variables.")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            result = session.run("RETURN 1 AS test")
            record = result.single()
            assert record is not None and record["test"] == 1
        print("Neo4j connection successful.")
    except Exception as e:
        pytest.fail(f"Neo4j connection failed: {e}")

def test_groq_llm_connection():
    """Test connection to Groq LLM via LangChain"""
    api_key = os.getenv("LLM_API_KEY")
    model = os.environ.get("GROQ_MODEL", "llama3-70b-8192")
    try:
        llm = ChatGroq(model=model, api_key=SecretStr(api_key) if api_key else None)
        message = HumanMessage(content="Hello, world!")
        response = llm.invoke([message])
        assert response is not None and hasattr(response, "content") and len(response.content) > 0
        print("Groq LLM connection successful.")
        print(f"api: {api_key}")
    except Exception as e:
        pytest.fail(f"Groq LLM connection failed: {e}")

if __name__ == "__main__":
    config["llm"]["api_key"] = os.getenv("GROQ_API_KEY", config["llm"]["api_key"])
    print(f"API_KEY={config['llm']['api_key']}")
    test_bigquery_connection()
    test_neo4j_connection()
    test_groq_llm_connection()