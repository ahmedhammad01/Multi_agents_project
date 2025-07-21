
# src/graph/__init__.py
"""
Graph package for the Agentic AI Platform.
Contains utilities for:
- Connecting to and querying Neo4j knowledge graph
- GraphRAG indexing and summarization for contextual queries
"""

from .neo4j_driver import Neo4jDriver
from .graphrag_indexer import GraphRAGIndexer

__all__ = [
    "Neo4jDriver",
    "GraphRAGIndexer",
]
