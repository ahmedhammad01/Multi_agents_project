
import streamlit as st
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class QueryForm:
    def __init__(self):
        self.query_key = "query_input"
        self.refinement_key = "refinement_input"
        self.example_queries = [
            "What are the main risk factors for diabetes complications?",
            "Which patients need urgent follow-up care?",
            "Compare the effectiveness of insulin pumps vs. injections for Type 1 diabetes",
            "What medication adjustments could improve outcomes?",
            "Show trends in COPD patient outcomes"
        ]

    def render(self) -> Tuple[str, str]:
        """Render the query input form in Streamlit"""
        logger.info("📋 Rendering query form")
        try:
            # Query input section
            st.markdown("### Submit a Query")
            st.write("Enter a natural language query or select an example below:")
            
            # Dropdown for example queries
            example_query = st.selectbox(
                "Example Queries",
                ["Select an example..."] + self.example_queries,
                key="example_query_select"
            )
            
            # Text input for custom query
            query = st.text_input(
                "Custom Query",
                placeholder="e.g., What are the main risk factors for diabetes complications?",
                key=self.query_key
            )
            
            # Use example query if selected and no custom query
            if example_query != "Select an example..." and not query:
                query = example_query

            # Refinement input for multi-turn interaction
            st.markdown("### Refine Query (Optional)")
            st.write("Add details to refine the query (e.g., 'Focus on patients over 60'):")
            refinement = st.text_input(
                "Refinement",
                placeholder="e.g., Focus on patients over 60",
                key=self.refinement_key
            )

            # Submit button
            submit = st.button("Submit Query")
            
            if submit and query:
                logger.info(f"✅ Query submitted: {query}")
                if refinement:
                    logger.info(f"🔄 Refinement: {refinement}")
                return query, refinement
            else:
                return "", ""

        except Exception as e:
            logger.error(f"❌ Query form rendering failed: {e}")
            st.error(f"Failed to render query form: {e}")
            return "", ""
