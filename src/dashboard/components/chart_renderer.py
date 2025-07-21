
import streamlit as st
import streamlit.components.v1 as components
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class ChartRenderer:
    def __init__(self):
        pass

    def render(self, chart_html: str) -> None:
        """Render a Plotly chart in Streamlit"""
        logger.info("📊 Rendering chart in dashboard")
        try:
            if not chart_html:
                logger.warning("⚠️ No chart data provided")
                st.warning("No visualization available for this query")
                return
            # Render Plotly chart HTML
            components.html(chart_html, height=500)
            logger.info("✅ Chart rendered successfully")
            logger.info("✅ Chart rendered successfully")
        except Exception as e:
            logger.error(f"❌ Chart rendering failed: {e}")
            st.error(f"Failed to render chart: {e}")

    def render_multiple(self, charts: list) -> None:
        """Render multiple charts in Streamlit"""
        logger.info(f"📊 Rendering {len(charts)} charts")
        try:
            if not charts:
                logger.warning("⚠️ No charts provided")
                st.warning("No visualizations available")
                return
            
            for i, chart_html in enumerate(charts):
                with st.container():
                    st.markdown(f"**Chart {i+1}**")
                    self.render(chart_html)
        except Exception as e:
            logger.error(f"❌ Multiple chart rendering failed: {e}")
            st.error(f"Failed to render charts: {e}")
