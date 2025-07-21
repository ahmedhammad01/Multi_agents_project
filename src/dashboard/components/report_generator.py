
import streamlit as st
import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self):
        pass

    def render(self, report_content: str, stakeholder: str) -> None:
        """Render a stakeholder-specific report in Streamlit"""
        logger.info(f"📋 Rendering {stakeholder} report")
        try:
            if not report_content:
                logger.warning(f"⚠️ No {stakeholder} report content provided")
                st.warning(f"No {stakeholder} report available")
                return
            
            # Render report as markdown
            st.markdown("### Report Content")
            st.markdown(report_content)
            
            # Add download button for report
            st.download_button(
                label=f"Download {stakeholder.capitalize()} Report",
                data=report_content,
                file_name=f"{stakeholder}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
            logger.info(f"✅ {stakeholder.capitalize()} report rendered successfully")
        except Exception as e:
            logger.error(f"❌ {stakeholder.capitalize()} report rendering failed: {e}")
            st.error(f"Failed to render {stakeholder} report: {e}")

    def render_multiple(self, reports: Dict[str, str]) -> None:
        """Render multiple reports for different stakeholders"""
        logger.info(f"📋 Rendering {len(reports)} reports")
        try:
            if not reports:
                logger.warning("⚠️ No reports provided")
                st.warning("No reports available")
                return
            
            for stakeholder, content in reports.items():
                with st.container():
                    st.markdown(f"**{stakeholder.capitalize()} Report**")
                    self.render(content, stakeholder)
        except Exception as e:
            logger.error(f"❌ Multiple report rendering failed: {e}")
            st.error(f"Failed to render reports: {e}")
