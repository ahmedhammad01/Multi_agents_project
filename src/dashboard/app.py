
import streamlit as st
import logging
from src.utils.config_loader import load_config
from src.agents.coordinator import WorkflowCoordinator
from src.dashboard.components.query_form import QueryForm
from src.dashboard.components.chart_renderer import ChartRenderer
from src.dashboard.components.report_generator import ReportGenerator

logger = logging.getLogger(__name__)

def run_dashboard():
    """Run the Streamlit AG-UI dashboard"""
    # Setup logging
    st.set_page_config(page_title="Healthcare Pathway Evaluation", layout="wide")
    logger.info("🏥 Starting AG-UI Dashboard")

    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        st.error(f"Failed to load configuration: {e}")
        logger.error(f"❌ Config load failed: {e}")
        return

    # Initialize coordinator
    try:
        coordinator = WorkflowCoordinator(config)
    except Exception as e:
        st.error(f"Failed to initialize coordinator: {e}")
        logger.error(f"❌ Coordinator initialization failed: {e}")
        return

    # Dashboard layout
    st.title("Healthcare Pathway Evaluation Platform")
    st.markdown("Query treatment pathways, view insights, and generate reports for diabetes and COPD.")

    # Query form
    query_form = QueryForm()
    query, refinement = query_form.render()

    # Session state for multi-turn queries
    if 'session_state' not in st.session_state:
        st.session_state['session_state'] = {}

    # Process query
    if query:
        with st.spinner("Processing query..."):
            try:
                result = coordinator.run_query(query, st.session_state['session_state'])
                if result["status"] != "success":
                    st.error(f"Query failed: {result.get('error', 'Unknown error')}")
                    logger.warning(f"⚠️ Query failed: {result.get('error')}")
                    return
                
                # Update session state for refinements
                st.session_state['session_state'] = result.get("session_state", {})

                # Display results
                st.subheader("Analysis Results")
                analysis = result["analysis"]
                explanation = result["explanation"]

                # Render analysis
                st.markdown(f"**Query**: {query}")
                if analysis.get("hypotheses"):
                    st.markdown("**Hypotheses**:")
                    for hyp in analysis["hypotheses"]:
                        st.markdown(f"- {hyp}")
                
                if analysis.get("pathways"):
                    st.markdown("**Pathways**:")
                    for pathway in analysis["pathways"]:
                        st.markdown(f"- {pathway['treatment']}: Impact {pathway['impact']:.2f} (p={pathway['p_value']:.3f})")

                # Render visualizations
                chart_renderer = ChartRenderer()
                if explanation.get("visualization"):
                    st.subheader("Visualizations")
                    chart_renderer.render(explanation["visualization"])

                # Render explanation
                st.subheader("Explanation")
                st.markdown(explanation["explanation"])
                st.markdown(f"**Confidence**: {explanation['confidence']}%")
                if explanation.get("fairness_note"):
                    st.warning(explanation["fairness_note"])

                # Render reports
                report_generator = ReportGenerator()
                st.subheader("Reports")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Clinician Report**")
                    report_generator.render(explanation["reports"]["clinician"], "clinician")
                with col2:
                    st.markdown("**Administrator Report**")
                    report_generator.render(explanation["reports"]["admin"], "admin")

                # Handle refinements
                if refinement:
                    with st.spinner("Processing refinement..."):
                        refined_result = coordinator.explanation_agent.handle_refinement(query, analysis, refinement)
                        if refined_result["status"] == "success":
                            st.subheader("Refined Explanation")
                            st.markdown(refined_result["explanation"])
                            st.markdown(f"**Confidence**: {refined_result['confidence']}%")
                            if refined_result.get("visualization"):
                                chart_renderer.render(refined_result["visualization"])
                        else:
                            st.error(f"Refinement failed: {refined_result.get('error')}")
            except Exception as e:
                st.error(f"Query processing failed: {e}")
                logger.error(f"❌ Query processing failed: {e}")

if __name__ == "__main__":
    run_dashboard()
