
import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List
import numpy as np
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class VisualizationUtils:
    def __init__(self, config: Dict):
        self.config = config
        self.response_time = config["dashboard"]["query_response_time"]

    def create_chart(self, data: Dict, chart_type: str, title: str, annotations: Optional[List[Dict]] = None) -> str:
        """Generate a Plotly chart based on data and type"""
        logger.info(f"📊 Generating {chart_type} chart: {title}")
        try:
            df = pd.DataFrame(data.get("data", []))
            if df.empty:
                logger.warning("⚠️ Empty data for visualization")
                return ""

            fig = None
            if chart_type == "bar":
                fig = px.bar(df, x=data.get("x", df.columns[0]), y=data.get("y", df.columns[1]), title=title)
            elif chart_type == "line":
                fig = px.line(df, x=data.get("x", df.columns[0]), y=data.get("y", df.columns[1]), title=title)
            elif chart_type == "heatmap":
                if "z" in data:
                    fig = px.density_heatmap(df, x=data.get("x", df.columns[0]), y=data.get("y", df.columns[1]), 
                                             z=data.get("z", df.columns[2]), title=title)
                else:
                    logger.warning("⚠️ Heatmap requires z data")
                    return ""
            elif chart_type == "network":
                fig = self._create_network_diagram(data, title)
            else:
                logger.warning(f"⚠️ Unsupported chart type: {chart_type}, defaulting to bar")
                fig = px.bar(df, x=data.get("x", df.columns[0]), y=data.get("y", df.columns[1]), title=title)

            # Add annotations
            if annotations:
                for ann in annotations:
                    fig.add_annotation(
                        text=ann.get("text", ""),
                        x=ann.get("x", 0.5),
                        y=ann.get("y", 0.95),
                        showarrow=ann.get("showarrow", False)
                    )

            # Customize layout for clinician usability
            fig.update_layout(
                xaxis_title=data.get("x_label", data.get("x", "X")),
                yaxis_title=data.get("y_label", data.get("y", "Y")),
                showlegend=True,
                template="plotly_white"  # Clean theme
            )

            # Ensure response time constraint
            chart_html = fig.to_html(full_html=False, include_plotlyjs=False)
            logger.info(f"✅ Chart generated: {chart_type}")
            return chart_html
        except Exception as e:
            logger.error(f"❌ Chart generation failed: {e}")
            return ""

    def _create_network_diagram(self, data: Dict, title: str) -> go.Figure:
        """Generate a network diagram for graph relationships"""
        logger.info("🌐 Generating network diagram")
        try:
            nodes = data.get("nodes", [])
            edges = data.get("relationships", [])
            if not nodes or not edges:
                logger.warning("⚠️ No nodes or edges for network diagram")
                return go.Figure()

            # Create node positions
            node_x, node_y = [], []
            for node in nodes:
                node_x.append(node.get("x", np.random.random()))
                node_y.append(node.get("y", np.random.random()))
            
            # Create edge traces
            edge_x, edge_y = [], []
            for edge in edges:
                source = edge.get("source")
                target = edge.get("target")
                source_idx = next((i for i, n in enumerate(nodes) if n["id"] == source), None)
                target_idx = next((i for i, n in enumerate(nodes) if n["id"] == target), None)
                if source_idx is not None and target_idx is not None:
                    edge_x.extend([node_x[source_idx], node_x[target_idx], None])
                    edge_y.extend([node_y[source_idx], node_y[target_idx], None])

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color="#888"),
                hoverinfo="none",
                mode="lines"
            )

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode="markers+text",
                hoverinfo="text",
                marker=dict(size=10, color="blue"),
                text=[node["id"] for node in nodes]
            )

            fig = go.Figure(data=[edge_trace, node_trace],
                           layout=go.Layout(
                               title=title,
                               showlegend=False,
                               hovermode="closest",
                               xaxis=dict(showgrid=False, zeroline=False),
                               yaxis=dict(showgrid=False, zeroline=False)
                           ))
            logger.info("✅ Network diagram generated")
            return fig
        except Exception as e:
            logger.error(f"❌ Network diagram generation failed: {e}")
            return go.Figure()

    def generate_dashboard_charts(self, analysis_results: Dict, query: str) -> List[str]:
        """Generate multiple charts for dashboard display"""
        logger.info(f"📊 Generating dashboard charts for query: {query}")
        charts = []
        try:
            # Example charts based on analysis
            if "pathways" in analysis_results:
                chart_data = {
                    "data": [
                        {"x": p["treatment"], "y": p["impact"]}
                        for p in analysis_results["pathways"]
                    ],
                    "x_label": "Treatment",
                    "y_label": "Impact (%)",
                    "chart_type": "bar",
                    "title": f"Pathway Impact: {query}",
                    "annotations": [{"text": f"Significant: {len(analysis_results['pathways'])} pathways"}]
                }
                charts.append(self.create_chart(chart_data, "bar", chart_data["title"], chart_data["annotations"]))

            if "correlations" in analysis_results:
                chart_data = {
                    "data": [
                        {"x": k, "y": v}
                        for k, v in analysis_results["correlations"].items()
                    ],
                    "x_label": "Feature",
                    "y_label": "Correlation",
                    "chart_type": "bar",
                    "title": f"Correlations: {query}"
                }
                charts.append(self.create_chart(chart_data, "bar", chart_data["title"]))

            logger.info(f"✅ Generated {len(charts)} dashboard charts")
            return [c for c in charts if c]  # Filter out empty charts
        except Exception as e:
            logger.error(f"❌ Dashboard chart generation failed: {e}")
            return []