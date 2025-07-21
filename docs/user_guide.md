# User Guide for the Agentic AI Platform Dashboard

## Introduction

The Agentic AI Platform Dashboard (AG-UI) is a user-friendly, web-based interface designed for clinicians and administrators to evaluate treatment pathways for chronic conditions like diabetes and COPD. Built with Streamlit, it allows users to submit natural language queries, view interactive visualizations, and access tailored reports. This guide explains how to use the dashboard, ensuring quick onboarding (<10 minutes) and clear, jargon-free interaction.

## Prerequisites

- **Access**: Ensure you have login credentials for the dashboard (if restricted).
- **Browser**: Use a modern browser (e.g., Chrome, Firefox).
- **System Setup**: The platform must be preprocessed (`python main.py --preprocess`) to load MIMIC-IV data and build the knowledge graph.
- **Training**: No prior technical knowledge required; this guide suffices.

## Accessing the Dashboard

1. **Launch the Dashboard**:
   - Run the command:
     ```bash
     streamlit run src/dashboard/app.py
     ```
   - Open your browser to `http://localhost:8501` (or the configured port).

2. **Login (if applicable)**:
   - If authentication is enabled, enter your credentials (contact the system administrator for access).

## Using the Dashboard

### 1. Submitting a Query

The dashboard's main page displays a query form for analyzing treatment pathways.

- **Select an Example Query**:
  - Use the dropdown to choose from predefined queries, such as:
    - "What are the main risk factors for diabetes complications?"
    - "Compare the effectiveness of insulin pumps vs. injections for Type 1 diabetes"
    - "Which patients need urgent follow-up care?"
  - This helps new users explore common analyses.

- **Enter a Custom Query**:
  - Type a natural language question in the "Custom Query" text box (e.g., "Show trends in COPD patient outcomes").
  - Keep queries clear and focused on pathways, risks, or outcomes.

- **Submit**:
  - Click the "Submit Query" button to process the query.

### 2. Refining Queries (Optional)

For more specific analysis, add a refinement to your query:

- **Refinement Input**:
  - In the "Refinement" text box, enter details like "Focus on patients over 60" or "Compare by gender."
  - This supports multi-turn interactions for deeper insights.

- **Submit Again**:
  - Click "Submit Query" to process the refined query.

### 3. Viewing Results

After submitting a query, the dashboard displays:

- **Analysis Results**:
  - **Hypotheses**: Generated insights (e.g., "High HbA1c predicts complications").
  - **Pathways**: Treatment comparisons (e.g., "Insulin: 15% reduction in complications, p=0.01").
  - **Fairness Notes**: Bias warnings if detected (e.g., "Gender disparity detected").

- **Visualizations**:
  - Interactive charts (bar, line, heatmap, or network) showing pathway impacts, correlations, or risk predictions.
  - Hover over charts for details; use filters (if available) to drill down.

- **Explanation**:
  - A plain-language summary of findings (e.g., "Insulin pumps reduce HbA1c by 0.5% due to consistent dosing").
  - Confidence score (0-100%) indicates reliability.

### 4. Downloading Reports

The dashboard provides stakeholder-specific reports:

- **Clinician Report**:
  - Actionable insights for patient care (e.g., recommended treatments, risk factors).
  - Download as a text file by clicking "Download Clinician Report."

- **Administrator Report**:
  - Focuses on cost-efficiency and resource allocation (e.g., cost-effective treatments).
  - Download as a text file by clicking "Download Administrator Report."

### Example Workflow

1. **Query**: Select "What are the main risk factors for diabetes complications?"
2. **Results**: View hypotheses (e.g., "High HbA1c increases risk"), pathways, and a bar chart of risk factors.
3. **Refine**: Enter "Focus on patients over 60" and resubmit.
4. **Review**: Check updated explanation and download the clinician report.
5. **Act**: Use insights to prioritize follow-up care or adjust treatments.

## Tips for Effective Use

- **Be Specific**: Clear queries (e.g., "Compare insulin vs. metformin for Type 2 diabetes") yield better results.
- **Use Refinements**: Narrow results by demographics or conditions for targeted insights.
- **Check Confidence**: Low confidence (<50%) may indicate sparse data; consider refining or reprocessing.
- **Report Bias**: If fairness warnings appear, consult the system administrator for mitigation steps.

## Troubleshooting

- **Dashboard Not Loading**:
  - Ensure `streamlit run src/dashboard/app.py` is running.
  - Check `logs/app.log` for errors.
- **Query Fails**:
  - Verify preprocessing was run (`python main.py --preprocess`).
  - Ensure BigQuery (`gcloud auth application-default login`) and Neo4j are accessible.
- **No Visualizations**:
  - Check if query returned valid results; refine query if needed.
- **Contact Support**: Reach out to the system administrator for persistent issues.

## Ethical Considerations

- **Data Privacy**: Patient data is anonymized (PII masked).
- **Fairness**: Bias warnings are displayed if detected; recommendations aim to be equitable.
- **Transparency**: All results include confidence scores and audit logs (in `logs/`).

## Additional Resources

- **Architecture**: See `docs/architecture.md` for system design details.
- **API Reference**: Check `docs/api_reference.md` for code documentation.
- **Support**: Contact the project team for assistance.

---

**Version**: 1.0.0  
**Last Updated**: July 20, 2025
