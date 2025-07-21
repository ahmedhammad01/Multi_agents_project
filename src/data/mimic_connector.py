
import logging
import pandas as pd
from google.cloud import bigquery
from tenacity import retry, stop_after_attempt, wait_exponential
import time
import os
import json

logger = logging.getLogger(__name__)

class MIMICDataConnector:
    def __init__(self, config: dict):
        self.config = config
        self.project_id = config["bigquery"]["project_id"]
        self.dataset_id = config["bigquery"]["dataset_id"]
        self.max_retries = config["bigquery"]["max_retries"]
        self.retry_delay = config["bigquery"]["retry_delay"]
        self.client = None

    def _execute_query_with_retry(self, query: str, operation_name: str = "query") -> pd.DataFrame:
        """Execute BigQuery query with retry logic"""
        if not self.client:
            logger.error("❌ BigQuery client not connected. Call connect() first.")
            return pd.DataFrame()
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                logger.info(f"🔍 {operation_name} (attempt {attempt + 1}/{self.max_retries})")
                result_df = self.client.query(query).to_dataframe()
                return result_df
            except Exception as e:
                last_error = str(e).lower()
                transient_errors = ['timeout', 'deadline exceeded', 'internal error', 'service unavailable', 'rate limit']
                permanent_errors = ['not found', 'permission denied', 'invalid', 'syntax error']
                if any(err in last_error for err in permanent_errors):
                    logger.error(f"❌ Permanent error in {operation_name}: {e}")
                    break
                elif any(err in last_error for err in transient_errors) and attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"⚠️ Transient error in {operation_name}: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ Max retries exceeded for {operation_name}: {e}")
                    break
        return pd.DataFrame()

    def connect(self) -> bool:
        """Initialize BigQuery client using gcloud auth"""
        try:
            self.client = bigquery.Client(project=self.project_id)
            test_query = "SELECT 1 AS test_connection"
            self.client.query(test_query).result()
            logger.info(f"✅ Connected to BigQuery project: {self.project_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to BigQuery: {e}")
            logger.info("💡 Ensure gcloud auth is set: gcloud auth application-default login")
            return False

    def get_diabetes_patients(self, limit: int = 15000) -> pd.DataFrame:
        """Fetch diabetes patients from MIMIC-IV (ICD-10: E10-E14)"""
        if not self.client:
            logger.error("❌ BigQuery client not connected")
            return pd.DataFrame()
        
        query = f"""
        SELECT DISTINCT
            p.subject_id,
            p.gender,
            p.anchor_age as age,
            d.icd_code,
            d.icd_version,
            a.hadm_id,
            a.admittime,
            a.dischtime,
            a.admission_type,
            a.discharge_location,
            CASE 
                WHEN d.icd_code LIKE 'E10%' THEN 'Type 1 Diabetes'
                WHEN d.icd_code LIKE 'E11%' THEN 'Type 2 Diabetes'
                WHEN d.icd_code LIKE 'E12%' THEN 'Malnutrition-related Diabetes'
                WHEN d.icd_code LIKE 'E13%' THEN 'Other Specified Diabetes'
                WHEN d.icd_code LIKE 'E14%' THEN 'Unspecified Diabetes'
                ELSE 'Diabetes'
            END as condition_type
        FROM `{self.dataset_id}.patients` p
        JOIN `{self.dataset_id}.diagnoses_icd` d ON p.subject_id = d.subject_id
        JOIN `{self.dataset_id}.admissions` a ON d.hadm_id = a.hadm_id
        WHERE d.icd_code LIKE 'E1%' 
        AND d.icd_version = 10
        AND p.anchor_age >= 18
        LIMIT {limit}
        """
        
        try:
            df = self._execute_query_with_retry(query, f"Querying unique diabetes patients (limit: {limit})")
            logger.info(f"✅ Retrieved {len(df)} unique diabetes patient records")
            return df
        except Exception as e:
            logger.error(f"❌ Diabetes query failed: {e}")
            return pd.DataFrame()

    def get_treatment_data(self, subject_ids: list) -> pd.DataFrame:
        """Fetch treatment/prescription data for specific patients"""
        if not self.client or not subject_ids:
            logger.error("❌ Client not connected or no subject IDs")
            return pd.DataFrame()
        
        ids_str = ','.join([str(id) for id in subject_ids])
        query = f"""
        SELECT 
            subject_id,
            hadm_id,
            drug,
            dose_val_rx,
            dose_unit_rx,
            starttime,
            stoptime,
            'medication' as treatment_category
        FROM `{self.dataset_id}.prescriptions`
        WHERE subject_id IN ({ids_str})
        AND (drug LIKE '%insulin%' OR drug LIKE '%metformin%' OR drug LIKE '%glyburide%' OR drug LIKE '%glipizide%')
        """
        
        try:
            df = self._execute_query_with_retry(query, f"Querying treatment data for {len(subject_ids)} patients")
            logger.info(f"✅ Retrieved {len(df)} treatment records")
            return df
        except Exception as e:
            logger.error(f"❌ Treatment query failed: {e}")
            return pd.DataFrame()

    def get_lab_results(self, subject_ids: list) -> pd.DataFrame:
        """Fetch lab results (outcomes) for patients
        Focus on HbA1c and glucose levels
        """
        if not self.client or not subject_ids:
            logger.error("❌ Client not connected or no subject IDs")
            return pd.DataFrame()
            
        ids_str = ','.join([str(id) for id in subject_ids])
        
        query = f"""
        SELECT 
            l.subject_id,
            l.hadm_id,
            l.itemid,
            l.charttime,
            l.value,
            l.valuenum,
            l.valueuom,
            d.label as lab_test
        FROM `{self.dataset_id}.labevents` l
        JOIN `{self.dataset_id}.d_labitems` d ON l.itemid = d.itemid
        WHERE l.subject_id IN ({ids_str})
        AND (d.label LIKE '%glucose%' OR d.label LIKE '%hemoglobin%' OR d.label LIKE '%hba1c%')
        AND l.valuenum IS NOT NULL
        """
        
        try:
            df = self._execute_query_with_retry(query, f"Querying lab results for {len(subject_ids)} patients")
            logger.info(f"✅ Retrieved {len(df)} lab result records")
            return df
        except Exception as e:
            logger.error(f"❌ Lab results query failed: {e}")
            return pd.DataFrame()

    def get_discharge_notes(self, subject_ids: list, limit_per_patient: int = 5) -> pd.DataFrame:
        """Get discharge summaries and clinical notes for patients
        These contain valuable clinical narratives for pathway analysis
        """
        if not self.client or not subject_ids:
            logger.error("❌ Client not connected or no subject IDs")
            return pd.DataFrame()
            
        ids_str = ','.join([str(id) for id in subject_ids])
        
        query = f"""
        SELECT 
            subject_id,
            hadm_id,
            note_id,
            note_type,
            note_seq,
            charttime,
            storetime,
            text
        FROM `{self.dataset_id}.discharge`
        WHERE subject_id IN ({ids_str})
        AND text IS NOT NULL
        AND LENGTH(text) > 100
        ORDER BY subject_id, charttime DESC
        """
        
        try:
            df = self._execute_query_with_retry(query, f"Querying discharge notes for {len(subject_ids)} patients")
            
            # Limit notes per patient to avoid overwhelming data
            if not df.empty and limit_per_patient > 0:
                df_limited = df.groupby('subject_id').head(limit_per_patient).reset_index(drop=True)
                logger.info(f"✅ Retrieved {len(df_limited)} clinical notes (limited to {limit_per_patient} per patient)")
                return df_limited
            else:
                logger.info(f"✅ Retrieved {len(df)} clinical notes")
                return df
        except Exception as e:
            logger.error(f"❌ Discharge notes query failed: {e}")
            return pd.DataFrame()

    def get_procedures(self, subject_ids: list) -> pd.DataFrame:
        """Get procedure data for patients
        Useful for understanding treatment interventions
        """
        if not self.client or not subject_ids:
            logger.error("❌ Client not connected or no subject IDs")
            return pd.DataFrame()
        
        ids_str = ','.join([str(id) for id in subject_ids])
        
        query = f"""
        SELECT 
            p.subject_id,
            p.hadm_id,
            p.icd_code,
            p.icd_version,
            d.long_title as procedure_name
        FROM `{self.dataset_id}.procedures_icd` p
        JOIN `{self.dataset_id}.d_icd_procedures` d ON p.icd_code = d.icd_code 
                                                    AND p.icd_version = d.icd_version
        WHERE p.subject_id IN ({ids_str})
        """
        
        try:
            df = self._execute_query_with_retry(query, f"Querying procedures for {len(subject_ids)} patients")
            logger.info(f"✅ Retrieved {len(df)} procedure records")
            return df
        except Exception as e:
            logger.error(f"❌ Procedures query failed: {e}")
            return pd.DataFrame()

    def save_raw_data(self, data_dict: dict, filepath: str) -> bool:
        """Save raw data to JSON"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            serializable_data = {}
            for key, df in data_dict.items():
                if isinstance(df, pd.DataFrame):
                    serializable_data[key] = df.to_dict('records')
                else:
                    serializable_data[key] = df
            
            with open(filepath, 'w') as f:
                json.dump(serializable_data, f, default=str, indent=2)
            
            logger.info(f"✅ Raw data saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save data: {e}")
            return False
