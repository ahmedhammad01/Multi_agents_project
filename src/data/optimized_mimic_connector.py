import pandas as pd
import logging
from typing import Optional, Tuple, Dict, Any
import time

logger = logging.getLogger(__name__)

class QualityGate:
    """Quality gates to ensure production-ready data"""
    
    MIN_LAB_COVERAGE = 0.70
    MIN_TREATMENT_COVERAGE = 0.30
    MIN_PROCEDURE_COVERAGE = 0.10  # Lowered for MIMIC-IV
    MIN_NOTES_COVERAGE = 0.50
    MAX_DROP_RATE = 0.10
    MIN_QUALITY_SCORE = 0.95
    MAX_BIAS_THRESHOLD = 0.15
    
    @staticmethod
    def validate_coverage(stats: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate data coverage meets minimum requirements"""
        failures = []
        
        if stats.get('lab_coverage', 0) < QualityGate.MIN_LAB_COVERAGE:
            failures.append(f"Lab coverage {stats['lab_coverage']:.1%} < {QualityGate.MIN_LAB_COVERAGE:.1%}")
        if stats.get('treatment_coverage', 0) < QualityGate.MIN_TREATMENT_COVERAGE:
            failures.append(f"Treatment coverage {stats['treatment_coverage']:.1%} < {QualityGate.MIN_TREATMENT_COVERAGE:.1%}")
        if stats.get('procedure_coverage', 0) < QualityGate.MIN_PROCEDURE_COVERAGE:
            failures.append(f"Procedure coverage {stats['procedure_coverage']:.1%} < {QualityGate.MIN_PROCEDURE_COVERAGE:.1%}")
        if stats.get('notes_coverage', 0) < QualityGate.MIN_NOTES_COVERAGE:
            failures.append(f"Notes coverage {stats['notes_coverage']:.1%} < {QualityGate.MIN_NOTES_COVERAGE:.1%}")
        
        if failures:
            return False, "; ".join(failures)
        return True, "Coverage requirements met"

class OptimizedMIMICConnector:
    def __init__(self, client, dataset_id="mimiciv_hosp"):
        self.client = client
        self.dataset_id = dataset_id

    def get_comprehensive_diabetes_cohort(self, target_patients=5000):
        """Get diabetes patients with guaranteed comprehensive data coverage"""
        if not self.client:
            logger.error("❌ BigQuery client not connected")
            return pd.DataFrame()
            
        query = f"""
        WITH comprehensive_patients AS (
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
                END as condition_type,
                (SELECT COUNT(DISTINCT l.itemid) 
                 FROM `{self.dataset_id}.labevents` l 
                 JOIN `{self.dataset_id}.d_labitems` di ON l.itemid = di.itemid
                 WHERE l.subject_id = p.subject_id 
                 AND (di.label LIKE '%glucose%' OR di.label LIKE '%hemoglobin a1c%' 
                      OR di.label LIKE '%hba1c%')
                 AND l.valuenum IS NOT NULL AND l.valuenum > 0
                ) as lab_types,
                (SELECT COUNT(*) 
                 FROM `{self.dataset_id}.prescriptions` pr
                 WHERE pr.subject_id = p.subject_id 
                 AND pr.hadm_id = a.hadm_id
                 AND (LOWER(pr.drug) LIKE '%insulin%' OR LOWER(pr.drug) LIKE '%metformin%')
                ) as treatment_count,
                (SELECT COUNT(*) 
                 FROM `{self.dataset_id}.procedures_icd` proc
                 JOIN `{self.dataset_id}.d_icd_procedures` dp ON proc.icd_code = dp.icd_code
                 WHERE proc.subject_id = p.subject_id 
                 AND proc.hadm_id = a.hadm_id
                 AND (dp.long_title LIKE '%diabetes%' OR dp.long_title LIKE '%glucose%')
                ) as procedure_count,
                (SELECT COUNT(*) 
                 FROM `{self.dataset_id}.note.discharge` dn
                 WHERE dn.subject_id = p.subject_id 
                 AND dn.hadm_id = a.hadm_id
                 AND LENGTH(dn.text) > 500
                 AND (LOWER(dn.text) LIKE '%diabetes%' OR LOWER(dn.text) LIKE '%glucose%')
                ) as note_count
            FROM `{self.dataset_id}.patients` p
            JOIN `{self.dataset_id}.diagnoses_icd` d ON p.subject_id = d.subject_id
            JOIN `{self.dataset_id}.admissions` a ON d.hadm_id = a.hadm_id
            WHERE d.icd_code LIKE 'E1%' 
            AND d.icd_version = 10
            AND p.anchor_age >= 18 
            AND p.anchor_age <= 89
        ),
        qualified_patients AS (
            SELECT *,
                   CASE 
                       WHEN lab_types >= 3 THEN 0.25 ELSE 0 
                   END +
                   CASE 
                       WHEN treatment_count >= 1 THEN 0.25 ELSE 0 
                   END +
                   CASE 
                       WHEN procedure_count >= 1 THEN 0.25 ELSE 0 
                   END +
                   CASE 
                       WHEN note_count >= 1 THEN 0.25 ELSE 0 
                   END as comprehensiveness_score
            FROM comprehensive_patients
        )
        SELECT * FROM qualified_patients
        WHERE comprehensiveness_score >= 0.75
        ORDER BY comprehensiveness_score DESC, lab_types DESC, treatment_count DESC
        LIMIT {target_patients}
        """
        
        start_time = time.time()
        try:
            df = self.client.query(query).to_dataframe()
            query_time = time.time() - start_time
            
            if len(df) == 0:
                logger.error("❌ No comprehensive diabetes patients found")
                return pd.DataFrame()
                
            lab_coverage = len(df[df['lab_types'] >= 3]) / len(df)
            treatment_coverage = len(df[df['treatment_count'] >= 1]) / len(df)
            procedure_coverage = len(df[df['procedure_count'] >= 1]) / len(df)
            note_coverage = len(df[df['note_count'] >= 1]) / len(df)
            
            logger.info(f"✅ Retrieved {len(df)} comprehensive diabetes patients ({query_time:.1f}s)")
            logger.info(f"📊 Coverage: Labs {lab_coverage:.1%}, Treatments {treatment_coverage:.1%}, "
                       f"Procedures {procedure_coverage:.1%}, Notes {note_coverage:.1%}")
            
            stats = {
                'lab_coverage': lab_coverage,
                'treatment_coverage': treatment_coverage, 
                'procedure_coverage': procedure_coverage,
                'notes_coverage': note_coverage
            }
            
            passed, message = QualityGate.validate_coverage(stats)
            if not passed:
                logger.error(f"❌ Quality gate failed: {message}")
                raise ValueError(f"Data quality insufficient: {message}")
                
            logger.info(f"✅ Quality gates passed: {message}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Comprehensive query failed: {e}")
            return pd.DataFrame()

class BiasDetector:
    """Advanced bias detection and mitigation for healthcare data"""
    
    @staticmethod
    def detect_and_mitigate_bias(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Detect demographic bias and apply mitigation strategies"""
        if df.empty:
            return df, {}
            
        original_count = len(df)
        bias_metrics = {}
        
        gender_counts = df['gender'].value_counts(normalize=True)
        gender_min_pct = gender_counts.min()
        bias_metrics['gender_min_representation'] = gender_min_pct
        
        if gender_min_pct < QualityGate.MAX_BIAS_THRESHOLD:
            logger.warning(f"⚠️ Gender bias detected: minimum representation {gender_min_pct:.1%}")
            min_count = df['gender'].value_counts().min()
            df = df.groupby('gender').apply(
                lambda x: x.sample(min(len(x), int(min_count * 1.2)))
            ).reset_index(drop=True)
            logger.info(f"✅ Gender bias mitigated: {original_count} → {len(df)} patients")
        
        df['age_group'] = pd.cut(df['age'], bins=[18, 35, 50, 65, 89], labels=['18-35', '36-50', '51-65', '66-89'])
        age_counts = df['age_group'].value_counts(normalize=True)
        age_min_pct = age_counts.min()
        bias_metrics['age_min_representation'] = age_min_pct
        
        if age_min_pct < QualityGate.MAX_BIAS_THRESHOLD:
            logger.warning(f"⚠️ Age bias detected: minimum representation {age_min_pct:.1%}")
            min_age_count = df['age_group'].value_counts().min()
            df = df.groupby('age_group').apply(
                lambda x: x.sample(min(len(x), int(min_age_count * 1.3)))
            ).reset_index(drop=True)
            logger.info(f"✅ Age bias mitigated: balanced across age groups")
        
        condition_counts = df['condition_type'].value_counts(normalize=True)
        condition_min_pct = condition_counts.min()
        bias_metrics['condition_min_representation'] = condition_min_pct
        
        final_count = len(df)
        bias_metrics['mitigation_drop_rate'] = (original_count - final_count) / original_count
        
        logger.info(f"📊 Bias mitigation complete: {original_count} → {final_count} patients "
                   f"({bias_metrics['mitigation_drop_rate']:.1%} reduction)")
                   
        return df, bias_metrics

class Neo4jPrep:
    """Prepare data structures for Neo4j knowledge graph integration"""
    
    @staticmethod
    def prepare_nodes_and_relationships(patients_df: pd.DataFrame, 
                                      treatments_df: Optional[pd.DataFrame] = None,
                                      labs_df: Optional[pd.DataFrame] = None,
                                      procedures_df: Optional[pd.DataFrame] = None,
                                      notes_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Convert relational data to Neo4j-ready nodes and relationships"""
        nodes = []
        relationships = []
        
        for _, patient in patients_df.iterrows():
            nodes.append({
                'id': f"patient_{patient['subject_id']}",
                'type': 'Patient',
                'properties': {
                    'subject_id': patient['subject_id'],
                    'gender': patient['gender'],
                    'age': patient['age'],
                    'condition_type': patient['condition_type'],
                    'admission_type': patient.get('admission_type', 'Unknown')
                }
            })
            
            condition_id = f"condition_{patient['condition_type'].replace(' ', '_').lower()}"
            if not any(n['id'] == condition_id for n in nodes):
                nodes.append({
                    'id': condition_id,
                    'type': 'Condition',
                    'properties': {
                        'name': patient['condition_type'],
                        'icd_code_pattern': patient.get('icd_code', 'E11')[:3]
                    }
                })
            
            relationships.append({
                'from': f"patient_{patient['subject_id']}",
                'to': condition_id,
                'type': 'HAS_CONDITION',
                'properties': {
                    'icd_code': patient.get('icd_code', 'E11'),
                    'icd_version': patient.get('icd_version', 10)
                }
            })
        
        if treatments_df is not None and not treatments_df.empty:
            for _, treatment in treatments_df.iterrows():
                treatment_id = f"treatment_{treatment.get('drug', 'unknown').replace(' ', '_').lower()}"
                if not any(n['id'] == treatment_id for n in nodes):
                    nodes.append({
                        'id': treatment_id,
                        'type': 'Treatment',
                        'properties': {
                            'drug': treatment.get('drug', 'Unknown'),
                            'dose': treatment.get('dose_val_rx', 'Unknown'),
                            'starttime': str(treatment.get('starttime', ''))
                        }
                    })
                
                relationships.append({
                    'from': f"patient_{treatment['subject_id']}",
                    'to': treatment_id,
                    'type': 'RECEIVED_TREATMENT',
                    'properties': {
                        'hadm_id': treatment.get('hadm_id')
                    }
                })
        
        if labs_df is not None and not labs_df.empty:
            unique_labs = labs_df['lab_test'].unique()
            for lab_label in unique_labs:
                lab_id = f"lab_{lab_label.replace(' ', '_').lower()}"
                nodes.append({
                    'id': lab_id,
                    'type': 'LabTest',
                    'properties': {
                        'label': lab_label,
                        'category': 'diabetes_related'
                    }
                })
            
            for _, lab in labs_df.iterrows():
                lab_id = f"lab_{lab['lab_test'].replace(' ', '_').lower()}"
                relationships.append({
                    'from': f"patient_{lab['subject_id']}",
                    'to': lab_id,
                    'type': 'HAD_LAB_TEST',
                    'properties': {
                        'value': lab.get('valuenum'),
                        'unit': lab.get('valueuom', ''),
                        'charttime': str(lab.get('charttime', ''))
                    }
                })
        
        graph_structure = {
            'nodes': nodes,
            'relationships': relationships,
            'metadata': {
                'total_nodes': len(nodes),
                'total_relationships': len(relationships),
                'node_types': list(set(n['type'] for n in nodes)),
                'relationship_types': list(set(r['type'] for r in relationships)),
                'density': len(relationships) / len(nodes) if nodes else 0
            }
        }
        
        logger.info(f"🕸️ Neo4j structure prepared: {len(nodes)} nodes, {len(relationships)} relationships")
        logger.info(f"📊 Node types: {graph_structure['metadata']['node_types']}")
        logger.info(f"🔗 Relationship types: {graph_structure['metadata']['relationship_types']}")
        
        return graph_structure
