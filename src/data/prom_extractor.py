
import logging
import pandas as pd
from typing import Dict, Any, List, Tuple
from multiprocessing import Pool, cpu_count
import spacy
from datetime import datetime

logger = logging.getLogger(__name__)

class PROMExtractor:
    def __init__(self, config: Dict):
        self.config = config
        self.min_confidence = config["data_processing"]["prom_confidence_min"]
        self.nlp = None
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except Exception as e:
            logger.warning(f"⚠️ Failed to load spaCy model: {e}. Using fallback rules.")

    def _extract_prom_single(self, args: Tuple[Any, pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> Dict:
        patient, patient_notes, patient_labs, patient_treatments = args
        subject_id = patient['subject_id']
        
        base_score = 70.0
        
        age_adjustment = 0
        if 'age' in patient and pd.notna(patient['age']):
            age = float(patient['age'])
            if age < 40:
                age_adjustment = 10
            elif age >= 60:
                age_adjustment = -15
        
        condition_adjustment = 0
        condition_type = patient.get('condition_type', 'Unknown')
        if 'Type 1' in condition_type:
            condition_adjustment = -10
        elif 'Type 2' in condition_type:
            condition_adjustment = -5
        elif 'COPD' in condition_type:
            condition_adjustment = -12
        
        lab_adjustment = 0
        if not patient_labs.empty:
            glucose_labs = patient_labs[patient_labs['lab_test'].str.contains('Glucose|glucose', case=False, na=False)]
            if not glucose_labs.empty:
                avg_glucose = glucose_labs['valuenum'].mean()
                if avg_glucose > 180:
                    lab_adjustment -= 15
                elif avg_glucose > 140:
                    lab_adjustment -= 8
                elif avg_glucose < 80:
                    lab_adjustment -= 5
            
            hba1c_labs = patient_labs[patient_labs['lab_test'].str.contains('Hemoglobin A1c|HbA1c', case=False, na=False)]
            if not hba1c_labs.empty:
                avg_hba1c = hba1c_labs['valuenum'].mean()
                if avg_hba1c > 9:
                    lab_adjustment -= 20
                elif avg_hba1c > 7:
                    lab_adjustment -= 10
        
        treatment_adjustment = 0
        if not patient_treatments.empty:
            unique_treatments = patient_treatments['treatment_category'].nunique()
            if unique_treatments > 5:
                treatment_adjustment = -8
            elif unique_treatments > 3:
                treatment_adjustment = -4
        
        notes_adjustment = 0
        combined_notes = ''  # Initialize to avoid undefined error
        if not patient_notes.empty and 'text' in patient_notes.columns:
            combined_notes = ' '.join(
                note for note in patient_notes['text'].astype(str).str.lower()
                if any(kw in note for kw in ['diabetes', 'glucose', 'insulin', 'pain', 'mobility'])
            )
        
        if combined_notes:
            if self.nlp:
                doc = self.nlp(combined_notes)
                positive_count = sum(1 for token in doc if token.text in ['stable', 'improved', 'well-controlled', 'good'])
                negative_count = sum(1 for token in doc if token.text in ['complications', 'poorly controlled', 'worsened', 'pain'])
            else:
                positive_count = sum(combined_notes.count(kw) for kw in ['stable', 'improved', 'well-controlled', 'good'])
                negative_count = sum(combined_notes.count(kw) for kw in ['complications', 'poorly controlled', 'worsened', 'pain'])
                
            if positive_count > negative_count:
                notes_adjustment += min(10, positive_count * 2)
            elif negative_count > positive_count:
                notes_adjustment -= min(15, negative_count * 2)
        
        final_score = max(15, min(95, base_score + age_adjustment + condition_adjustment + 
                                 lab_adjustment + treatment_adjustment + notes_adjustment))
        
        confidence = 0.6
        if not patient_labs.empty:
            confidence += 0.15
        if not patient_treatments.empty:
            confidence += 0.1
        if combined_notes:
            confidence += 0.1
        if 'age' in patient and pd.notna(patient['age']):
            confidence += 0.05
        confidence = min(0.95, confidence)
        
        return {
            'subject_id': subject_id,
            'quality_of_life_score': round(final_score, 1),
            'pain_level': round(max(1, min(10, 8 - (final_score - 30) / 10)), 1),
            'mobility_score': round(max(1, min(5, 1 + (final_score - 20) / 20)), 1),
            'confidence': round(confidence, 2),
            'data_completeness': {
                'has_labs': not patient_labs.empty,
                'has_treatments': not patient_treatments.empty,
                'has_notes': bool(combined_notes)
            }
        }

    def extract_proms(self, patients: pd.DataFrame, notes: pd.DataFrame, labs: pd.DataFrame, treatments: pd.DataFrame) -> List[Dict]:
        """Extract PROMs from clinical notes with lab/treatment adjustments"""
        logger.info(f"📊 Extracting PROMs for {len(patients)} patients...")
        prom_scores = []
        
        try:
            if patients.empty or notes.empty:
                logger.warning("⚠️ No patients or notes for PROM extraction")
                return prom_scores
            
            notes_grouped = notes.groupby('subject_id')
            labs_grouped = labs.groupby('subject_id')
            treatments_grouped = treatments.groupby('subject_id')
            
            batch_size = 1000
            for i in range(0, len(patients), batch_size):
                batch = patients[i:i + batch_size]
                args = [
                    (patient, 
                     notes_grouped.get_group(patient['subject_id']) if patient['subject_id'] in notes_grouped.groups else pd.DataFrame(),
                     labs_grouped.get_group(patient['subject_id']) if patient['subject_id'] in labs_grouped.groups else pd.DataFrame(),
                     treatments_grouped.get_group(patient['subject_id']) if patient['subject_id'] in treatments_grouped.groups else pd.DataFrame())
                    for _, patient in batch.iterrows()
                ]
                
                num_processes = min(cpu_count(), max(1, len(args) // 100 + 1))
                with Pool(num_processes) as p:
                    batch_scores = p.map(self._extract_prom_single, args)
                prom_scores.extend(batch_scores)
                
                logger.info(f"✅ Processed batch {i//batch_size + 1}: {len(batch_scores)} PROMs")
            
            logger.info(f"✅ PROM extraction complete: {len(prom_scores)} scores generated")
            return prom_scores
        except Exception as e:
            logger.error(f"❌ PROM extraction failed: {e}")
            return []
