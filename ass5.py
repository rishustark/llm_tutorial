import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import re
import openai
import json
from dotenv import load_dotenv
import time
from tqdm import tqdm
import logging

# Load environment variables (for API keys)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

class MedicalDataProcessor:
    def __init__(self, dataset_path):
        """Initialize the processor with the path to the dataset."""
        self.dataset_path = dataset_path
        self.files = self._get_xml_files()
        self.patient_data = {}
        
    def _get_xml_files(self):
        """Get all XML files in the dataset directory."""
        return [f for f in os.listdir(self.dataset_path) if f.endswith('.xml')]
    
    def parse_xml_file(self, filename):
        """Parse an XML file and extract relevant medical information."""
        try:
            tree = ET.parse(os.path.join(self.dataset_path, filename))
            root = tree.getroot()
            
            # Define namespace for XML parsing
            ns = {'cda': 'urn:hl7-org:v3'}
            
            # Extract patient demographics
            patient_info = {}
            
            # Get patient name
            name_element = root.find('.//cda:patient/cda:name', ns)
            if name_element is not None:
                given = name_element.find('cda:given', ns)
                family = name_element.find('cda:family', ns)
                if given is not None and family is not None:
                    patient_info['name'] = f"{given.text} {family.text}"
            
            # Get patient gender
            gender = root.find('.//cda:administrativeGenderCode', ns)
            if gender is not None:
                gender_code = gender.get('code')
                patient_info['gender'] = 'Female' if gender_code == 'F' else 'Male'
            
            # Get patient birth date
            birth_time = root.find('.//cda:birthTime', ns)
            if birth_time is not None:
                birth_date = birth_time.get('value')
                if birth_date:
                    # Format: YYYYMMDDHHMMSS
                    patient_info['birth_date'] = f"{birth_date[:4]}-{birth_date[4:6]}-{birth_date[6:8]}"
            
            # Extract vital signs
            vitals = []
            for observation in root.findall('.//cda:observation', ns):
                code_element = observation.find('cda:code', ns)
                if code_element is not None:
                    code = code_element.get('code')
                    display_name = code_element.get('displayName')
                    
                    value_element = observation.find('cda:value', ns)
                    if value_element is not None:
                        value = value_element.get('value')
                        unit = value_element.get('unit')
                        
                        effective_time = observation.find('cda:effectiveTime', ns)
                        date = None
                        if effective_time is not None:
                            date_value = effective_time.get('value')
                            if date_value:
                                date = f"{date_value[:4]}-{date_value[4:6]}-{date_value[6:8]}"
                        
                        if display_name and value:
                            vitals.append({
                                'type': display_name,
                                'value': value,
                                'unit': unit,
                                'date': date
                            })
            
            patient_info['vitals'] = vitals
            
            # Extract medications
            medications = []
            for medication in root.findall('.//cda:substanceAdministration', ns):
                consumable = medication.find('.//cda:manufacturedMaterial/cda:name', ns)
                if consumable is not None and consumable.text:
                    medications.append(consumable.text)
            
            patient_info['medications'] = medications
            
            # Extract mental health assessments
            mental_health = []
            for assessment in root.findall('.//cda:observation', ns):
                code_element = assessment.find('cda:code', ns)
                if code_element is not None:
                    display_name = code_element.get('displayName')
                    if display_name and any(term in display_name for term in ['PHQ', 'GAD', 'AUDIT', 'DAST']):
                        value_element = assessment.find('cda:value', ns)
                        if value_element is not None:
                            score = value_element.get('value')
                            if score:
                                mental_health.append({
                                    'assessment': display_name,
                                    'score': score
                                })
            
            patient_info['mental_health'] = mental_health
            
            return patient_info
            
        except Exception as e:
            logger.error(f"Error parsing {filename}: {str(e)}")
            return None
    
    def process_all_files(self, limit=None):
        """Process all XML files and store the extracted data."""
        files_to_process = self.files[:limit] if limit else self.files
        
        for filename in tqdm(files_to_process, desc="Processing files"):
            patient_data = self.parse_xml_file(filename)
            if patient_data:
                self.patient_data[filename] = patient_data
        
        logger.info(f"Processed {len(self.patient_data)} files successfully")
        return self.patient_data
    
    def convert_to_dataframe(self):
        """Convert the extracted patient data to a pandas DataFrame."""
        # Create a list to hold flattened patient records
        records = []
        
        for filename, data in self.patient_data.items():
            record = {
                'filename': filename,
                'name': data.get('name', ''),
                'gender': data.get('gender', ''),
                'birth_date': data.get('birth_date', ''),
                'medication_count': len(data.get('medications', [])),
                'medications': ', '.join(data.get('medications', []))
            }
            
            # Add the latest vital signs
            vitals = data.get('vitals', [])
            if vitals:
                # Sort by date (most recent first)
                vitals_sorted = sorted(vitals, key=lambda x: x.get('date', ''), reverse=True)
                
                # Add the most recent vitals
                for vital in vitals_sorted:
                    vital_type = vital.get('type', '').replace(' ', '_').lower()
                    if vital_type and vital_type not in record:
                        record[vital_type] = f"{vital.get('value', '')} {vital.get('unit', '')}"
            
            # Add mental health assessments
            for assessment in data.get('mental_health', []):
                assessment_name = assessment.get('assessment', '').replace(' ', '_').lower()
                if assessment_name:
                    record[assessment_name] = assessment.get('score', '')
            
            records.append(record)
        
        return pd.DataFrame(records)

class MedicalAIAnalyzer:
    def __init__(self, model="gpt-3.5-turbo"):
        """Initialize the AI analyzer with the specified model."""
        self.model = model
    
    def _call_openai_api(self, prompt, max_tokens=1000):
        """Call the OpenAI API with the given prompt."""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            return f"Error: {str(e)}"
    
    def summarize_patient_data(self, patient_data):
        """Summarize patient data using a basic prompt."""
        prompt = f"""
        Please provide a concise medical summary for the following patient:
        
        Name: {patient_data.get('name', 'Unknown')}
        Gender: {patient_data.get('gender', 'Unknown')}
        Birth Date: {patient_data.get('birth_date', 'Unknown')}
        
        Medications:
        {', '.join(patient_data.get('medications', ['None']))}
        
        Vital Signs:
        {json.dumps(patient_data.get('vitals', []), indent=2)}
        
        Mental Health Assessments:
        {json.dumps(patient_data.get('mental_health', []), indent=2)}
        
        Please include key health insights and potential concerns based on this data.
        """
        
        return self._call_openai_api(prompt)
    
    def summarize_with_zero_shot(self, patient_data):
        """Summarize patient data using zero-shot learning."""
        prompt = f"""
        Create a comprehensive medical summary for the following patient data. Focus on identifying key health issues, medication effects, and mental health status.
        
        Patient Data:
        {json.dumps(patient_data, indent=2)}
        """
        
        return self._call_openai_api(prompt)
    
    def summarize_with_few_shot(self, patient_data):
        """Summarize patient data using few-shot learning with examples."""
        prompt = f"""
        I'll provide you with examples of good medical summaries, followed by new patient data. Please create a similar summary for the new patient.
        
        Example 1:
        Patient: John Doe, 45-year-old male
        Data: BMI 28.5, BP 140/90, Heart rate 88, PHQ-9 score 12, Medications: lisinopril, metformin
        Summary: John is a middle-aged male with indicators of hypertension (elevated blood pressure 140/90) and overweight status (BMI 28.5). His elevated heart rate (88 bpm) suggests possible cardiovascular stress. The PHQ-9 score of 12 indicates moderate depression requiring attention. Current medications (lisinopril for hypertension, metformin typically for diabetes) suggest management of cardiometabolic conditions. Recommend follow-up on depression symptoms, blood pressure monitoring, and lifestyle modifications for weight management.
        
        Example 2:
        Patient: Jane Smith, 62-year-old female
        Data: BMI 24.1, BP 125/75, Heart rate 72, GAD-7 score 8, Medications: atorvastatin, levothyroxine
        Summary: Jane is a 62-year-old female with well-controlled vital signs (normal BP 125/75, normal heart rate 72 bpm) and healthy BMI (24.1). Her GAD-7 score of 8 indicates mild to moderate anxiety that should be monitored. Current medications suggest management of cholesterol (atorvastatin) and thyroid function (levothyroxine). Recommend continued monitoring of anxiety symptoms and regular thyroid function tests.
        
        New Patient:
        {json.dumps(patient_data, indent=2)}
        
        Please provide a comprehensive medical summary for this patient:
        """
        
        return self._call_openai_api(prompt)
    
    def summarize_with_cot(self, patient_data):
        """Summarize patient data using Chain of Thought reasoning."""
        prompt = f"""
        I need to create a medical summary for a patient. I'll think through this step by step.
        
        Patient Data:
        {json.dumps(patient_data, indent=2)}
        
        Let me analyze this data step by step:
        1. First, I'll identify the basic demographics and what they tell us.
        2. Next, I'll analyze the vital signs and their clinical significance.
        3. Then, I'll review the medications and what conditions they typically treat.
        4. After that, I'll interpret any mental health assessments and their implications.
        5. Finally, I'll synthesize all this information into a coherent clinical picture.
        
        Now, based on this reasoning process, please provide a comprehensive medical summary:
        """
        
        return self._call_openai_api(prompt)
    
    def summarize_with_tot(self, patient_data):
        """Summarize patient data using Tree of Thoughts reasoning."""
        prompt = f"""
        I need to create a medical summary for a patient using a tree of thoughts approach, where I explore multiple reasoning paths before reaching a conclusion.
        
        Patient Data:
        {json.dumps(patient_data, indent=2)}
        
        Path 1 - Physical Health Focus:
        - What do the vital signs indicate about cardiovascular health?
        - Are there any concerning trends in weight, BMI, or blood pressure?
        - What do the medications suggest about existing physical health conditions?
        
        Path 2 - Mental Health Focus:
        - What do the mental health assessments reveal about psychological well-being?
        - Are there connections between physical symptoms and mental health scores?
        - How might medications impact mental health status?
        
        Path 3 - Integrated Analysis:
        - How do physical and mental health factors interact in this patient?
        - What are the most pressing health concerns considering all data points?
        - What preventive measures or interventions might benefit this patient most?
        
        Based on exploring these multiple reasoning paths, please provide a comprehensive medical summary:
        """
        
        return self._call_openai_api(prompt)
    
    def tabular_to_text_basic(self, df_row):
        """Convert tabular patient data to text using a basic prompt."""
        prompt = f"""
        Convert the following tabular patient data into a natural language description:
        
        {df_row.to_dict()}
        """
        
        return self._call_openai_api(prompt)
    
    def tabular_to_text_with_cot(self, df_row):
        """Convert tabular patient data to text using Chain of Thought reasoning."""
        prompt = f"""
        I need to convert tabular patient data into a natural language description. I'll think through this step by step.
        
        Patient Data:
        {df_row.to_dict()}
        
        Let me analyze this data step by step:
        1. First, I'll identify the key demographic information.
        2. Next, I'll analyze any vital sign measurements.
        3. Then, I'll review the medications and what they might indicate.
        4. After that, I'll interpret any assessment scores.
        5. Finally, I'll synthesize all this information into a coherent narrative.
        
        Based on this reasoning process, please provide a natural language description of this patient:
        """
        
        return self._call_openai_api(prompt)
    
    def tabular_to_text_with_tot(self, df_row):
        """Convert tabular patient data to text using Tree of Thoughts reasoning."""
        prompt = f"""
        I need to convert tabular patient data into a natural language description using a tree of thoughts approach.
        
        Patient Data:
        {df_row.to_dict()}
        
        Path 1 - Clinical Narrative:
        - How would a clinician describe this patient's status?
        - What are the key clinical findings and their significance?
        - What medical terminology would be appropriate to use?
        
        Path 2 - Patient-Friendly Explanation:
        - How would this information be explained to the patient?
        - What analogies or simplified explanations might help understanding?
        - How can medical jargon be translated into accessible language?
        
        Path 3 - Research Perspective:
        - How would this patient data be described in a research context?
        - What patterns or correlations are notable from a research standpoint?
        - How does this case compare to population norms or clinical guidelines?
        
        Based on exploring these multiple reasoning paths, please provide a comprehensive natural language description:
        """
        
        return self._call_openai_api(prompt)
    
    def evaluate_summary(self, patient_data, summary):
        """Evaluate the quality of a patient data summary."""
        prompt = f"""
        Please evaluate the quality of the following medical summary based on accuracy, completeness, clarity, and clinical relevance.
        
        Original Patient Data:
        {json.dumps(patient_data, indent=2)}
        
        Summary to Evaluate:
        {summary}
        
        Please provide a detailed assessment addressing:
        1. Accuracy: Does the summary correctly represent the patient data?
        2. Completeness: Does it include all relevant information?
        3. Clarity: Is the summary clear and well-organized?
        4. Clinical Relevance: Does it highlight clinically important aspects?
        5. Overall Rating: On a scale of 1-10, how would you rate this summary?
        
        Please provide specific examples to support your evaluation.
        """
        
        return self._call_openai_api(prompt)

def main():
    # Path to the dataset
    dataset_path = "AI_In_healthcare/Assignment5/dataset"
    
    # Initialize the processor
    processor = MedicalDataProcessor(dataset_path)
    
    # Process a limited number of files for demonstration
    patient_data = processor.process_all_files(limit=5)
    
    # Convert to DataFrame
    df = processor.convert_to_dataframe()
    print("\nDataFrame Preview:")
    print(df.head())
    
    # Initialize the AI analyzer
    analyzer = MedicalAIAnalyzer()
    
    # Select a sample patient for demonstration
    sample_patient_key = list(patient_data.keys())[0]
    sample_patient = patient_data[sample_patient_key]
    
    print(f"\n\nAnalyzing patient: {sample_patient.get('name', 'Unknown')}")
    
    # Demonstrate different summarization methods
    print("\n1. Basic Summarization:")
    basic_summary = analyzer.summarize_patient_data(sample_patient)
    print(basic_summary)
    
    print("\n2. Zero-Shot Learning Summarization:")
    zero_shot_summary = analyzer.summarize_with_zero_shot(sample_patient)
    print(zero_shot_summary)
    
    print("\n3. Few-Shot Learning Summarization:")
    few_shot_summary = analyzer.summarize_with_few_shot(sample_patient)
    print(few_shot_summary)
    
    print("\n4. Chain of Thought Summarization:")
    cot_summary = analyzer.summarize_with_cot(sample_patient)
    print(cot_summary)
    
    print("\n5. Tree of Thoughts Summarization:")
    tot_summary = analyzer.summarize_with_tot(sample_patient)
    print(tot_summary)
    
    # Evaluate a summary
    print("\nSummary Evaluation:")
    evaluation = analyzer.evaluate_summary(sample_patient, cot_summary)
    print(evaluation)
    
    # Demonstrate tabular to text conversion
    sample_row = df.iloc[0]
    
    print("\n1. Basic Tabular to Text Conversion:")
    basic_text = analyzer.tabular_to_text_basic(sample_row)
    print(basic_text)
    
    print("\n2. Chain of Thought Tabular to Text Conversion:")
    cot_text = analyzer.tabular_to_text_with_cot(sample_row)
    print(cot_text)
    
    print("\n3. Tree of Thoughts Tabular to Text Conversion:")
    tot_text = analyzer.tabular_to_text_with_tot(sample_row)
    print(tot_text)

if __name__ == "__main__":
    main()
