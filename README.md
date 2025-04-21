# Medical Data Analysis with Generative AI

This project demonstrates the application of generative AI technologies to healthcare data analysis, focusing on medical information summarization and tabular-to-text conversion using various prompt engineering techniques.

## Overview

The system processes synthetic medical data in XML format, extracts relevant patient information, and applies different AI-powered analysis methods to generate insights and natural language descriptions.

## Features

- **Medical Data Processing**:
  - Parse XML clinical documents
  - Extract patient demographics, vital signs, medications, and mental health assessments
  - Convert structured data to pandas DataFrame for analysis

- **AI-Powered Analysis**:
  - Summarize patient medical information using multiple prompt engineering methods
  - Convert tabular patient data to natural language descriptions
  - Evaluate the quality of generated medical summaries

- **Prompt Engineering Techniques**:
  - Basic prompting
  - Zero-shot learning
  - Few-shot learning with examples
  - Chain of Thought (CoT) reasoning
  - Tree of Thoughts (ToT) reasoning

## Requirements

- Python 3.7+
- OpenAI API key
- Required Python packages:
  ```
  pandas
  numpy
  matplotlib
  openai
  python-dotenv
  tqdm
  ```

## Setup

1. Clone the repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. Ensure your dataset is in the correct location (`AI_In_healthcare/Assignment5/dataset`)

## Usage

Run the main script:
```
python AI_In_healthcare/Assignment5/ass5.py
```

The script will:
1. Process a sample of patient XML files
2. Display a preview of the extracted data
3. Demonstrate different summarization methods on a sample patient
4. Evaluate the quality of a generated summary
5. Show different tabular-to-text conversion methods

## Example Output

The program will output:
- DataFrame preview of processed patient data
- Medical summaries generated using different prompt engineering techniques
- Evaluation of summary quality
- Natural language descriptions converted from tabular data

## Project Structure

- `ass5.py`: Main script containing all functionality
- `dataset/`: Directory containing XML patient data files

## Classes

### MedicalDataProcessor

Handles the extraction and processing of medical data from XML files.

### MedicalAIAnalyzer

Implements various AI-powered analysis methods using the OpenAI API.

## Notes

- This project uses synthetic medical data for demonstration purposes
- The OpenAI API calls may incur costs depending on your usage
- Processing large datasets may take significant time and API resources

## Future Improvements

- Add support for more data formats
- Implement additional prompt engineering techniques
- Create visualizations of patient data
- Add batch processing capabilities for large datasets
- Implement more sophisticated evaluation metrics
