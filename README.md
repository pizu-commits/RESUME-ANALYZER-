# RESUME-ANALYZER-#
AI-powered resume analysis tool using machine learning to evaluate and score resumes.

## Features
- Classifies technical vs non-technical resumes
- Clusters resumes into 5 profile types
- Extracts key skills and terms
- Estimates years of experience
- Provides improvement suggestions

## Installation
bash
git clone https://github.com/yourusername/resume-analyzer-pro.git
cd resume-analyzer-pro
pip install -r requirements.txt
Usage
###################################bash##########################
streamlit run resume_analyzer.py
Upload PDF/DOCX/TXT resumes to get:
Overall score (0-100%)
Technical strength rating
Experience analysis
Top influential terms
Personalized suggestions

############################Dependencies################################
Python 3.8+
streamlit
scikit-learn
pandas
PyPDF2/pdfplumber (for PDF parsing)
docx2txt (for DOCX parsing)

######################Models Used####################################
TF-IDF Vectorizer (text feature extraction)
Logistic Regression (classification)
K-Means Clustering (profile grouping)

########################File Structure############################
text
/models/       # Saved ML models
resume_analyzer.py  # Main application
requirements.txt    # Dependencies


#######################LimitatioNs###################################
Works best with text-based resumes
First run requires model training
Accuracy depends on input quality
