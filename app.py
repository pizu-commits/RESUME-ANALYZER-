# -*- coding: utf-8 -*-
"""
Resume Analyzer Pro - Complete Version with Dynamic Learning
"""

import streamlit as st
def colored_header(label, description=None, color=None):
    st.markdown(f"""
    <h2 style='color:{color or "#4f8bf9"};'>{label}</h2>
    {f"<p>{description}</p>" if description else ""}
    """, unsafe_allow_html=True)
import joblib
import time
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import plotly.express as px
import re
import os
import pandas as pd
from io import BytesIO
import datetime

# Set page config
st.set_page_config(
    page_title="Resume Analyzer Pro",
    layout="wide",
    page_icon="📝",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# Resume Analysis Tool v4.0"
    }
)


# --- CORE FUNCTIONS ---
def train_models():
    """Enhanced training with expanded dataset and better feature extraction"""
    training_data = [
        # Technical resumes (25 examples)
        ("python java sql machine learning data science algorithms pandas numpy", 1),
        ("javascript react node web development frontend backend html css", 1),
        ("data analysis pandas numpy sklearn tensorflow pytorch visualization", 1),
        ("devops aws docker kubernetes cloud infrastructure ci/cd terraform", 1),
        ("c++ embedded systems firmware iot robotics arm cortex", 1),
        ("networking cybersecurity firewall encryption vulnerability pentest", 1),
        ("mobile android ios swift kotlin react-native flutter", 1),
        ("database sql nosql mongodb postgresql mysql oracle", 1),
        ("ai ml nlp computer-vision deep-learning neural-networks", 1),
        ("testing selenium pytest unittest test-automation qa", 1),
        ("eggplant test-automation framework scripting test-execution", 1),
        ("embedded c arm cortex microcontroller rtos device-drivers", 1),
        ("java spring hibernate microservices rest api maven", 1),
        ("python django flask fastapi web-development backend", 1),
        ("data-engineering etl pipeline airflow spark hadoop", 1),
        ("frontend react angular vue typescript webpack redux", 1),
        ("cloud aws azure gcp serverless lambda containers", 1),
        ("linux bash shell-scripting system-admin devops", 1),
        ("c# .net core entity-framework asp.net mvc", 1),
        ("ruby rails rspec postgresql redis sidekiq", 1),
        ("golang grpc microservices docker kubernetes", 1),
        ("rust systems-programming blockchain wasm embedded", 1),
        ("scala spark functional-programming akka big-data", 1),
        ("php laravel wordpress mysql javascript jquery", 1),
        ("swift ios xcode cocoa-touch core-data uikit", 1),

        # Non-technical resumes (25 examples)
        ("accounting finance taxes auditing bookkeeping payroll quickbooks", 0),
        ("human resources recruiting hiring onboarding interviews talent", 0),
        ("marketing sales advertising social media seo sem analytics", 0),
        ("customer service support helpdesk troubleshooting client", 0),
        ("administration office management scheduling coordination", 0),
        ("graphic design photoshop illustrator branding ui/ux figma", 0),
        ("content writing editing copywriting blogging seo creative", 0),
        ("operations supply-chain logistics inventory procurement", 0),
        ("healthcare medical nursing patient care hospital clinic", 0),
        ("education teaching curriculum lesson-planning classroom", 0),
        ("legal law paralegal contracts litigation corporate", 0),
        ("real-estate property sales leasing brokerage housing", 0),
        ("retail store management merchandising customer-service", 0),
        ("hospitality hotel restaurant management food beverage", 0),
        ("fashion design merchandising styling retail textiles", 0),
        ("architecture design drafting construction cad bim", 0),
        ("journalism news reporting editing broadcasting media", 0),
        ("public-relations communications media-relations press", 0),
        ("event planning coordination weddings conferences meetings", 0),
        ("fitness training nutrition wellness personal-trainer", 0),
        ("interior-design space-planning residential commercial", 0),
        ("film video production editing cinematography script", 0),
        ("music performance composition audio-production sound", 0),
        ("psychology counseling therapy mental-health social-work", 0),
        ("finance banking investment wealth-management advisor", 0)
    ]

    X_train = [text for text, label in training_data]
    y_train = [label for text, label in training_data]

    # Create vectorizer separately for easier access later
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.8
    )

    # Text processing pipeline
    text_pipeline = make_pipeline(
        vectorizer,
        StandardScaler(with_mean=False)
    )

    X_train_transformed = text_pipeline.fit_transform(X_train)

    # Logistic Regression for classification (technical vs non-technical)
    classifier = LogisticRegression(
        random_state=42,
        class_weight='balanced',
        max_iter=1000,
        C=0.1,
        solver='saga'
    )
    classifier.fit(X_train_transformed, y_train)

    # KMeans for clustering resumes into 5 categories
    clusterer = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=20,
        max_iter=300
    )
    clusterer.fit(X_train_transformed)

    return text_pipeline, classifier, clusterer, vectorizer


def extract_text_from_file(uploaded_file):
    """Robust text extraction with multiple fallback methods"""
    text = ""
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)  # Reset file pointer

    # PDF Extraction
    if uploaded_file.type == "application/pdf":
        try:
            import PyPDF2
            try:
                reader = PyPDF2.PdfReader(BytesIO(file_bytes))
                text = "\n".join([page.extract_text() or "" for page in reader.pages])
                if text.strip(): return text
            except PyPDF2.errors.PdfReadError:
                st.warning("PyPDF2 couldn't read this PDF (might be scanned/image-based)")
        except ImportError:
            st.warning("PyPDF2 not available for PDF extraction")

        try:
            import pdfplumber
            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                text = "\n".join([page.extract_text() or "" for page in pdf.pages])
            if text.strip(): return text
        except ImportError:
            st.warning("pdfplumber not available for PDF extraction")
        except Exception as e:
            st.warning(f"pdfplumber extraction failed: {str(e)}")

    # DOCX Extraction
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        try:
            import docx2txt
            text = docx2txt.process(BytesIO(file_bytes))
            if text.strip(): return text
        except ImportError:
            st.warning("docx2txt not available for DOCX extraction")
        except Exception as e:
            st.warning(f"docx2txt extraction failed: {str(e)}")

    # Plain Text
    elif uploaded_file.type == "text/plain":
        try:
            text = file_bytes.decode("utf-8", errors="replace")
            if text.strip(): return text
        except UnicodeDecodeError:
            try:
                text = file_bytes.decode("latin-1", errors="replace")
                if text.strip(): return text
            except Exception as e:
                st.warning(f"Text decoding failed: {str(e)}")

    # Final fallback
    try:
        text = file_bytes.decode("utf-8", errors="replace")
        if text.strip(): return text
    except:
        pass

    if not text.strip():
        st.error("""
        Could not extract any readable text. Possible reasons:
        1. Scanned/image-based PDF
        2. Password protected
        3. Unsupported format
        4. Corrupted file

        Try converting to DOCX or plain text.
        """)
        return None

    return text


def clean_resume_text(text):
    """Enhanced cleaning that preserves skill context"""
    text = text.lower()
    text = re.sub(r'[^\w\s-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_experience(text):
    """Comprehensive experience detection"""
    patterns = [
        r'(\d+)\s*(years?|yrs?)\s*(experience|exp)',
        r'experience\s*:\s*(\d+)\s*(years?|yrs?)',
        r'(\d+)\+?\s*(years?|yrs?)\s*(in|of)',
        r'(\d+)\s*\+\s*years',
        r'(\d+)\s*-\s*(\d+)\s*years?',
        r'(\d+)\s*(years?|yrs?)\s*professional',
        r'(\d+)\s*(years?|yrs?)\s*industry',
        r'over\s*(\d+)\s*years',
        r'(\d+)\s*(years?|yrs?)\s*hands?-on'
    ]

    max_years = 0
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            numeric_groups = [int(g) for g in match.groups() if g and g.isdigit()]
            if numeric_groups:
                current_max = max(numeric_groups)
                if current_max > max_years:
                    max_years = current_max

    # Date range detection
    date_range_matches = re.finditer(
        r'(20\d{2}|19\d{2})\s*[-–]\s*(20\d{2}|present|current)',
        text,
        re.IGNORECASE
    )
    for match in date_range_matches:
        try:
            start_year = int(match.group(1))
            end_year = match.group(2)
            if end_year.lower() in ['present', 'current']:
                end_year = datetime.datetime.now().year
            else:
                end_year = int(end_year)
            years = end_year - start_year
            if years > max_years:
                max_years = years
        except:
            continue

    return max_years


def evaluate_resume(text, models):
    """Dynamic evaluation with comprehensive feature analysis"""
    text_pipeline, classifier, clusterer, vectorizer = models
    cleaned_text = clean_resume_text(text)

    # Transform using full pipeline
    text_features = text_pipeline.transform([cleaned_text])

    # Get prediction probabilities from Logistic Regression classifier
    tech_prob = classifier.predict_proba(text_features)[0][1]

    # Get top predictive terms
    feature_names = vectorizer.get_feature_names_out()
    feature_scores = text_features.toarray()[0]
    top_terms = sorted(
        [(feature_names[i], round(feature_scores[i], 3))
         for i in text_features.nonzero()[1]],
        key=lambda x: x[1],
        reverse=True
    )[:15]

    # Get cluster from KMeans
    cluster = clusterer.predict(text_features)[0]
    cluster_labels = {
        0: ("Core Technical", "#4CAF50"),
        1: ("General Technical", "#8BC34A"),
        2: ("Technical Support", "#CDDC39"),
        3: ("Semi-Technical", "#FFC107"),
        4: ("Non-Technical", "#FF9800")
    }
    cluster_name, cluster_color = cluster_labels.get(cluster, ("Unknown", "#9E9E9E"))

    experience_years = extract_experience(text)
    project_count = len(re.findall(
        r'project\s*:|projects?\s*\(|developed\s*a\s|built\s*a\s|implemented\s*a\s',
        cleaned_text, re.IGNORECASE
    ))
    education = bool(re.search(
        r'education|degree|university|college|b\.?s\.?|b\.?a\.?|m\.?s\.?|ph\.?d\.?',
        cleaned_text, re.IGNORECASE
    ))

    # Calculate scores
    tech_strength = min(1.0, tech_prob * 1.2)
    exp_strength = min(1.0, experience_years / 10)
    base_score = (0.5 * tech_strength +
                  0.3 * exp_strength +
                  0.1 * (project_count / 5) +
                  0.1 * (1 if education else 0.3))
    final_score = min(0.99, max(0.1, base_score))

    # Generate insights
    insights = []
    if tech_strength < 0.5:
        insights.append("Increase technical keywords relevant to your target role")
    if experience_years < 2:
        insights.append("Highlight academic/personal projects as professional experience")
    if project_count < 2:
        insights.append("Include more projects with technologies used and outcomes")
    if not education:
        insights.append("Add education section if applicable")
    if len(top_terms) < 5:
        insights.append("Expand your skills section with relevant technologies")

    return {
        "final_score": final_score,
        "cluster": cluster_name,
        "cluster_color": cluster_color,
        "top_terms": top_terms,
        "experience_years": experience_years,
        "tech_strength": tech_strength,
        "project_count": project_count,
        "education": education,
        "insights": insights
    }


# --- UI COMPONENTS ---
def set_custom_style():
    """Enhanced CSS styles with modern look"""
    st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 10px 24px;
        }
        .metric-card {
            background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
            color: white;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 25px;
        }
        .skill-badge {
            display: inline-block;
            background-color: #e3f2fd;
            color: #1565c0;
            padding: 5px 10px;
            border-radius: 16px;
            margin: 4px;
        }
    </style>
    """, unsafe_allow_html=True)


def display_results(result):
    """Enhanced results display"""
    st.success("✅ Analysis Complete!", icon="✅")

    # Score Card
    with st.container():
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin-top:0; color:white;">Resume Analysis Report</h3>
            <h1 style="text-align:center; margin:10px 0; font-size:3em;">{result['final_score']:.0%}</h1>
            <div style="display:flex; justify-content:center;">
                <span style="background-color:{result['cluster_color']}; 
                color:white; padding:5px 15px; border-radius:20px;">
                {result['cluster']} Profile
                </span>
            </div>
            <div style="margin-top:15px; display:flex; justify-content:space-between;">
                <div>🔧 Tech Strength: {result['tech_strength']:.0%}</div>
                <div>⏳ Experience: {result['experience_years']} yrs</div>
                <div>🏆 Projects: {result['project_count']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.subheader("🔍 Top Influential Terms")
        terms_html = "".join(
            f'<span style="display:inline-block; background:#f5f5f5; padding:4px 8px; border-radius:4px; margin:2px;">{term} (weight: {score})</span>'
            for term, score in result['top_terms']
        )
        st.markdown(f'<div style="line-height:2.5;">{terms_html}</div>',
                    unsafe_allow_html=True)

        if result['insights']:
            st.subheader("💡 Improvement Suggestions")
            for insight in result['insights']:
                st.markdown(f"""
                <div style="background-color:#e8f5e9; border-left:4px solid #4CAF50; padding:12px 16px; border-radius:4px; margin-bottom:10px;">
                    ✨ {insight}
                </div>
                """, unsafe_allow_html=True)

    with col2:
        st.subheader("📊 Key Metrics")
        with st.expander("Experience Analysis", expanded=True):
            st.metric("Total Years", f"{result['experience_years']} years")
        with st.expander("Project Portfolio"):
            st.metric("Projects Detected", result['project_count'])
        with st.expander("Education"):
            st.metric("Education Section",
                      "Found" if result['education'] else "Not Found")

        st.subheader("🧠 Technical Profile")
        st.metric("Technical Strength", f"{result['tech_strength']:.0%}")
        st.progress(result['tech_strength'])


# --- MAIN APP ---
def main():
    set_custom_style()

    colored_header(
        label="📝 Resume Analyzer Pro",
        description="AI-powered resume analysis with dynamic learning",
        color_name="blue-70"
    )

    with st.expander("🔍 How It Works", expanded=True):
        st.markdown("""
        **Machine Learning Models Used:**
        - 🤖 **Logistic Regression**: Classifies resumes as technical/non-technical
        - 🌀 **KMeans Clustering**: Groups resumes into 5 profile types
        - 🔠 **TF-IDF Vectorizer**: Extracts key terms and their importance

        **Tips for best results:**
        1. Use text-based formats (not scanned PDFs)
        2. Include specific technologies
        3. Quantify experience and achievements
        """)

    # Model Loading
    @st.cache_resource(show_spinner="Loading analysis models...")
    def load_models():
        try:
            models = (
                joblib.load('models/text_pipeline.pkl'),
                joblib.load('models/classifier.pkl'),
                joblib.load('models/clusterer.pkl'),
                joblib.load('models/vectorizer.pkl')
            )
            st.sidebar.success("✔ Models loaded from cache")
            return models
        except:
            st.sidebar.warning("⚠ Training models (first run only)...")
            with st.spinner('Training models... This may take a minute'):
                text_pipeline, classifier, clusterer, vectorizer = train_models()
                os.makedirs('models', exist_ok=True)
                joblib.dump(text_pipeline, 'models/text_pipeline.pkl')
                joblib.dump(classifier, 'models/classifier.pkl')
                joblib.dump(clusterer, 'models/clusterer.pkl')
                joblib.dump(vectorizer, 'models/vectorizer.pkl')
                models = (text_pipeline, classifier, clusterer, vectorizer)
            st.sidebar.success("✔ Models trained and cached")
            return models

    models = load_models()

    # Sample Resumes
    with st.expander("📋 Sample Resumes (Try These)"):
        good_resume = """
        John Doe
        Senior Data Scientist
        5+ years experience in machine learning and data analysis

        TECHNICAL SKILLS:
        - Python (Pandas, NumPy, Scikit-learn)
        - Machine Learning (TensorFlow, PyTorch)
        - SQL and NoSQL databases
        - Data Visualization (Matplotlib, Seaborn)

        EXPERIENCE:
        Data Scientist at TechCorp (2019-Present)
        - Developed predictive models improving revenue by 15%
        - Built ETL pipelines processing 1TB+ daily data

        Data Analyst at AnalyticsCo (2017-2019)
        - Created dashboards reducing reporting time by 30%

        EDUCATION:
        MS in Computer Science - State University (2016)
        """

        bad_resume = """
        Jane Smith
        Office Assistant

        WORK HISTORY:
        - Answered phones
        - Filed documents
        - Made coffee
        - Scheduled meetings

        SKILLS:
        - Microsoft Word
        - Email
        - Phone etiquette
        """

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Good Technical Resume**")
            st.code(good_resume)
        with col2:
            st.write("**Bad Non-Technical Resume**")
            st.code(bad_resume)

    # File Upload
    st.subheader("📤 Upload Your Resume")
    uploaded_file = st.file_uploader(
        "Choose a file (PDF, DOCX, or TXT)",
        type=['pdf', 'docx', 'txt'],
        label_visibility="collapsed"
    )

    if uploaded_file:
        try:
            if uploaded_file.size > 10 * 1024 * 1024:
                st.error("File too large (max 10MB).")
                st.stop()

            with st.spinner('⏳ Analyzing your resume...'):
                text = extract_text_from_file(uploaded_file)
                if text is None:
                    st.stop()

                result = evaluate_resume(text, models)
                display_results(result)

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.stop()

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#6c757d;">
        <p>Resume Analyzer Pro v4.0 | Uses Logistic Regression and KMeans</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
