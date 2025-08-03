#!/usr/bin/env python3
"""
Fake Job Posting Classifier - Streamlit App
===========================================

A professional web application for detecting fake job postings using a trained Random Forest model.
Features: Clean UI/UX, real-time classification, confidence scores, and optional GPT-4 comparison.
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import time

# ------------------------------------------------------------------------------------
# 1. IMPORTS
# ------------------------------------------------------------------------------------
# All imports are at the top of the file

# ------------------------------------------------------------------------------------
# 2. PAGE CONFIG
# ------------------------------------------------------------------------------------
st.set_page_config(
    page_title="🔍 Fake Job Posting Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------------
# 3. LOAD MODEL
# ------------------------------------------------------------------------------------
@st.cache_resource
def load_model_and_vectorizer():
    """
    Load the trained model and vectorizer with caching for performance
    """
    try:
        # Load the optimized Random Forest model
        model = joblib.load('models/random_forest_cv.pkl')
        
        # Load the TF-IDF vectorizer
        vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
        
        return model, vectorizer
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

# Load the model and vectorizer
model, vectorizer = load_model_and_vectorizer()

# ------------------------------------------------------------------------------------
# 4. HELPER FUNCTIONS
# ------------------------------------------------------------------------------------
def clean_text(text):
    """
    Clean and preprocess the input text to match training data format
    """
    import re
    from html import unescape
    import string
    
    if not text or text.strip() == '':
        return ''
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Unescape HTML entities
    text = unescape(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def predict_job_posting(text, model, vectorizer):
    """
    Make prediction on the input text
    """
    try:
        # Clean the text
        cleaned_text = clean_text(text)
        
        if not cleaned_text:
            return None, None, "Empty or invalid input text"
        
        # Transform text using the vectorizer
        text_vectorized = vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction = model.predict(text_vectorized)[0]
        probability = model.predict_proba(text_vectorized)[0]
        
        # Get confidence score
        confidence = max(probability)
        
        return prediction, confidence, None
        
    except Exception as e:
        return None, None, f"Prediction error: {e}"

def get_prediction_insights(prediction, confidence, text_length):
    """
    Generate insights based on the prediction
    """
    insights = []
    
    if prediction == 1:  # Fake
        if confidence > 0.8:
            insights.append("🔴 High confidence fake detection")
        elif confidence > 0.6:
            insights.append("⚠️ Medium confidence fake detection")
        else:
            insights.append("❓ Low confidence fake detection")
    else:  # Real
        if confidence > 0.8:
            insights.append("✅ High confidence real job")
        elif confidence > 0.6:
            insights.append("⚠️ Medium confidence real job")
        else:
            insights.append("❓ Low confidence real job")
    
    # Add text length insight
    if text_length < 100:
        insights.append("⚠️ Very short description - limited information")
    elif text_length < 500:
        insights.append("📝 Short description - moderate information")
    else:
        insights.append("📄 Detailed description - good information")
    
    return insights

# ------------------------------------------------------------------------------------
# 5. UI LAYOUT
# ------------------------------------------------------------------------------------

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-container {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .insight-item {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
        border-left: 4px solid #1f77b4;
        color: #333;
        font-weight: 500;
    }
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# 4.1 HEADER
st.markdown('<h1 class="main-header">🔍 Fake Job Posting Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Detect fraudulent job postings using advanced machine learning</p>', unsafe_allow_html=True)

# Check if model loaded successfully
if model is None or vectorizer is None:
    st.error("❌ Model failed to load. Please check the model files and try again.")
    st.stop()

# Create two columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    # 4.2 USER INPUT
    st.markdown("### 📝 Job Description Input")
    
    # Initialize session state for job description
    if 'job_description' not in st.session_state:
        st.session_state.job_description = ""
    
    # Text area for job description
    job_description = st.text_area(
        "Paste the job description below:",
        value=st.session_state.job_description,
        height=300,
        placeholder="Enter the job title, company description, requirements, benefits, and any other relevant information...",
        help="Include as much detail as possible for better accuracy"
    )
    
    # 4.3 BUTTON
    classify_button = st.button(
        "🔍 Classify Job Posting",
        type="primary",
        use_container_width=True,
        help="Click to analyze the job posting"
    )

with col2:
    # Sidebar with information
    st.markdown("### ℹ️ How It Works")
    st.markdown("""
    This classifier uses a **Random Forest model** trained on thousands of real and fake job postings.
    
    **Features analyzed:**
    - Job title and description
    - Company profile
    - Requirements and benefits
    - Language patterns and red flags
    
    **Accuracy:** 98.1% on standard cases
    """)
    
    # Model info
    st.markdown("### 📊 Model Information")
    st.markdown("""
    - **Model Type:** Random Forest
    - **Training Data:** 17,880 job postings
    - **Features:** TF-IDF text vectors
    - **Optimization:** Cross-validation + hyperparameter tuning
    """)
    
    # Tips
    st.markdown("### 💡 Tips for Better Results")
    st.markdown("""
    - Include the complete job description
    - Add company information if available
    - Include requirements and benefits
    - Longer descriptions provide better accuracy
    """)

# 4.4 RESULT CONTAINER
if classify_button:
    if not job_description or job_description.strip() == '':
        st.warning("⚠️ Please enter a job description to classify.")
    else:
        # Show loading spinner
        with st.spinner("🔍 Analyzing job posting..."):
            start_time = time.time()  # Start timing
            
            # Make prediction
            prediction, confidence, error = predict_job_posting(job_description, model, vectorizer)
            
            end_time = time.time()  # End timing
            analysis_time = end_time - start_time
            
            if error:
                st.error(f"❌ {error}")
            else:
                # Calculate text length
                text_length = len(job_description)
                
                # Get insights
                insights = get_prediction_insights(prediction, confidence, text_length)
                
                # Display result
                st.markdown("### 🎯 Classification Result")
                
                if prediction == 1:  # Fake
                    st.error("🚨 **FAKE JOB POSTING DETECTED**")
                    st.markdown("""
                    This job posting shows characteristics commonly associated with fraudulent listings.
                    **Exercise caution** and verify the company and job details before proceeding.
                    """)
                else:  # Real
                    st.success("✅ **LEGITIMATE JOB POSTING**")
                    st.markdown("""
                    This job posting appears to be legitimate based on our analysis.
                    **Still recommended** to verify company details and research the employer.
                    """)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Confidence Score",
                        f"{confidence:.1%}",
                        help="Model's confidence in this prediction"
                    )
                
                with col2:
                    st.metric(
                        "Text Length",
                        f"{text_length:,} chars",
                        help="Length of the job description"
                    )
                
                with col3:
                    st.metric(
                        "Analysis Time",
                        f"{analysis_time:.2f}s",
                        help="Time taken for analysis"
                    )
                
                # Display insights
                st.markdown("### 🔍 Analysis Insights")
                for insight in insights:
                    st.markdown(f'<div class="insight-item">{insight}</div>', unsafe_allow_html=True)
                
                # Show feature importance (if available)
                if hasattr(model, 'feature_importances_'):
                    st.markdown("### 📈 Key Factors")
                    st.info("""
                    The model considers various text features including:
                    - Specific keywords and phrases
                    - Language patterns
                    - Text structure and formatting
                    - Presence of suspicious elements
                    """)

# ------------------------------------------------------------------------------------
# 6. OPTIONAL GPT COMPARISON
# ------------------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🤖 Advanced Analysis Options")

# GPT-4 comparison checkbox
gpt4_comparison = st.checkbox(
    "Compare with GPT-4 Analysis",
    help="Get additional analysis from GPT-4 (requires OpenAI API key)"
)

if gpt4_comparison:
    st.info("🔧 **GPT-4 Integration Coming Soon**")
    st.markdown("""
    This feature will provide:
    - **Detailed reasoning** for the classification
    - **Specific red flags** identified in the text
    - **Recommendations** for verification
    - **Comparison** between ML model and GPT-4 predictions
    
    **Setup required:**
    1. OpenAI API key
    2. GPT-4 API access
    3. Environment variables configuration
    
    *This feature is currently in development.*
    """)

# ------------------------------------------------------------------------------------
# 7. ADDITIONAL FEATURES
# ------------------------------------------------------------------------------------
st.markdown("---")

# Sample job postings for testing
with st.expander("🧪 Test with Sample Job Postings"):
    st.markdown("### Sample Real Job Posting")
    real_sample = """
    Software Engineer - Python Development
    
    Company: TechCorp Inc.
    Location: San Francisco, CA
    Type: Full-time
    
    We are seeking a talented Python developer to join our growing team. 
    You will work on developing scalable web applications and APIs.
    
    Requirements:
    - 3+ years Python experience
    - Knowledge of Django/Flask
    - Experience with databases
    - Bachelor's degree in Computer Science
    
    Benefits:
    - Competitive salary
    - Health insurance
    - 401(k) matching
    - Flexible work hours
    - Professional development budget
    """
    
    if st.button("Load Real Sample"):
        st.session_state.job_description = real_sample
        st.success("✅ Real job sample loaded! Click 'Classify Job Posting' to test.")
        st.rerun()
    
    st.markdown("### Sample Fake Job Posting")
    fake_sample = """
    WORK FROM HOME - EARN $5000 PER WEEK!!!
    
    NO EXPERIENCE NEEDED - START IMMEDIATELY!!!
    
    Join our amazing team and make money from your couch!
    
    Requirements:
    - Computer and internet
    - No experience needed
    - No education required
    
    Benefits:
    - Work from anywhere
    - Flexible hours
    - High pay
    - Quick start
    
    CLICK HERE TO APPLY NOW!!!
    LIMITED TIME OFFER!!!
    """
    
    if st.button("Load Fake Sample"):
        st.session_state.job_description = fake_sample
        st.success("✅ Fake job sample loaded! Click 'Classify Job Posting' to test.")
        st.rerun()
    
    # Clear button
    if st.button("Clear Text"):
        st.session_state.job_description = ""
        st.rerun()

# ------------------------------------------------------------------------------------
# 8. FOOTER
# ------------------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🔍 Fake Job Posting Classifier | Built with Streamlit & Machine Learning</p>
    <p>Model Accuracy: 98.1% | Last Updated: August 2024</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------------
# 9. DEPLOYMENT TIPS (IN COMMENTS)
# ------------------------------------------------------------------------------------
"""
DEPLOYMENT INSTRUCTIONS:
=======================

LOCAL RUNNING:
-------------
1. Install requirements: pip install -r requirements.txt
2. Run the app: streamlit run app.py
3. Open browser: http://localhost:8501

STREAMLIT CLOUD DEPLOYMENT:
--------------------------
1. Push code to GitHub repository
2. Go to share.streamlit.io
3. Connect GitHub account
4. Select repository and main file (app.py)
5. Deploy automatically

REQUIRED FILES:
--------------
- app.py (this file)
- models/random_forest_cv.pkl
- model/tfidf_vectorizer.pkl
- requirements.txt

ENVIRONMENT VARIABLES (for GPT-4):
---------------------------------
- OPENAI_API_KEY (if implementing GPT-4 comparison)
""" 