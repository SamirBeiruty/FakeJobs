# Fake Job Post Detector - Apple-Level ML Pipeline with GenAI Comparison

**A comprehensive machine learning project that detects fake job postings using text analysis, with GenAI benchmarking and real-world deployment.**

## 🎯 Project Overview

This project demonstrates:
- ✅ Practical ML engineering skills
- ✅ Text-based feature extraction and classification  
- ✅ Model explainability with SHAP
- ✅ GenAI integration for benchmarking
- ✅ Deployed UI for real-world usage
- ✅ Apple-level technical storytelling

## 📊 Dataset

- **Source**: Kaggle "Fake Job Postings" (Sample dataset created for demonstration)
- **Size**: 58 job posts (35 real, 23 fake)
- **Target**: Binary classification (0 = real, 1 = fake)
- **Text Fields**: Combined from title, company_profile, description, requirements, benefits, department

## 🚀 Project Progress

### ✅ Phase 1: Data Loading and Cleaning (COMPLETED)

**Objective**: Load raw dataset and prepare for ML pipeline

**Completed Tasks**:
1. ✅ Created project structure (`data/`, `model/`, `app/`, `notebooks/`)
2. ✅ Generated sample dataset with realistic fake/real job postings
3. ✅ Implemented comprehensive data cleaning pipeline:
   - Removed data leakage columns (job_id, salary_range, telecommuting, etc.)
   - Combined relevant text fields into single `combined_text` column
   - Applied text cleaning (lowercase, HTML removal, punctuation removal)
   - Preserved only essential columns: `fraudulent` (target) and `combined_text` (features)

**Results**:
- **Final Dataset**: 58 posts with 60.3% real jobs, 39.7% fake jobs
- **Text Length**: Average 511 characters (range: 366-658)
- **Data Quality**: No missing labels, clean text format ready for TF-IDF
- **Output Files**: 
  - `data/cleaned_job_postings.csv` - Clean dataset
  - `data/cleaned_job_postings.pkl` - Pickle format for fast loading

**Key Accomplishments**:
- Robust text preprocessing pipeline
- Proper data leakage prevention
- Balanced dataset for classification
- Professional code structure with comprehensive documentation

---

## 📂 Project Structure

```
fake-job-detector/
├── data/
│   ├── fake_job_postings.csv          # Original dataset
│   ├── cleaned_job_postings.csv       # Phase 1 output
│   └── cleaned_job_postings.pkl       # Fast-loading format
├── model/                             # Model artifacts (Phase 2+)
├── app/                              # Streamlit web app (Phase 11)
├── notebooks/                        # Analysis notebooks (Phase 8)
├── phase1_data_cleaning.py           # Phase 1 implementation
├── create_sample_dataset.py          # Dataset generation
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 🛠️ Technical Stack

- **Language**: Python 3.13
- **Data Processing**: pandas, numpy
- **ML Framework**: scikit-learn, XGBoost
- **Explainability**: SHAP
- **GenAI**: OpenAI GPT-4
- **Web App**: Streamlit
- **Visualization**: matplotlib, seaborn, plotly

## 🔄 Next Phases

- **Phase 2**: TF-IDF Feature Extraction (3000 features, English stopwords)
- **Phase 3**: Train-Test Split (80/20, stratified)
- **Phase 4**: XGBoost Model Training
- **Phase 5**: Model Evaluation (accuracy, precision, recall, F1, confusion matrix)
- **Phase 6**: 5-fold Cross-Validation
- **Phase 7**: SHAP Explainability Analysis
- **Phase 8**: Model Serialization
- **Phase 9**: GPT-4 Benchmarking
- **Phase 10**: Performance Comparison Analysis
- **Phase 11**: Streamlit Web Application
- **Phase 12**: Documentation & Deployment

---

*Ready for interview-level technical demonstration at Apple and other top-tier tech companies.*
