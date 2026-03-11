import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from fpdf import FPDF

# Page Configuration - Centered layout for mobile, collapse sidebar
st.set_page_config(
    page_title='Stroke Risk Predictor',
    page_icon='❤️',
    layout='centered',
    initial_sidebar_state='collapsed'
)

# Custom CSS for Mobile-like Web App look
st.markdown("""
    <style>
    /* Hide Streamlit native headers and footers to look like a standalone web app */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Adjust container for mobile screens */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 600px;
    }
    
    /* Add a subtle background color for depth */
    .stApp {
        background-color: #f8fafc;
    }

    /* Style the main predict button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        font-weight: 600;
        border-radius: 12px;
        height: 56px;
        font-size: 18px;
        border: none;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        box-shadow: 0 6px 16px rgba(239, 68, 68, 0.4);
        transform: translateY(-2px);
    }
    
    /* Typography improvements */
    h1, h2, h3 {
        color: #0f172a;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    dt, dd, p {
        color: #334155;
    }

    /* Prediction Result Cards */
    .prediction-box {
        padding: 30px 20px;
        border-radius: 20px;
        text-align: center;
        margin: 24px 0;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        animation: slideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1);
    }
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .safe {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #86efac;
        color: #166534;
    }
    .safe h2 { color: #15803d; }
    
    .danger {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 2px solid #fca5a5;
        color: #991b1b;
    }
    .danger h2 { color: #b91c1c; }
    
    /* Custom input card wrapper */
    .input-card {
        background: white;
        padding: 24px;
        border-radius: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        margin-bottom: 24px;
        border: 1px solid #f1f5f9;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_assets():
    model = load_model('model.h5')
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# =============== UI LAYOUT ===============

# 1. App Header
col_logo, col_title = st.columns([1, 4])
with col_logo:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=65)
with col_title:
    st.markdown("<h2 style='margin:0; padding-top:12px; font-weight: 800;'>Stroke Risk AI</h2>", unsafe_allow_html=True)

st.markdown("<p style='color: #64748b; margin-top: -5px; margin-bottom: 24px;'>Advanced CNN-based risk assessment tool</p>", unsafe_allow_html=True)

# 2. Main Form Inputs (Moved from Sidebar to Center)
st.markdown("<div class='input-card'>", unsafe_allow_html=True)
st.markdown("<h4 style='margin-top:0; color:#1e293b;'>📋 Patient Details</h4>", unsafe_allow_html=True)

age = st.slider('Age (Years)', 0, 100, 45)

# Use columns to tightly pack inputs on mobile
col_h1, col_h2 = st.columns(2)
with col_h1:
    hypertension = st.selectbox('Hypertension?', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
with col_h2:
    heart_disease = st.selectbox('Heart Disease?', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

col_v1, col_v2 = st.columns(2)
with col_v1:
    avg_glucose = st.number_input('Glucose (mg/dL)', 50.0, 300.0, 120.0)
with col_v2:
    bmi = st.number_input('BMI Rating', 10.0, 60.0, 25.0)

st.markdown("</div>", unsafe_allow_html=True)

# 3. Predict Button
predict_btn = st.button('Analyze Risk Now', use_container_width=True)

# =============== APP LOGIC ===============

if predict_btn:
    # Prepare Input
    input_data = np.array([[age, hypertension, heart_disease, avg_glucose, bmi]])
    
    # Scale & Reshape for 1D CNN
    input_scaled = scaler.transform(input_data)
    input_scaled = input_scaled.reshape(1, input_scaled.shape[1], 1)
    
    # Predict
    with st.spinner('Analyzing biometrics...'):
        prediction_prob = model.predict(input_scaled)[0][0]
    
    # Generate Output
    st.markdown("---")
    
    risk_percentage = prediction_prob * 100
    
    if prediction_prob > 0.5:
        st.markdown(f"""
            <div class="prediction-box danger">
                <h2>⚠️ HIGH RISK DETECTED</h2>
                <p>The model predicts an elevated probability of stroke.</p>
                <h1 style="font-size: 3.5rem; margin: 15px 0;">{risk_percentage:.1f}%</h1>
                <p style="font-size: 0.9rem; margin-top: 10px; font-weight: bold;">Immediate medical consultation recommended</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="prediction-box safe">
                <h2>✅ LOW RISK</h2>
                <p>The model predicts a low probability of stroke.</p>
                <h1 style="font-size: 3.5rem; margin: 15px 0;">{risk_percentage:.1f}%</h1>
            </div>
        """, unsafe_allow_html=True)
    
    st.caption("**Risk Probability Meter**")
    st.progress(float(prediction_prob))

    # --- Examination Report Section ---
    st.write("")
    st.markdown("#### 📄 Examination Report")
    
    report_data = {
        "Metric": ["Age", "Hypertension", "Heart Disease", "Avg. Glucose", "BMI", "Risk Score"],
        "Value": [
            f"{age}",
            "Yes" if hypertension == 1 else "No",
            "Yes" if heart_disease == 1 else "No",
            f"{avg_glucose}",
            f"{bmi}",
            f"{risk_percentage:.1f}%"
        ]
    }
    
    report_df = pd.DataFrame(report_data)
    
    # Display table beautifully
    st.dataframe(report_df, use_container_width=True, hide_index=True)
    
    # --- Generate PDF Background Logic ---
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 16)
            self.cell(0, 10, 'Stroke Risk Assessment Report', 0, 1, 'C')
            self.set_font('Arial', 'I', 10)
            self.cell(0, 10, f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", 0, 1, 'C')
            self.ln(10)

        def chapter_title(self, title):
            self.set_font('Arial', 'B', 12)
            self.set_fill_color(240, 240, 240)
            self.cell(0, 10, title, 0, 1, 'L', 1)
            self.ln(4)

        def add_metric(self, metric, value, reference):
            self.set_font('Arial', 'B', 11)
            self.cell(60, 10, metric, 0, 0)
            self.set_font('Arial', '', 11)
            self.cell(60, 10, str(value), 0, 0)
            self.set_font('Arial', 'I', 10)
            self.cell(0, 10, f"Ref: {reference}", 0, 1)

    pdf = PDF()
    pdf.add_page()
    
    pdf.chapter_title('Patient Details')
    pdf.add_metric('Age:', f"{age} years", "0-100")
    pdf.add_metric('Hypertension:', "Yes" if hypertension else "No", "-")
    pdf.add_metric('Heart Disease:', "Yes" if heart_disease else "No", "-")
    pdf.ln(5)
    
    pdf.chapter_title('Clinical Metrics')
    pdf.add_metric('Avg. Glucose Level:', f"{avg_glucose} mg/dL", "50-300")
    pdf.add_metric('BMI:', f"{bmi}", "10-60")
    pdf.ln(5)
    
    pdf.chapter_title('Analysis Result')
    pdf.set_font('Arial', 'B', 14)
    if prediction_prob > 0.5:
        pdf.set_text_color(194, 24, 7)
        pdf.cell(0, 10, f"Result: HIGH RISK ({risk_percentage:.1f}%)", 0, 1, 'L')
    else:
        pdf.set_text_color(34, 139, 34)
        pdf.cell(0, 10, f"Result: LOW RISK ({risk_percentage:.1f}%)", 0, 1, 'L')
        
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    
    pdf.set_font('Arial', 'I', 9)
    pdf.multi_cell(0, 10, "DISCLAIMER: This report is generated by an AI model and should not be considered a medical diagnosis. Please consult a healthcare professional.")

    pdf_output = pdf.output(dest='S').encode('latin-1')
    
    st.write("")
    st.download_button(
        label="📥 Download PDF Report",
        data=pdf_output,
        file_name="stroke_prediction_report.pdf",
        mime="application/pdf",
        use_container_width=True
    )

st.markdown("<br>", unsafe_allow_html=True)
with st.expander("ℹ️ About the Model Sandbox"):
    st.write("""
    This application utilizes a 1D Convolutional Neural Network (CNN) trained on patient health data.
    
    **Key Factors Evaluated:**
    - **Age**: Risk increases with age.
    - **Hypertension**: High blood pressure is a major risk factor.
    - **Heart Disease**: History of heart issues correlates with stroke risk.
    - **Glucose & BMI**: Metabolic health indicators.
    ---
    *Disclaimer: This is an AI-assisted tool for educational/demonstration purposes only. Always consult a healthcare professional for medical advice.*
    """)