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

    /* Button Styles */
    div.stButton > button {
        font-weight: 600;
        border-radius: 12px;
        height: 56px;
        font-size: 18px;
        transition: all 0.2s ease;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        border: none;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        box-shadow: 0 6px 16px rgba(239, 68, 68, 0.4);
        transform: translateY(-2px);
    }
    div.stButton > button[kind="secondary"] {
        background-color: white;
        color: #475569;
        border: 2px solid #cbd5e1;
    }
    div.stButton > button[kind="secondary"]:hover {
        border-color: #94a3b8;
        background-color: #f8fafc;
    }
    
    /* Typography improvements */
    h1, h2, h3 {
        color: #0f172a;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    dt, dd, p {
        color: #334155;
    }

    /* Input Card Form */
    .input-card {
        background: white;
        padding: 30px 24px;
        border-radius: 20px;
        box-shadow: 0 4px 10px -2px rgba(0, 0, 0, 0.05), 0 2px 5px -2px rgba(0, 0, 0, 0.05);
        margin-top: 16px;
        margin-bottom: 24px;
        border: 1px solid #f1f5f9;
        min-height: 250px;
    }
    .question-title {
        color: #1e293b;
        font-family: 'Inter', -apple-system, sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 20px;
        margin-top: 0;
        line-height: 1.3;
    }
    .step-text {
        font-size: 0.85rem;
        color: #64748b;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 700;
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

# =============== SESSION STATE SETUP ===============
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'p_name' not in st.session_state:
    st.session_state.p_name = ""
if 'p_age' not in st.session_state:
    st.session_state.p_age = 45
if 'p_hypertension' not in st.session_state:
    st.session_state.p_hypertension = "No"
if 'p_heart_disease' not in st.session_state:
    st.session_state.p_heart_disease = "No"
if 'p_glucose' not in st.session_state:
    st.session_state.p_glucose = 120.0
if 'p_bmi' not in st.session_state:
    st.session_state.p_bmi = 25.0

def next_step(): st.session_state.step += 1
def prev_step(): st.session_state.step -= 1
def reset_app():
    for key in ['step', 'p_name', 'p_age', 'p_hypertension', 'p_heart_disease', 'p_glucose', 'p_bmi']:
        if key in st.session_state:
            del st.session_state[key]

# =============== UI LAYOUT (WIZARD) ===============

# 1. App Header
col_logo, col_title = st.columns([1, 4])
with col_logo:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=65)
with col_title:
    st.markdown("<h2 style='margin:0; padding-top:12px; font-weight: 800;'>Stroke Risk AI</h2>", unsafe_allow_html=True)

st.markdown("<p style='color: #64748b; margin-top: -5px; margin-bottom: 20px;'>Step-by-step biometric assessment</p>", unsafe_allow_html=True)

total_steps = 6

if st.session_state.step <= total_steps:
    # Progress Bar
    progress_val = (st.session_state.step - 1) / total_steps
    st.progress(progress_val)
    
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='step-text'>STEP {st.session_state.step} OF {total_steps}</div>", unsafe_allow_html=True)
    
    if st.session_state.step == 1:
        st.markdown("<div class='question-title'>What is the patient's full name?</div>", unsafe_allow_html=True)
        st.text_input("Name", key="p_name", label_visibility="collapsed", placeholder="e.g. John Doe")
        st.write("")
        if st.button("Continue ➔", type="primary", use_container_width=True):
            if st.session_state.p_name.strip():
                next_step()
                st.rerun()
            else:
                st.error("Please enter a name to continue.")
                
    elif st.session_state.step == 2:
        first_name = st.session_state.p_name.split()[0] if st.session_state.p_name else "the patient"
        st.markdown(f"<div class='question-title'>How old is {first_name}?</div>", unsafe_allow_html=True)
        st.slider("Age (Years)", 0, 100, key="p_age", label_visibility="collapsed")
        st.write("")
        col1, col2 = st.columns(2)
        with col1:
            st.button("⬅️ Back", on_click=prev_step, use_container_width=True)
        with col2:
            st.button("Continue ➔", on_click=next_step, type="primary", use_container_width=True)
            
    elif st.session_state.step == 3:
        st.markdown(f"<div class='question-title'>Do they have hypertension (high blood pressure)?</div>", unsafe_allow_html=True)
        st.selectbox("Hypertension", ["No", "Yes"], key="p_hypertension", label_visibility="collapsed")
        st.write("")
        col1, col2 = st.columns(2)
        with col1:
            st.button("⬅️ Back", on_click=prev_step, use_container_width=True)
        with col2:
            st.button("Continue ➔", on_click=next_step, type="primary", use_container_width=True)
            
    elif st.session_state.step == 4:
        st.markdown(f"<div class='question-title'>Is there any history of heart disease?</div>", unsafe_allow_html=True)
        st.selectbox("Heart Disease", ["No", "Yes"], key="p_heart_disease", label_visibility="collapsed")
        st.write("")
        col1, col2 = st.columns(2)
        with col1:
            st.button("⬅️ Back", on_click=prev_step, use_container_width=True)
        with col2:
            st.button("Continue ➔", on_click=next_step, type="primary", use_container_width=True)
            
    elif st.session_state.step == 5:
        st.markdown(f"<div class='question-title'>What is their average blood glucose level (mg/dL)?</div>", unsafe_allow_html=True)
        st.number_input("Average Glucose Level", 50.0, 300.0, key="p_glucose", label_visibility="collapsed")
        st.caption("*(Normal fasting glucose is ~70-99 mg/dL)*")
        st.write("")
        col1, col2 = st.columns(2)
        with col1:
            st.button("⬅️ Back", on_click=prev_step, use_container_width=True)
        with col2:
            st.button("Continue ➔", on_click=next_step, type="primary", use_container_width=True)
            
    elif st.session_state.step == 6:
        st.markdown(f"<div class='question-title'>Finally, what is their Body Mass Index (BMI)?</div>", unsafe_allow_html=True)
        st.number_input("BMI", 10.0, 60.0, key="p_bmi", label_visibility="collapsed")
        st.caption("*(Normal healthy BMI is 18.5-24.9)*")
        st.write("")
        col1, col2 = st.columns(2)
        with col1:
            st.button("⬅️ Back", on_click=prev_step, use_container_width=True)
        with col2:
            st.button("Analyze Risk", on_click=next_step, type="primary", use_container_width=True)
            
    st.markdown("</div>", unsafe_allow_html=True)

# =============== RESULT PAGE (STEP 7) ===============
else:
    st.progress(1.0)
    
    # Process inputs for model
    age = st.session_state.p_age
    hypertension = 1 if st.session_state.p_hypertension == "Yes" else 0
    heart_disease = 1 if st.session_state.p_heart_disease == "Yes" else 0
    avg_glucose = st.session_state.p_glucose
    bmi = st.session_state.p_bmi
    patient_name = st.session_state.p_name

    input_data = np.array([[age, hypertension, heart_disease, avg_glucose, bmi]])
    
    # Scale & Reshape
    input_scaled = scaler.transform(input_data)
    input_scaled = input_scaled.reshape(1, input_scaled.shape[1], 1)
    
    # Predict
    with st.spinner('Analyzing biometrics...'):
        prediction_prob = model.predict(input_scaled)[0][0]
    
    # Output Result
    risk_percentage = prediction_prob * 100
    
    if prediction_prob > 0.5:
        st.markdown(f"""
            <div class="prediction-box danger">
                <h2>⚠️ HIGH RISK DETECTED</h2>
                <p>The model predicts an elevated probability of stroke for {patient_name}.</p>
                <h1 style="font-size: 3.5rem; margin: 15px 0;">{risk_percentage:.1f}%</h1>
                <p style="font-size: 0.9rem; margin-top: 10px; font-weight: bold;">Immediate medical consultation recommended</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="prediction-box safe">
                <h2>✅ LOW RISK</h2>
                <p>The model predicts a low probability of stroke for {patient_name}.</p>
                <h1 style="font-size: 3.5rem; margin: 15px 0;">{risk_percentage:.1f}%</h1>
            </div>
        """, unsafe_allow_html=True)
    
    st.caption("**Risk Probability Meter**")
    st.progress(float(prediction_prob))

    # --- Examination Report Section ---
    st.write("")
    st.markdown("#### 📄 Examination Report")
    
    report_data = {
        "Metric": ["Patient Name", "Age", "Hypertension", "Heart Disease", "Avg. Glucose", "BMI", "Risk Score"],
        "Value": [
            f"{patient_name}",
            f"{age}",
            "Yes" if hypertension == 1 else "No",
            "Yes" if heart_disease == 1 else "No",
            f"{avg_glucose}",
            f"{bmi}",
            f"{risk_percentage:.1f}%"
        ]
    }
    
    report_df = pd.DataFrame(report_data)
    st.dataframe(report_df, use_container_width=True, hide_index=True)
    
    # --- Generate PDF Logic ---
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
    pdf.add_metric('Name:', f"{patient_name}", "-")
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
    col_d, col_r = st.columns(2)
    with col_d:
        st.download_button(
            label="📥 Download PDF",
            data=pdf_output,
            file_name=f"stroke_report_{patient_name.replace(' ', '_')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    with col_r:
        st.button("↺ Start Over", on_click=reset_app, use_container_width=True)

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