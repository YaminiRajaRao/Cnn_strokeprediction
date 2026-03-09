import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from fpdf import FPDF

# Page Configuration
st.set_page_config(
    page_title='Stroke Risk Predictor',
    page_icon='❤️',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        height: 50px;
        font-size: 18px;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        border-color: #ff3333;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .safe {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .danger {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
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

# Header
st.title('❤️ Stroke Risk Prediction AI')
st.markdown("### Advanced CNN-based Risk Assessment Tool")
st.markdown("---")

# Sidebar for Inputs
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100)
    st.header("Patient Data")
    
    age = st.slider('Age', 0, 100, 45, help="Patient's age in years")
    
    col1, col2 = st.columns(2)
    with col1:
        hypertension = st.selectbox('Hypertension', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    with col2:
        heart_disease = st.selectbox('Heart Disease', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    
    avg_glucose = st.number_input('Average Glucose Level', 50.0, 300.0, 120.0, help="Average blood glucose level (mg/dL)")
    bmi = st.number_input('BMI', 10.0, 60.0, 25.0, help="Body Mass Index")
    
    st.markdown("---")
    predict_btn = st.button('Analyze Risk')

# Main Content Area
col_main, col_info = st.columns([2, 1])

with col_main:
    if predict_btn:
        # Prepare Input
        input_data = np.array([[age, hypertension, heart_disease, avg_glucose, bmi]])
        
        # Scale & Reshape
        input_scaled = scaler.transform(input_data)
        input_scaled = input_scaled.reshape(1, input_scaled.shape[1], 1)
        
        # Predict
        with st.spinner('Analyzing...'):
            prediction_prob = model.predict(input_scaled)[0][0]
        
        # Display Results
        st.subheader("Analysis Result")
        
        risk_percentage = prediction_prob * 100
        
        if prediction_prob > 0.5:
            st.markdown(f"""
                <div class="prediction-box danger">
                    <h2>⚠️ HIGH RISK DETECTED</h2>
                    <p>The model predicts a high probability of stroke.</p>
                    <h1>{risk_percentage:.1f}%</h1>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="prediction-box safe">
                    <h2>✅ LOW RISK</h2>
                    <p>The model predicts a low probability of stroke.</p>
                    <h1>{risk_percentage:.1f}%</h1>
                </div>
            """, unsafe_allow_html=True)
        
        st.write("")
        st.write("### Risk Probability")
        st.progress(float(prediction_prob))

        # --- Report Section ---
        st.markdown("---")
        st.subheader("📋 Patient Examination Report")
        
        # Create Report Data
        report_data = {
            "Parameter": ["Age", "Hypertension", "Heart Disease", "Average Glucose Level", "BMI", "Stroke Risk Probability"],
            "Value": [
                f"{age} years",
                "Yes" if hypertension == 1 else "No",
                "Yes" if heart_disease == 1 else "No",
                f"{avg_glucose} mg/dL",
                f"{bmi}",
                f"{risk_percentage:.2f}%"
            ],
            "Reference / context": [
                "0 - 100",
                "-",
                "-", 
                "50 - 300 (Observed Min-Max)", 
                "10 - 60 (Observed Min-Max)",
                "0% - 100%"
            ]
        }
        
        report_df = pd.DataFrame(report_data)
        st.table(report_df)
        
        # Generate PDF Report
        class PDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 16)
                self.cell(0, 10, 'Stroke Risk Assessment Report', 0, 1, 'C')
                self.set_font('Arial', 'I', 10)
                self.cell(0, 10, f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
                self.ln(10)

            def chapter_title(self, title):
                self.set_font('Arial', 'B', 12)
                self.set_fill_color(240, 240, 240)
                self.cell(0, 10, title, 0, 1, 'L', 1)
                self.ln(4)

            def chapter_body(self, body):
                self.set_font('Arial', '', 11)
                self.multi_cell(0, 10, body)
                self.ln()
                
            def add_metric(self, metric, value, reference):
                self.set_font('Arial', 'B', 11)
                self.cell(60, 10, metric, 0, 0)
                self.set_font('Arial', '', 11)
                self.cell(60, 10, str(value), 0, 0)
                self.set_font('Arial', 'I', 10)
                self.cell(0, 10, f"Ref: {reference}", 0, 1)

        pdf = PDF()
        pdf.add_page()
        
        # Patient Details
        pdf.chapter_title('Patient Details')
        pdf.add_metric('Age:', f"{age} years", "0-100")
        pdf.add_metric('Hypertension:', "Yes" if hypertension else "No", "-")
        pdf.add_metric('Heart Disease:', "Yes" if heart_disease else "No", "-")
        pdf.ln(5)
        
        # Clinical Metrics
        pdf.chapter_title('Clinical Metrics')
        pdf.add_metric('Avg. Glucose Level:', f"{avg_glucose} mg/dL", "50-300")
        pdf.add_metric('BMI:', f"{bmi}", "10-60")
        pdf.ln(5)
        
        # Analysis Result
        pdf.chapter_title('Analysis Result')
        pdf.set_font('Arial', 'B', 14)
        if prediction_prob > 0.5:
            pdf.set_text_color(194, 24, 7) # Red
            pdf.cell(0, 10, f"Result: HIGH RISK ({risk_percentage:.1f}%)", 0, 1, 'L')
        else:
            pdf.set_text_color(34, 139, 34) # Green
            pdf.cell(0, 10, f"Result: LOW RISK ({risk_percentage:.1f}%)", 0, 1, 'L')
            
        pdf.set_text_color(0, 0, 0) # Reset color
        pdf.ln(5)
        
        # Disclaimer
        pdf.set_font('Arial', 'I', 9)
        pdf.multi_cell(0, 10, "DISCLAIMER: This report is generated by an AI model and should not be considered a medical diagnosis. Please consult a healthcare professional.")

        pdf_output = pdf.output(dest='S').encode('latin-1')
        
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_output,
            file_name="stroke_prediction_report.pdf",
            mime="application/pdf"
        )

    st.info("ℹ️ **About the Model**")
    st.write("""
    This tool uses a 1D Convolutional Neural Network (CNN) trained on patient health data to estimate stroke risk.
    
    **Key Factors:**
    - **Age**: Risk increases with age.
    - **Hypertension**: High blood pressure is a major risk factor.
    - **Heart Disease**: History of heart issues correlates with stroke risk.
    - **Glucose & BMI**: Metabolic health indicators.
    """)
    st.warning("⚠️ **Disclaimer**: This is an AI-assisted tool for educational purposes only. Always consult a healthcare professional for medical advice.")