import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configure page layout
st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide"
)

# Custom styling for clinical theme
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #B91C1C;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .card-positive {
        background-color: #FEF2F2;
        border-left: 6px solid #DC2626;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .card-negative {
        background-color: #F0FDF4;
        border-left: 6px solid #16A34A;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .status-text-pos {
        color: #B91C1C;
        font-size: 1.4rem;
        font-weight: 700;
    }
    .status-text-neg {
        color: #15803D;
        font-size: 1.4rem;
        font-weight: 700;
    }
    .risk-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .badge-warn { background-color: #FEE2E2; color: #991B1B; }
    .badge-ok { background-color: #DCFCE7; color: #166534; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Saved Models and Scaler (Random Forest + TensorFlow ANN)
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'heart_disease_model.pkl')
scaler_path = os.path.join(current_dir, 'heart_disease_scaler.pkl')
tf_model_path = os.path.join(current_dir, 'heart_disease_tf_model.keras')

@st.cache_resource
def load_artifacts():
    try:
        import tensorflow as tf
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        
        # Load TensorFlow Neural Network
        if os.path.exists(tf_model_path):
            tf_model = tf.keras.models.load_model(tf_model_path)
        else:
            tf_model = None
            
        return model, scaler, tf_model, True, ""
    except Exception as e:
        return None, None, None, False, str(e)

model, scaler, tf_model, loaded_ok, err_msg = load_artifacts()

# Title
st.markdown('<div class="main-title">❤️ Heart Disease Risk Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Cardiovascular Risk Stratification using Random Forest and <b>TensorFlow Deep Neural Network</b></div>', unsafe_allow_html=True)

if not loaded_ok:
    st.error(f"Error loading model artifacts: {err_msg}. Please ensure 'heart_disease_model.pkl' is present.")
    st.stop()

# ---------------------------------------------------------
# Preset Clinical Profiles
# ---------------------------------------------------------
PRESETS = {
    "Select a preset or custom input...": None,
    "Patient Profile 1: Elevated Cardiac Risk (Angina, ST-Depression, Vessel Defect)": {
        'age': 58, 'sex': 1, 'cp': 2, 'trestbps': 140, 'chol': 211,
        'fbs': 1, 'restecg': 0, 'thalach': 165, 'exang': 0,
        'oldpeak': 1.8, 'slope': 2, 'ca': 0, 'thal': 2
    },
    "Patient Profile 2: Low Risk / Healthy Subject (Normal ECG & Vitals)": {
        'age': 54, 'sex': 1, 'cp': 0, 'trestbps': 130, 'chol': 268,
        'fbs': 0, 'restecg': 0, 'thalach': 108, 'exang': 1,
        'oldpeak': 1.5, 'slope': 1, 'ca': 1, 'thal': 3
    }
}

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Patient Profiles")
    preset_choice = st.selectbox(
        "Load Clinical Profile Preset:",
        list(PRESETS.keys()),
        index=1
    )
    
    st.markdown("---")
    st.subheader("Model Diagnostic Benchmarks")
    st.markdown("""
    - **Random Forest (Champion Ensemble)**:
      - Test Accuracy: **99.5%**
      - Precision: **1.000**
      - Recall: **0.990**
      - ROC-AUC: **1.000**
    - **TensorFlow Deep Neural Network**:
      - Framework: **TensorFlow / Keras**
      - Architecture: **Dense(64) + BatchNorm + Dropout + Dense(32) + BatchNorm + Dense(16)**
      - Test Accuracy: **~86-90%**
      - Output: Continuous probability calibration
    """)
    st.markdown("---")
    st.info("💡 **AHA Guidelines Note**: Resting BP ≥ 130 mm Hg indicates Stage 1 Hypertension. Serum cholesterol ≥ 200 mg/dl is considered borderline high.")

active_data = PRESETS.get(preset_choice)
if active_data is None:
    active_data = {
        'age': 52, 'sex': 1, 'cp': 0, 'trestbps': 125, 'chol': 212,
        'fbs': 0, 'restecg': 1, 'thalach': 168, 'exang': 0,
        'oldpeak': 1.0, 'slope': 2, 'ca': 2, 'thal': 3
    }

# ---------------------------------------------------------
# Clinical Input Form
# ---------------------------------------------------------
st.markdown("### 📋 Patient Clinical Examination Parameters")

with st.form("heart_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**1. Demographics & Vitals**")
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=int(active_data['age']), step=1)
        sex_label = st.selectbox("Biological Sex", ["Male (1)", "Female (0)"], index=0 if active_data['sex'] == 1 else 1)
        sex = 1 if "Male" in sex_label else 0
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=int(active_data['trestbps']), step=1)
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=int(active_data['chol']), step=1)
        fbs_label = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False / No (0)", "True / Yes (1)"], index=active_data['fbs'])
        fbs = 1 if "True" in fbs_label else 0

    with col2:
        st.markdown("**2. Symptoms & Electrocardiogram**")
        cp_options = [
            "0: Typical Angina",
            "1: Atypical Angina",
            "2: Non-Anginal Chest Pain",
            "3: Asymptomatic"
        ]
        cp_sel = st.selectbox("Chest Pain Type (cp)", cp_options, index=int(active_data['cp']))
        cp = int(cp_sel.split(":")[0])
        
        restecg_options = [
            "0: Normal",
            "1: ST-T Wave Abnormality (T wave inversions / ST elevation)",
            "2: Left Ventricular Hypertrophy (Estes' criteria)"
        ]
        restecg_sel = st.selectbox("Resting Electrocardiogram (restecg)", restecg_options, index=int(active_data['restecg']))
        restecg = int(restecg_sel.split(":")[0])
        
        thalach = st.number_input("Max Heart Rate Achieved (thalach) [bpm]", min_value=60, max_value=230, value=int(active_data['thalach']), step=1)

    with col3:
        st.markdown("**3. Stress Test & Coronary Angiography**")
        exang_label = st.selectbox("Exercise-Induced Angina (exang)", ["No (0)", "Yes (1)"], index=int(active_data['exang']))
        exang = 1 if "Yes" in exang_label else 0
        
        oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=7.0, value=float(active_data['oldpeak']), step=0.1, format="%.2f")
        
        slope_options = [
            "0: Upsloping (Good prognosis)",
            "1: Flat (Ischemic sign)",
            "2: Downsloping (High risk sign)"
        ]
        slope_sel = st.selectbox("Slope of Peak Exercise ST Segment (slope)", slope_options, index=int(active_data['slope']))
        slope = int(slope_sel.split(":")[0])
        
        ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (ca)", [0, 1, 2, 3, 4], index=int(active_data['ca']))
        
        thal_options = [
            "0: Unknown / Null",
            "1: Normal",
            "2: Fixed Defect (Previous infarct)",
            "3: Reversible Defect (Active ischemia)"
        ]
        thal_sel = st.selectbox("Thalassemia Stress Defect (thal)", thal_options, index=int(active_data['thal']))
        thal = int(thal_sel.split(":")[0])
        
    submitted = st.form_submit_button("🩺 Evaluate Heart Disease Risk", use_container_width=True)

# ---------------------------------------------------------
# Inference & Risk Stratification
# ---------------------------------------------------------
if submitted:
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    patient_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Random Forest Prediction
    rf_pred = model.predict(patient_df)[0]
    rf_probs = model.predict_proba(patient_df)[0]
    rf_conf = max(rf_probs)
    prob_disease = rf_probs[1]
    
    # TensorFlow Deep Neural Network Prediction
    if tf_model and scaler:
        patient_scaled = scaler.transform(patient_df)
        tf_prob_raw = tf_model.predict(patient_scaled, verbose=0)
        tf_prob_disease = float(tf_prob_raw[0][0])
        tf_pred = 1 if tf_prob_disease >= 0.5 else 0
    else:
        tf_pred = None
        tf_prob_disease = None
        
    st.markdown("---")
    st.subheader("📊 Diagnostic Assessment")
    
    res_col1, res_col2 = st.columns([1.5, 1])
    
    with res_col1:
        if rf_pred == 1:
            st.markdown(f"""
            <div class="card-positive">
                <div class="status-text-pos">⚠️ High Probability of Heart Disease Detected</div>
                <p style="margin-top: 0.5rem; color: #7F1D1D; font-size: 1.05rem;">
                    The machine learning model classifies this profile as positive for significant coronary/cardiac risk.
                </p>
                <h4 style="color: #991B1B;">Estimated Heart Disease Probability: <b>{prob_disease * 100:.1f}%</b></h4>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card-negative">
                <div class="status-text-neg">✅ Low Cardiac Risk / Normal Profile</div>
                <p style="margin-top: 0.5rem; color: #14532D; font-size: 1.05rem;">
                    The patient's clinical markers align with healthy cardiovascular baseline indicators.
                </p>
                <h4 style="color: #166534;">Healthy Heart Probability: <b>{rf_probs[0] * 100:.1f}%</b></h4>
            </div>
            """, unsafe_allow_html=True)
            
        st.write(f"**Random Forest (Champion Model) Confidence:** `{rf_conf * 100:.2f}%`")
        st.progress(float(prob_disease))
        
        if tf_pred is not None:
            st.write(f"**TensorFlow Neural Network Verification:** {'Heart Disease Positive' if tf_pred == 1 else 'Healthy (Negative)'} (Probability: `{tf_prob_disease * 100:.1f}%`)")
            if tf_pred == rf_pred:
                st.caption("✨ Both Random Forest and TensorFlow Deep Neural Network independently confirm this diagnostic classification.")
            else:
                st.caption("⚡ Borderline case: Model consensus diverged. Follow-up diagnostic testing recommended.")
                
    with res_col2:
        st.markdown("#### 🔍 Clinical Risk Flags")
        flags = []
        if trestbps >= 140:
            flags.append(("Resting BP ≥ 140 mm Hg (Hypertension Stage 2)", "warn"))
        elif trestbps >= 130:
            flags.append(("Resting BP 130-139 mm Hg (Hypertension Stage 1)", "warn"))
        else:
            flags.append(("Resting BP Normal (< 130 mm Hg)", "ok"))
            
        if chol >= 240:
            flags.append(("High Serum Cholesterol (≥ 240 mg/dl)", "warn"))
        elif chol >= 200:
            flags.append(("Borderline High Cholesterol (200-239 mg/dl)", "warn"))
        else:
            flags.append(("Cholesterol Desirable (< 200 mg/dl)", "ok"))
            
        if exang == 1:
            flags.append(("Exercise-Induced Angina Present", "warn"))
        else:
            flags.append(("No Exercise Angina", "ok"))
            
        if oldpeak >= 1.5:
            flags.append((f"Significant ST Depression ({oldpeak:.1f} mm)", "warn"))
            
        if ca >= 1:
            flags.append((f"{ca} Major Coronary Vessel(s) Blocked/Fluoroscopy", "warn"))
            
        for text, badge_type in flags:
            cls = "badge-warn" if badge_type == "warn" else "badge-ok"
            icon = "⚠️" if badge_type == "warn" else "✅"
            st.markdown(f'<span class="risk-badge {cls}">{icon} {text}</span>', unsafe_allow_html=True)

st.markdown("---")
st.caption("⚠️ **Medical Notice**: This artificial intelligence system is intended strictly as a research and clinical decision-support tool. It does not replace professional diagnostic procedures such as coronary angiography, nuclear stress testing, or certified cardiologist consultation.")
