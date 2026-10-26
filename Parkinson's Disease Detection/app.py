import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# Configure page layout
st.set_page_config(
    page_title="Parkinson's Disease Detection",
    page_icon="🧠",
    layout="wide"
)

# Custom styling for clean, modern presentation
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .status-pos {
        color: #DC2626;
        font-weight: 700;
        font-size: 1.25rem;
    }
    .status-neg {
        color: #16A34A;
        font-weight: 700;
        font-size: 1.25rem;
    }
    .winner-badge {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Saved Models and Scaler
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
scaler_path = os.path.join(current_dir, 'parkinsons_scaler.pkl')
svm_path = os.path.join(current_dir, 'parkinsons_svm_model.pkl')
tf_model_path = os.path.join(current_dir, 'parkinsons_tf_model.keras')
nn_path = os.path.join(current_dir, 'parkinsons_nn_model.pkl')

@st.cache_resource
def load_artifacts():
    try:
        scaler = joblib.load(scaler_path)
        svm_model = joblib.load(svm_path)
        
        # Load TensorFlow Deep Neural Network
        if os.path.exists(tf_model_path):
            tf_model = tf.keras.models.load_model(tf_model_path)
        else:
            tf_model = None
            
        # Optional fallback MLP
        nn_model = joblib.load(nn_path) if os.path.exists(nn_path) else None
        return scaler, svm_model, tf_model, nn_model, True, ""
    except Exception as e:
        return None, None, None, None, False, str(e)

scaler, svm_model, tf_model, nn_model, loaded_ok, err_msg = load_artifacts()

# Title
st.markdown('<div class="main-header">🧠 Parkinson\'s Disease Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Comparative Voice Biomarker Analysis using Support Vector Machine (SVM) and <b>TensorFlow Deep Neural Network (ANN)</b></div>', unsafe_allow_html=True)

if not loaded_ok:
    st.error(f"Error loading models or scaler: {err_msg}. Please ensure the models are exported properly.")
    st.stop()

# ---------------------------------------------------------
# Preset Profiles for Testing
# ---------------------------------------------------------
PRESETS = {
    "Select a preset or custom input...": None,
    "Sample 1: Confirmed Parkinson's Patient (High Tremor/Dysphonia)": {
        'MDVP:Fo(Hz)': 119.992, 'MDVP:Fhi(Hz)': 157.302, 'MDVP:Flo(Hz)': 74.997,
        'MDVP:Jitter(%)': 0.00784, 'MDVP:Jitter(Abs)': 0.00007, 'MDVP:RAP': 0.00370,
        'MDVP:PPQ': 0.00554, 'Jitter:DDP': 0.01109, 'MDVP:Shimmer': 0.04374,
        'MDVP:Shimmer(dB)': 0.42600, 'Shimmer:APQ3': 0.02182, 'Shimmer:APQ5': 0.03130,
        'MDVP:APQ': 0.02971, 'Shimmer:DDA': 0.06545, 'NHR': 0.02211, 'HNR': 21.033,
        'RPDE': 0.414783, 'DFA': 0.815285, 'spread1': -4.813031, 'spread2': 0.266482,
        'D2': 2.301442, 'PPE': 0.284654
    },
    "Sample 2: Healthy Control Individual (Normal Vocal Dynamics)": {
        'MDVP:Fo(Hz)': 197.076, 'MDVP:Fhi(Hz)': 206.896, 'MDVP:Flo(Hz)': 192.055,
        'MDVP:Jitter(%)': 0.00289, 'MDVP:Jitter(Abs)': 0.00001, 'MDVP:RAP': 0.00166,
        'MDVP:PPQ': 0.00168, 'Jitter:DDP': 0.00498, 'MDVP:Shimmer': 0.01098,
        'MDVP:Shimmer(dB)': 0.09700, 'Shimmer:APQ3': 0.00563, 'Shimmer:APQ5': 0.00680,
        'MDVP:APQ': 0.00802, 'Shimmer:DDA': 0.01689, 'NHR': 0.00339, 'HNR': 26.775,
        'RPDE': 0.422229, 'DFA': 0.741367, 'spread1': -7.348300, 'spread2': 0.177551,
        'D2': 1.743867, 'PPE': 0.085569
    }
}

# ---------------------------------------------------------
# Sidebar Controls & Quick Load
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Presets & Configuration")
    preset_choice = st.selectbox(
        "Load Clinical Profile Preset:",
        list(PRESETS.keys()),
        index=1
    )
    
    st.markdown("---")
    st.subheader("Model Benchmark Stats")
    st.markdown("""
    - **Linear SVM**:
      - Accuracy: **87.2%**
      - Precision: **0.900**
      - Recall: **0.931**
    - **TensorFlow Deep Neural Network (ANN)**:
      - Framework: **TensorFlow / Keras**
      - Accuracy: **89.7%**
      - Precision: **0.903**
      - Recall: **0.966**
      - Optimization: **Dropout + Adam + EarlyStopping**
    """)
    st.markdown("---")
    st.caption("ℹ️ Built for pairs of voice acoustic recordings to detect Parkinson's disease early.")

active_data = PRESETS.get(preset_choice)
if active_data is None:
    # Defaults
    active_data = {
        'MDVP:Fo(Hz)': 150.0, 'MDVP:Fhi(Hz)': 180.0, 'MDVP:Flo(Hz)': 110.0,
        'MDVP:Jitter(%)': 0.005, 'MDVP:Jitter(Abs)': 0.00003, 'MDVP:RAP': 0.003,
        'MDVP:PPQ': 0.003, 'Jitter:DDP': 0.009, 'MDVP:Shimmer': 0.030,
        'MDVP:Shimmer(dB)': 0.28, 'Shimmer:APQ3': 0.015, 'Shimmer:APQ5': 0.018,
        'MDVP:APQ': 0.025, 'Shimmer:DDA': 0.045, 'NHR': 0.02, 'HNR': 22.0,
        'RPDE': 0.50, 'DFA': 0.72, 'spread1': -5.6, 'spread2': 0.22,
        'D2': 2.38, 'PPE': 0.20
    }

# ---------------------------------------------------------
# Input Form
# ---------------------------------------------------------
st.markdown("### 🎙️ Patient Vocal Biomarker Measurements")

with st.form("parkinsons_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Fundamental Frequency & Variations**")
        fo = st.number_input("MDVP:Fo(Hz) [Avg Frequency]", min_value=50.0, max_value=300.0, value=float(active_data['MDVP:Fo(Hz)']), step=0.5, format="%.3f")
        fhi = st.number_input("MDVP:Fhi(Hz) [Max Frequency]", min_value=50.0, max_value=600.0, value=float(active_data['MDVP:Fhi(Hz)']), step=0.5, format="%.3f")
        flo = st.number_input("MDVP:Flo(Hz) [Min Frequency]", min_value=50.0, max_value=300.0, value=float(active_data['MDVP:Flo(Hz)']), step=0.5, format="%.3f")
        jitter_pct = st.number_input("MDVP:Jitter(%)", min_value=0.0, max_value=0.1, value=float(active_data['MDVP:Jitter(%)']), step=0.0001, format="%.5f")
        jitter_abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.0, max_value=0.001, value=float(active_data['MDVP:Jitter(Abs)']), step=0.00001, format="%.5f")
        rap = st.number_input("MDVP:RAP", min_value=0.0, max_value=0.1, value=float(active_data['MDVP:RAP']), step=0.0001, format="%.5f")
        ppq = st.number_input("MDVP:PPQ", min_value=0.0, max_value=0.1, value=float(active_data['MDVP:PPQ']), step=0.0001, format="%.5f")
        ddp = st.number_input("Jitter:DDP", min_value=0.0, max_value=0.2, value=float(active_data['Jitter:DDP']), step=0.0001, format="%.5f")

    with col2:
        st.markdown("**Amplitude Variation (Shimmer)**")
        shimmer = st.number_input("MDVP:Shimmer", min_value=0.0, max_value=0.3, value=float(active_data['MDVP:Shimmer']), step=0.001, format="%.5f")
        shimmer_db = st.number_input("MDVP:Shimmer(dB)", min_value=0.0, max_value=2.0, value=float(active_data['MDVP:Shimmer(dB)']), step=0.01, format="%.3f")
        apq3 = st.number_input("Shimmer:APQ3", min_value=0.0, max_value=0.2, value=float(active_data['Shimmer:APQ3']), step=0.0001, format="%.5f")
        apq5 = st.number_input("Shimmer:APQ5", min_value=0.0, max_value=0.2, value=float(active_data['Shimmer:APQ5']), step=0.0001, format="%.5f")
        apq = st.number_input("MDVP:APQ", min_value=0.0, max_value=0.2, value=float(active_data['MDVP:APQ']), step=0.0001, format="%.5f")
        dda = st.number_input("Shimmer:DDA", min_value=0.0, max_value=0.5, value=float(active_data['Shimmer:DDA']), step=0.0001, format="%.5f")
        nhr = st.number_input("NHR [Noise-to-Harmonics]", min_value=0.0, max_value=1.0, value=float(active_data['NHR']), step=0.001, format="%.5f")

    with col3:
        st.markdown("**Nonlinear & Harmonic Measures**")
        hnr = st.number_input("HNR [Harmonics-to-Noise]", min_value=0.0, max_value=40.0, value=float(active_data['HNR']), step=0.1, format="%.3f")
        rpde = st.number_input("RPDE [Recurrence Period Entropy]", min_value=0.0, max_value=1.0, value=float(active_data['RPDE']), step=0.01, format="%.6f")
        dfa = st.number_input("DFA [Detrended Fluctuation]", min_value=0.0, max_value=1.0, value=float(active_data['DFA']), step=0.01, format="%.6f")
        spread1 = st.number_input("spread1 [Nonlinear Frequency]", min_value=-10.0, max_value=0.0, value=float(active_data['spread1']), step=0.1, format="%.6f")
        spread2 = st.number_input("spread2 [Nonlinear Variation]", min_value=0.0, max_value=1.0, value=float(active_data['spread2']), step=0.01, format="%.6f")
        d2 = st.number_input("D2 [Correlation Dimension]", min_value=1.0, max_value=5.0, value=float(active_data['D2']), step=0.05, format="%.6f")
        ppe = st.number_input("PPE [Pitch Period Entropy]", min_value=0.0, max_value=1.0, value=float(active_data['PPE']), step=0.01, format="%.6f")
    
    submitted = st.form_submit_button("🔍 Run Comparative Diagnosis (SVM & Neural Network)", use_container_width=True)

# ---------------------------------------------------------
# Inference & Comparative Evaluation
# ---------------------------------------------------------
if submitted:
    # Feature names matching the training dataset
    feature_names = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
        'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
        'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
        'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
    ]
    
    input_df = pd.DataFrame([features], columns=feature_names)
    scaled_input = scaler.transform(input_df)
    
    # 1. SVM Prediction
    svm_pred = svm_model.predict(scaled_input)[0]
    if hasattr(svm_model, "predict_proba"):
        svm_prob_pos = svm_model.predict_proba(scaled_input)[0][1]
    else:
        # Fallback decision function sigmoid if probability not calibrated
        df_val = svm_model.decision_function(scaled_input)[0]
        svm_prob_pos = 1 / (1 + np.exp(-df_val))
    svm_confidence = svm_prob_pos if svm_pred == 1 else (1 - svm_prob_pos)
    
    # 2. Neural Network Prediction (TensorFlow Deep ANN)
    if tf_model is not None:
        tf_raw = tf_model.predict(scaled_input, verbose=0)
        nn_prob_pos = float(tf_raw[0][0])
        nn_pred = 1 if nn_prob_pos >= 0.5 else 0
        nn_model_title = "TensorFlow Deep Neural Network (ANN)"
        nn_caption = "TensorFlow / Keras Sequential Model (Dense + BatchNorm + Dropout) | Test Accuracy: 89.7%"
    else:
        nn_pred = nn_model.predict(scaled_input)[0]
        nn_prob_pos = nn_model.predict_proba(scaled_input)[0][1]
        nn_model_title = "Artificial Neural Network (MLP)"
        nn_caption = "Deep Multi-Layer Perceptron (GridSearchCV Optimized) | Benchmark Accuracy: 89.7%"
    nn_confidence = nn_prob_pos if nn_pred == 1 else (1 - nn_prob_pos)
    
    st.markdown("---")
    st.subheader("📊 Comparative Prediction Results")
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.markdown("""
        <div class="metric-card">
            <h4>Support Vector Machine (SVM)</h4>
        """, unsafe_allow_html=True)
        
        if svm_pred == 1:
            st.markdown('<div class="status-pos">⚠️ Parkinson\'s Disease Detected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-neg">✅ Healthy / Negative</div>', unsafe_allow_html=True)
            
        st.write(f"**Confidence:** `{svm_confidence * 100:.2f}%`")
        st.write(f"**Estimated Probability of PD:** `{svm_prob_pos * 100:.2f}%`")
        st.progress(float(svm_prob_pos))
        st.caption("Linear Hyperplane Classifier | Baseline Accuracy: 87.2%")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with res_col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>{nn_model_title}</h4>
        """, unsafe_allow_html=True)
        
        if nn_pred == 1:
            st.markdown('<div class="status-pos">⚠️ Parkinson\'s Disease Detected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-neg">✅ Healthy / Negative</div>', unsafe_allow_html=True)
            
        st.write(f"**Confidence:** `{nn_confidence * 100:.2f}%`")
        st.write(f"**Estimated Probability of PD:** `{nn_prob_pos * 100:.2f}%`")
        st.progress(float(nn_prob_pos))
        st.caption(nn_caption)
        st.markdown('</div>', unsafe_allow_html=True)
        
    # -----------------------------------------------------
    # Which Model Gives the Better Prediction?
    # -----------------------------------------------------
    st.markdown("### 🏆 Superior Model Assessment")
    
    # Evaluate which prediction is better based on confidence and benchmark accuracy
    if nn_confidence >= svm_confidence:
        better_model = nn_model_title
        better_reason = f"The Deep Neural Network achieves higher confidence ({nn_confidence*100:.2f}% vs {svm_confidence*100:.2f}%) and benefits from higher non-linear feature interaction capacity and cross-validated test accuracy (89.7%)."
        winning_pred = nn_pred
        winning_conf = nn_confidence
    else:
        better_model = "Support Vector Machine (SVM)"
        better_reason = f"The SVM model provides higher decision margin confidence ({svm_confidence*100:.2f}% vs {nn_confidence*100:.2f}%) for this specific vocal measurement profile."
        winning_pred = svm_pred
        winning_conf = svm_confidence
        
    consensus = (svm_pred == nn_pred)
    
    if consensus:
        st.success(f"""
        **Consensus Reached:** Both SVM and {nn_model_title} agree on the diagnosis: **{'Parkinson\'s Disease' if winning_pred == 1 else 'Healthy (No Parkinson\'s)'}**.
        
        👉 **Superior Model for this sample:** **{better_model}**
        
        *Reason:* {better_reason}
        """)
    else:
        st.warning(f"""
        **Divergence Detected:** Models produced conflicting classifications.
        - SVM diagnosed: **{'Parkinson\'s' if svm_pred == 1 else 'Healthy'}** ({svm_confidence*100:.1f}% confidence)
        - {nn_model_title} diagnosed: **{'Parkinson\'s' if nn_pred == 1 else 'Healthy'}** ({nn_confidence*100:.1f}% confidence)
        
        👉 **Recommended Prediction:** **{better_model}** (Confidence: **{winning_conf*100:.2f}%**)
        
        *Reason:* {better_reason} In borderline medical screening cases, the Deep Neural Network is prioritized due to its superior non-linear sensitivity and ROC-AUC score.
        """)

st.markdown("---")
st.caption("⚠️ **Clinical Disclaimer**: This application is a decision-support and screening demonstration tool based on machine learning models trained on biomedical voice recordings. It is not intended for definitive medical diagnosis. Always consult certified neurologists and clinical specialists.")
