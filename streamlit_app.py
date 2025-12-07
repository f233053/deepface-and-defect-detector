import streamlit as st
import numpy as np
import pandas as pd
import librosa
import pickle
from io import BytesIO
import joblib


# Page config
st.set_page_config(
    page_title="ML Classification Suite",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton>button { width: 100%; }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_audio_models():
    try:
        # FIXES: Using joblib.load() and correcting the typo in 'audio_perceptron.joblib'
        scaler = joblib.load('audio_scaler.joblib')
        svm = joblib.load('audio_svm.joblib')
        lr = joblib.load('audio_lr.joblib')
        
        # Corrected typo: 'audio_perceptron,joblib' -> 'audio_perceptron.joblib'
        perceptron = joblib.load('audio_perceptron.joblib')
        
        # Assuming audio_mlp.joblib is the DNN model
        dnn = joblib.load('audio_mlp.joblib')
        
        return scaler, {'SVM': svm, 'Logistic Regression': lr, 'Perceptron': perceptron, 'DNN': dnn}
    except Exception as e:
        st.error(f"Error loading audio models: {e}")
        return None, None

@st.cache_resource
def load_defect_models():
    try:
        # Load TF-IDF vectorizer (20 features)
        tfidf = joblib.load('tfidf.pkl')

        svm = joblib.load('svm.pkl')
        lr = joblib.load('logreg.pkl')
        perceptron = joblib.load('perceptron.pkl')
        MLP = joblib.load('mlp.pkl')

        return tfidf, {
            'SVM': svm, 
            'Logistic Regression': lr, 
            'Perceptron': perceptron,
            'MLP Neural Network': MLP
        }

    except Exception as e:
        st.error(f"Error loading defect models: {e}")
        return None, None
    
    
# Feature extraction
def extract_audio_features(audio_array, sr=16000):
    try:
        # Pad or truncate to 3 seconds
        target_length = sr * 3
        if len(audio_array) > target_length:
            audio_array = audio_array[:target_length]
        else:
            audio_array = np.pad(audio_array, (0, target_length - len(audio_array)), mode='constant')
        
        # MFCC
        mfccs = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        
        # Mel spectrogram
        mel = librosa.feature.melspectrogram(y=audio_array, sr=sr, n_mels=40)
        mel_mean = np.mean(mel, axis=1)
        
        # Chroma
        chroma = librosa.feature.chroma_stft(y=audio_array, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=audio_array, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_array)
        zcr_mean = np.mean(zcr)
        
        features = np.concatenate([mfccs_mean, mfccs_std, mel_mean, chroma_mean, contrast_mean, [zcr_mean]])
        return features
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

# Main app
def main():
    st.title("🤖 Machine Learning Classification Suite")
    st.markdown("### Choose a task below")
    
    # Task selection
    task = st.radio("Select Task:", ["Audio Deepfake Detection", "Software Defect Prediction"], horizontal=True)
    
    st.markdown("---")
    
    if task == "Audio Deepfake Detection":
        audio_classification_app()
    else:
        defect_prediction_app()

def audio_classification_app():
    st.header("🎵 Urdu Deepfake Audio Detection")
    
    # Load models
    scaler, models = load_audio_models()
    
    if models is None:
        st.error("⚠️ Models not found. Please train the models first.")
        return
    
    # Model selection
    model_choice = st.selectbox("Select Model:", list(models.keys()))
    
    # File upload
    audio_file = st.file_uploader("Upload Audio File (WAV, MP3)", type=['wav', 'mp3'])
    
    if audio_file is not None:
        st.audio(audio_file)
        
        if st.button("🔍 Analyze Audio", type="primary"):
            with st.spinner("Processing audio..."):
                try:
                    # Load audio
                    audio_bytes = audio_file.read()
                    audio_array, sr = librosa.load(BytesIO(audio_bytes), sr=16000)
                    
                    # Extract features
                    features = extract_audio_features(audio_array, sr)
                    
                    if features is not None:
                        features_scaled = scaler.transform(features.reshape(1, -1))
                        
                        # Predict
                        model = models[model_choice]
                        prediction = model.predict(features_scaled)[0]
                        
                        # Get probabilities
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(features_scaled)[0]
                        else:
                            proba = [0.5, 0.5]
                        
                        # Display results
                        st.markdown("### 📊 Prediction Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Classification", "🟢 Bonafide" if prediction == 0 else "🔴 Deepfake")
                        
                        with col2:
                            confidence = proba[prediction] * 100
                            st.metric("Confidence", f"{confidence:.2f}%")
                        
                        # Probability scores
                        st.markdown("### 📈 Probability Scores")
                        prob_df = pd.DataFrame({
                            'Class': ['Bonafide', 'Deepfake'],
                            'Probability': [f"{proba[0]*100:.2f}%", f"{proba[1]*100:.2f}%"]
                        })
                        st.dataframe(prob_df, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error processing audio: {e}")

def defect_prediction_app():
    st.header("🐛 Multi-Label Software Defect Prediction")
    
    # Load models
    # 🚨 FIX 1: Rename the variables to acknowledge the scaler is gone/None
    # We now expect 'None' for the first return value.
    _scaler, models = load_defect_models() 
    
    if models is None:
        st.error("⚠️ Models not found. Please train the models first.")
        return
    
    # Model selection
    model_choice = st.selectbox("Select Model:", list(models.keys()))
    
    # Input method selection
    input_method = st.radio("Input Method:", ["Upload File", "Manual Entry"], horizontal=True)
    
    if input_method == "Upload File":
        uploaded_file = st.file_uploader("Upload CSV or XLSX file", type=['csv', 'xlsx'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                else:
                    df = pd.read_csv(uploaded_file)
                
                st.write("### 📄 Uploaded Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                if st.button("🔍 Predict Defects", type="primary"):
                    with st.spinner("Making predictions..."):
                        try:
                            X = df.values
                            
                            # 🚨 FIX 2 (Upload): Removed the scaling line. Use X directly.
                            X_processed = X
                            
                            model = models[model_choice]
                            predictions = model.predict(X_processed) # Use X_processed
                            
                            # Get probabilities (Logic remains correct for multi-label)
                            if hasattr(model, 'predict_proba'):
                                if model_choice in ['SVM', 'Logistic Regression']:
                                    # Note: This list comprehension assumes OneVsRestClassifier structure
                                    probas = np.array([est.predict_proba(X_processed)[:, 1] for est in model.estimators_]).T
                                else:
                                    probas = model.predict_proba(X_processed)
                            else:
                                probas = predictions.astype(float)
                            
                            # ... (rest of the upload file logic) ...
                            
                        except Exception as e:
                            st.error(f"Error making predictions: {e}")
                            
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    else:  # Manual Entry
        st.markdown("### ✍️ Enter text to classify")

        user_text = st.text_area(
            "Enter the software requirement / description:",
            height=200
        )

        if st.button("🔍 Predict Defects", type="primary"):
            try:
                if not user_text.strip():
                    st.warning("Please enter some text.")
                    return

                # Load TF-IDF vectorizer
                tfidf, models = load_defect_models()

                # Convert text → TF-IDF (20 features)
                X_processed = tfidf.transform([user_text]).toarray()

                model = models[model_choice]

                # Predict label vector
                prediction = model.predict(X_processed)[0]

                # Predict probabilities (if available)
                if hasattr(model, 'predict_proba'):
                    if model_choice in ['SVM', 'Logistic Regression']:
                        proba = np.array(
                            [est.predict_proba(X_processed)[0, 1] for est in model.estimators_]
                        )
                    else:
                        proba = model.predict_proba(X_processed)[0]
                else:
                    proba = prediction.astype(float)

                # Display predictions
                st.markdown("### 📊 Prediction Output")

                results = []
                label_names = [
                    "type_blocker",
                    "type_regression",
                    "type_bug",
                    "type_documentation",
                    "type_enhancement",
                    "type_task",
                    "type_dependency_upgrade"
                ]

                for idx, (pred, p) in enumerate(zip(prediction, proba)):
                    results.append({
                        "Label": label_names[idx],

                        "Predicted": "Yes" if pred == 1 else "No",
                        "Confidence": f"{p*100:.2f}%"
                    })

                st.dataframe(pd.DataFrame(results), use_container_width=True)

            except Exception as e:
                st.error(f"Error making prediction: {e}")


# Footer
def add_footer():
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>Built with Streamlit | Models: SVM, Logistic Regression, Perceptron, DNN</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    add_footer()
