import streamlit as st
import requests

import sys
try:
    import streamlit as st
except ImportError as e:
    if "altair" in str(e).lower():
        print("Warning: Altair not found, but Streamlit will try to run without it.")
        # On ignore l'erreur et on continue
        import streamlit as st
    else:
        raise e



# Configuration de la page
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style personnalisé
st.markdown("""
    <style>
        .stApp {
            background-color: #f8f9fa;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 24px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 8px;
        }
        .stNumberInput>div>div>input {
            border-radius: 8px;
            border: 1px solid #ddd;
            padding: 10px;
        }
        .stForm {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .prediction-box {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
        }
        .title {
            color: #2c3e50;
            text-align: center;
            font-size: 50px;
            margin-bottom: 20px;
        }
        .subtitle {
            color: #7f8c8d;
            text-align: center;
            font-size: 18px;
            margin-bottom: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# Titre et sous-titre
st.markdown('<p class="title">🩺 Diabetes Prediction App</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter patient details to predict diabetes risk</p>', unsafe_allow_html=True)

# Formulaire dans une carte
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("**Pregnancies**", min_value=0, value=6, help="Number of pregnancies")
        glucose = st.number_input("**Glucose**", min_value=0, value=148, help="Plasma glucose concentration")
        blood_pressure = st.number_input("**Blood Pressure (mm Hg)**", min_value=0, value=72, help="Diastolic blood pressure")
        skin_thickness = st.number_input("**Skin Thickness (mm)**", min_value=0, value=35, help="Triceps skin fold thickness")

    with col2:
        insulin = st.number_input("**Insulin (mu U/ml)**", min_value=0, value=0, help="2-Hour serum insulin")
        bmi = st.number_input("**BMI**", min_value=0.0, value=33.6, help="Body mass index")
        dpf = st.number_input("**Diabetes Pedigree Function**", min_value=0.0, value=0.627, help="Genetic diabetes risk")
        age = st.number_input("**Age**", min_value=0, value=50, help="Age in years")

    submitted = st.form_submit_button("🔮 Predict Diabetes Risk")

# Affichage du résultat
if submitted:
    data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }

    with st.spinner("Predicting..."):
        try:
            response = requests.post(
                "http://localhost:8000/predict",
                json=data
            )
            prediction = response.json()

            # Affichage du résultat dans une carte
            st.markdown("""
                <div class="prediction-box">
                    <h3 style="color: #2c3e50;">Prediction Result</h3>
                    <p style="font-size: 18px; color: #27ae60;"">{}</p>
                </div>
            """.format("✅ **Diabetic**" if prediction.get("prediction", 0) == 1 else "❌ **Not Diabetic**"), unsafe_allow_html=True)

            # Affichage des détails
            st.json({
                "Model Prediction": prediction.get("prediction", "N/A"),
                "Model Probability": prediction.get("probability", "N/A"),
            })
        except Exception as e:
            st.error(f"Error: {e}")
