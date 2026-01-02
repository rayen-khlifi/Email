import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Titre de l'application ---
st.title("Détecteur de Spam SMS 📩")

# --- Charger le modèle et le vecteur ---
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf.pkl")

# --- Entrée utilisateur ---
user_input = st.text_area("Entrez votre SMS ici :")

if st.button("Vérifier"):

    if user_input.strip() == "":
        st.warning("Veuillez entrer un message.")
    else:
        # Transformer le texte
        input_vec = vectorizer.transform([user_input])
        
        # Prédiction
        pred = model.predict(input_vec)[0]
        if pred == 1:
            st.error("⚠️ C’est un SPAM !")
        else:
            st.success("✅ C’est un message normal (HAM).")
