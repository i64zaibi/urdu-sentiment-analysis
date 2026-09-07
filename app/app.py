import streamlit as st
import torch
import re
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Page configuration
st.set_page_config(
    page_title="Urdu Sentiment Analysis",
    page_icon="🔍",
    layout="centered"
)

# PREPROCESSING

def preprocess_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    text = re.sub(r'@\w+', '@user', text)
    text = re.sub(r'</?[a-zA-Z][^>]*>', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "..", "model")
    model_path = os.path.abspath(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()


st.title("اردو جذبات کا تجزیہ")
st.subheader("Urdu Sentiment Analysis using XLM-RoBERTa")
st.write("Enter Urdu text below and click 'Analyze' to determine its sentiment.")


# TEXT INPUT

user_text = st.text_area(
    "اردو متن درج کریں (Enter Urdu Text):",
    height=150,
    placeholder="یہاں اردو متن لکھیں..."
)

# Analyze button
analyze_clicked = st.button("🔍 Analyze Sentiment", type="primary")

# PREDICTION LOGIC

if analyze_clicked:
    if not user_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing sentiment..."):

            #preprocessing
            cleaned_text = preprocess_text(user_text)

            # Tokenization
            inputs = tokenizer(
                cleaned_text,
                max_length=384,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )

            #prediction
            with torch.no_grad():
                outputs = model(**inputs)

            probs = torch.softmax(outputs.logits, dim=-1)
            predicted_id = torch.argmax(probs, dim=-1).item()
            predicted_label = model.config.id2label[predicted_id]
            confidence = probs[0][predicted_id].item()

            neg_prob = probs[0][0].item()
            pos_prob = probs[0][1].item()

# DISPLAY RESULTS


        if predicted_label == "Positive":
            st.success(f"**Sentiment: {predicted_label}** (Confidence: {confidence:.1%})")
        else:
            st.error(f"**Sentiment: {predicted_label}** (Confidence: {confidence:.1%})")

        st.write("**Class Probabilities:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Negative", f"{neg_prob:.1%}")
        with col2:
            st.metric("Positive", f"{pos_prob:.1%}")

        st.write("Negative:")
        st.progress(neg_prob)
        st.write("Positive:")
        st.progress(pos_prob)


# SAMPLE TEXTS

st.divider()
st.subheader("Sample Texts")
st.write("Click any sample to try it:")

samples = {
    "Positive Example": "یہ فلم بہت شاندار تھی بہترین اداکاری اور دلچسپ کہانی",
    "Negative Example": "بہت بری فلم تھی وقت کا ضیاع ہدایتکاری بہت کمزور تھی",
    "Mixed Example": "فلم کی کہانی اچھی تھی لیکن اداکاری کمزور تھی",
}

for label, sample in samples.items():
    if st.button(label, key=label):
        st.info(f"**Text:** {sample}")
        st.write("Copy this text and paste it in the input box above, then click Analyze.")


# MODEL INFORMATION

st.divider()
st.subheader("Model Information")
st.write("**Model:** XLM-RoBERTa-base")
st.write("**Task:** Binary Sentiment Classification (Positive / Negative)")
st.write("**Dataset:** Urdu Sentiment Analysis (Khan et al., 2020)")
st.write("**Test Accuracy:** 82.9%")
st.write("**Test F1-Score:** 0.829")
st.caption("Note: The confidence score represents model probability, not a guarantee of correctness.")