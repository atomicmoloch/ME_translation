import streamlit as st
import time
from transformers import pipeline

st.title("Enhanced Multilingual Translator Chatbot")

# Choose the translation models from Hugging Face
translation_models = {
    "English": "en",
    "German": "de",
    "French": "fr",
    "Urdu": "ur",
    "Spanish": "es",
    "Chinese": "zh",
    "Russian"
    # Add more language pairs as needed
}

trans_input = st.selectbox("Select input language", list(translation_models.keys()), key=1)
trans_output = st.selectbox("Select output language", list(translation_models.keys()), key=2)

while trans_input == trans_output:
    time.sleep(1)

selected_translation = "Helsinki-NLP/opus-mt-"+ translation_models[trans_input] + "-" + translation_models[trans_output]

# Load the translation pipeline
translator = pipeline(task="translation", model=selected_translation)

# User input for translation
user_input = st.text_area("Enter text for translation:", "")

# Display loading indicator
if st.button("Translate"):
    with st.spinner("Translating..."):
        # Simulate translation delay for demonstration
        time.sleep(2)
        if user_input:
            # Perform translation
            translated_text = translator(user_input, max_length=500)[0]['translation_text']
            st.success(f"Translated Text: {translated_text}")
        else:
            st.warning("Please enter text for translation.")

# Clear button to reset input and result
if st.button("Clear"):
    user_input = ""
    st.success("Input cleared.")
    st.empty()  # Clear previous results if any

st.markdown("---")

st.subheader("About")
st.write(
    "This is an enhanced Multilingual Translator chatbot that uses the Hugging Face Transformers library."
)
st.write(
    "Select a translation model from the dropdown, enter text, and click 'Translate' to see the translation."
)
