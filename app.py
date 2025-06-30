# app.py (The Definitive, Working Streamlit Version)

import os
import pandas as pd
import numpy as np
import google.generativeai as genai
import streamlit as st
from PIL import Image
from io import BytesIO

# --- Page Configuration ---
# This sets the title and icon that appear in the browser tab. It should be the first Streamlit command.
st.set_page_config(
    page_title="AI Teacher Portal",
    page_icon="🤖",
    layout="wide"
)

# --- Authentication and Data Loading (run once and cached) ---
# The @st.cache_resource decorator is a powerful Streamlit feature. It ensures this
# complex setup code runs only once when the app first starts, not every time a user interacts.
@st.cache_resource
def load_resources():
    """
    This function configures the Google AI API and loads the knowledge base from the CSV file.
    If anything fails here, it will display an error in the app.
    """
    # Configure Google AI API
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        # st.error shows a prominent error message in the UI.
        st.error("GOOGLE_API_KEY secret not found. Please set it in your Hugging Face Space secrets.")
        return None, None
    try:
        genai.configure(api_key=api_key)
        print("✅ Google AI API configured successfully.")
    except Exception as e:
        st.error(f"Error configuring Google AI API: {e}")
        return None, None

    # Load the knowledge base from the CSV file.
    try:
        dataframe = pd.read_csv("knowledge_base.csv")
        # The embedding column is read as a string, so we must convert it back to a list of floats.
        dataframe['embedding'] = dataframe['embedding'].apply(eval)
        print("✅ Knowledge base loaded successfully.")
        # Return both the generative model and the dataframe so the app can use them.
        return genai.GenerativeModel('gemini-1.5-flash-latest'), dataframe
    except FileNotFoundError:
        st.error("CRITICAL ERROR: 'knowledge_base.csv' not found. The app cannot function.")
        return None, None

# Load the resources when the app starts.
gen_model, df_embedded = load_resources()

# --- Core Logic Functions ---
def find_relevant_passage(query, dataframe):
    """Finds the most relevant text chunk from the knowledge base."""
    query_embedding_response = genai.embed_content(model='models/text-embedding-004', content=query, task_type="RETRIEVAL_QUERY")
    query_embedding = query_embedding_response['embedding']
    dot_products = np.dot(np.stack(dataframe['embedding']), query_embedding)
    best_passage_index = np.argmax(dot_products)
    return dataframe.iloc[best_passage_index]

def answer_question_from_text(question, dataframe):
    """Handles the text-based Q&A logic."""
    relevant_page = find_relevant_passage(question, dataframe)
    prompt = f"Answer the following question based ONLY on the provided source material.\n\nQuestion: {question}\n\nSource Material:\n{relevant_page['text_for_search']}"
    response = gen_model.generate_content(prompt)
    return response.text

def analyze_handwritten_image(image, instruction):
    """Handles the image analysis logic."""
    # For image analysis, we use the powerful gemini-pro-vision model.
    vision_model = genai.GenerativeModel('gemini-pro-vision')
    prompt = f"You are an expert AI Teacher. Analyze the handwritten work in the image based on this instruction: \"{instruction}\""
    response = vision_model.generate_content([prompt, image])
    return response.text

# --- Streamlit User Interface ---
# The code below creates the visual elements of your web app.

st.title("🤖 AI Teacher Portal")

# Use the sidebar for navigation between our two features.
with st.sidebar:
    st.header("App Mode")
    app_mode = st.radio("Choose a feature:", ("❓ Textbook Q&A", "✍️ Homework Helper"))

# --- Textbook Q&A Mode ---
if app_mode == "❓ Textbook Q&A":
    st.header("Ask a Question from the Textbook")
    
    text_question = st.text_area("Enter your question here:", height=150)
    
    if st.button("Get Answer"):
        # --- THIS IS THE CORRECTED CHECK ---
        # We now check if the df_embedded DataFrame is valid before proceeding.
        if df_embedded is None or df_embedded.empty:
            st.error("Knowledge base is not loaded. Cannot answer questions.")
        elif text_question:
            # st.spinner shows a nice "working" message while the AI is thinking.
            with st.spinner("The AI Teacher is thinking..."):
                response = answer_question_from_text(text_question, df_embedded)
                st.success("Here is your answer:")
                st.write(response)
        else:
            st.warning("Please enter a question.")

# --- Homework Helper Mode ---
elif app_mode == "✍️ Homework Helper":
    st.header("Get Help with Your Homework")
    
    # st.file_uploader creates the interactive file upload widget.
    uploaded_file = st.file_uploader("Upload an image of your work", type=["png", "jpg", "jpeg"])
    instruction = st.text_input("What should I do with this image?", placeholder="e.g., 'Solve for x' or 'Check my work'")
    
    if st.button("Analyze Image"):
        if uploaded_file is not None:
            if instruction:
                with st.spinner("The AI Teacher is analyzing your image..."):
                    image = Image.open(uploaded_file)
                    response = analyze_handwritten_image(image, instruction)
                    st.success("Here is the analysis:")
                    st.write(response)
            else:
                st.warning("Please provide an instruction for the image.")
        else:
            st.warning("Please upload an image.")
