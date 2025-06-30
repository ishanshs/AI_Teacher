# app.py (The Definitive Version using a Single Gemini 1.5 Flash Model)

import os
import pandas as pd
import numpy as np
import google.generativeai as genai
import streamlit as st
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Teacher Portal",
    page_icon="🤖",
    layout="wide"
)

# --- Authentication and Resource Loading (run once and cached) ---
@st.cache_resource
def load_resources():
    """
    This function configures the API, loads the single Gemini 1.5 Flash model,
    and loads the knowledge base from the CSV file.
    """
    # Configure Google AI API
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY secret not found. Please set it in your Hugging Face Space secrets.")
        return None, None
    try:
        genai.configure(api_key=api_key)
        print("✅ Google AI API configured successfully.")
    except Exception as e:
        st.error(f"Error configuring Google AI API: {e}")
        return None, None

    # Load the single, powerful Gemini 1.5 Flash model for all tasks.
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    print("✅ Gemini 1.5 Flash model loaded successfully.")

    # Load the knowledge base
    try:
        dataframe = pd.read_csv("knowledge_base.csv")
        dataframe['embedding'] = dataframe['embedding'].apply(eval)
        print("✅ Knowledge base loaded successfully.")
        # Return the model and the dataframe.
        return model, dataframe
    except FileNotFoundError:
        st.error("CRITICAL ERROR: 'knowledge_base.csv' not found. App cannot function.")
        return None, None

# Load the resources when the app starts.
model, df_embedded = load_resources()

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
    # Use the pre-loaded Flash model for fast text responses
    response = model.generate_content(prompt)
    return response.text

def analyze_handwritten_image(image, instruction):
    """Handles the image analysis logic using the pre-loaded Flash model."""
    if image is None:
        return "Please upload an image."
    if not instruction:
        return "Please provide an instruction for the image."
            
    print(f"Received image and instruction: {instruction}")
    # --- THIS IS THE CORRECTED LOGIC ---
    # We use the same 'gemini-1.5-flash-latest' model that we loaded at the start.
    # It is multimodal and can handle both text and images.
    prompt = f"You are an expert AI Teacher. Analyze the handwritten work in the image based on this instruction: \"{instruction}\""
    response = model.generate_content([prompt, image])
    return response.text

# --- Streamlit User Interface ---
st.title("🤖 AI Teacher Portal")

# Use the sidebar for navigation between our two features.
with st.sidebar:
    st.header("App Mode")
    app_mode = st.radio("Choose a feature:", ("❓ Textbook Q&A", "✍️ Homework Helper"))

# --- Textbook Q&A Mode ---
if app_mode == "❓ Textbook Q&A":
    st.header("Ask a Question from the Textbook")

    # Check if resources were loaded correctly before showing the UI
    if model and df_embedded is not None and not df_embedded.empty:
        text_question = st.text_area("Enter your question here:", height=150)
        
        if st.button("Get Answer"):
            if text_question:
                with st.spinner("The AI Teacher is thinking..."):
                    response = answer_question_from_text(text_question, df_embedded)
                    st.success("Here is your answer:")
                    st.write(response)
            else:
                st.warning("Please enter a question.")
    else:
        st.error("Application is not ready. Please check the logs or secrets.")


# --- Homework Helper Mode ---
elif app_mode == "✍️ Homework Helper":
    st.header("Get Help with Your Homework")

    if model:
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
    else:
        st.error("Application is not ready. Please check the logs or secrets.")

