# app.py (The Definitive Version with Bulletproof Startup)

import os
import pandas as pd
import numpy as np
import google.generativeai as genai
import streamlit as st
from PIL import Image

# --- Page Configuration ---
st.set_page_config(page_title="AI Teacher Portal", page_icon="🤖", layout="wide")

# --- Authentication and Resource Loading (run once and cached) ---
@st.cache_resource
def load_resources():
    """
    This function configures the API and loads all necessary resources.
    It is wrapped in full error handling to ensure it never crashes the app.
    """
    try:
        # Configure Google AI API using the secret key from Hugging Face settings.
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            # We display an error in the UI and return None to signal failure.
            st.error("GOOGLE_API_KEY secret not found. Please set it in your Space settings.")
            return None, None
            
        genai.configure(api_key=api_key)
        print("✅ Google AI API configured successfully.")

        # Load the generative models we will use.
        models = {
            "flash": genai.GenerativeModel('gemini-1.5-flash-latest'),
            "pro_vision": genai.GenerativeModel('gemini-1.5-pro-latest')
        }
        print("✅ Generative models loaded successfully.")

        # Load our pre-processed knowledge base from the CSV file.
        dataframe = pd.read_csv("knowledge_base.csv")
        dataframe['embedding'] = dataframe['embedding'].apply(eval)
        print("✅ Knowledge base loaded successfully.")
        
        # If everything is successful, return the resources.
        return models, dataframe

    except FileNotFoundError:
        st.error("CRITICAL ERROR: 'knowledge_base.csv' not found. App cannot function.")
        return None, None
    except Exception as e:
        # Catch any other unexpected errors during startup.
        st.error(f"An unexpected error occurred during resource loading: {e}")
        return None, None

# Load the resources when the app starts.
models, df_embedded = load_resources()

# --- Core Logic Functions ---
# (These functions remain the same as the last version)
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
    prompt = f"Answer the question based ONLY on the source material.\n\nQuestion: {question}\n\nSource Material:\n{relevant_page['text_for_search']}"
    response = models["flash"].generate_content(prompt)
    return response.text

def analyze_handwritten_image(image, instruction):
    """Handles the image analysis logic."""
    prompt = f"You are an AI Teacher. Analyze the handwritten work based on this instruction: \"{instruction}\"\n\nCRITICAL RULE: Provide ONLY the step-by-step solution. Do NOT add any summary or critique."
    response = models["pro_vision"].generate_content([prompt, image])
    return response.text


# --- Streamlit User Interface ---
st.title("🤖 AI Teacher Portal")

# --- THIS IS THE KEY CHANGE: A Master Safety Check ---
# We check if the resources were loaded successfully. If not, we display an
# error message and stop the script here, preventing any 'NameError' crashes.
if not models or df_embedded is None or df_embedded.empty:
    st.error("🚨 Application failed to initialize. Please check the container logs on Hugging Face for critical errors.")
    st.stop() # This command halts the execution of the rest of the script.

# --- If the script passes the check, it builds the UI ---
with st.sidebar:
    st.header("App Mode")
    app_mode = st.radio("Choose a feature:", ("❓ Textbook Q&A", "✍️ Homework Helper"))

if app_mode == "❓ Textbook Q&A":
    st.header("Ask a Question from the Textbook")
    text_question = st.text_area("Enter your question here:", height=150)
    if st.button("Get Answer"):
        if text_question:
            with st.spinner("The AI Teacher is thinking..."):
                response = answer_question_from_text(text_question, df_embedded)
                st.success("Here is your answer:")
                st.write(response)
        else:
            st.warning("Please enter a question.")

elif app_mode == "✍️ Homework Helper":
    st.header("Get Help with Your Homework")
    uploaded_file = st.file_uploader("Upload an image of your work", type=["png", "jpg", "jpeg"])
    instruction = st.text_input("What should I do with this image?", placeholder="e.g., 'Solve for x' or 'Check my work'")
    if st.button("Analyze Image"):
        if uploaded_file and instruction:
            with st.spinner("The AI Teacher is analyzing your image..."):
                image = Image.open(uploaded_file)
                response = analyze_handwritten_image(image, instruction)
                st.success("Here is the analysis:")
                st.write(response)
        else:
            st.warning("Please upload an image and provide an instruction.")
