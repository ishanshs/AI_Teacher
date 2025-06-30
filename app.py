# app.py (The Definitive Version with a Single, Correct Model)

# --- Section 1: Import All Necessary Libraries ---
import os
import pandas as pd
import numpy as np
import google.generativeai as genai
import streamlit as st
from PIL import Image

# --- Section 2: Page Configuration ---
st.set_page_config(page_title="AI Teacher Portal", page_icon="🤖", layout="wide")

# --- Section 3: Authentication and Resource Loading (Cached) ---
@st.cache_resource
def load_resources():
    """Configures the API and loads all necessary resources."""
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            st.error("CRITICAL ERROR: GOOGLE_API_KEY secret not found in Space settings.")
            return None, None
        genai.configure(api_key=api_key)
        print("✅ Google AI API configured successfully.")
    except Exception as e:
        st.error(f"CRITICAL ERROR during API configuration: {e}")
        return None, None
    
    # --- We now only load ONE model for all tasks ---
    # This ensures we always use the Flash model with its generous free tier.
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    print("✅ Gemini 1.5 Flash model loaded successfully.")

    try:
        dataframe = pd.read_csv("knowledge_base.csv")
        dataframe['embedding'] = dataframe['embedding'].apply(eval)
        print("✅ Knowledge base loaded successfully.")
        return model, dataframe
    except FileNotFoundError:
        st.error("CRITICAL ERROR: 'knowledge_base.csv' not found. App cannot function.")
        return None, None

# Load resources when the app starts.
model, df_embedded = load_resources()


# --- Section 4: Core Logic Functions ---
# All functions now correctly use the single 'model' object passed to them.

def find_relevant_context(query, dataframe, k=3):
    """Finds the top 'k' most relevant text chunks from the knowledge base."""
    query_embedding_response = genai.embed_content(model='models/text-embedding-004', content=query, task_type="RETRIEVAL_QUERY")
    query_embedding = query_embedding_response['embedding']
    dot_products = np.dot(np.stack(dataframe['embedding']), query_embedding)
    top_k_indices = np.argsort(dot_products)[-k:][::-1]
    return "\n\n---\n\n".join(dataframe.iloc[top_k_indices]['text_for_search'].tolist())

def answer_question_from_text(question, dataframe, ai_model):
    """Handles the text-based Q&A logic with our refined persona."""
    context = find_relevant_context(question, dataframe)
    prompt = f"""
    **Persona:** You are a friendly, cheerful, and patient AI Teacher...
    **Task & Rules:** 1. Answer the user's question clearly... 2. Base your answer on the 'Source Material'... 3. **IMPORTANT:** Never refer to the source material directly...
    ---
    **Source Material:**
    {context}
    ---
    **User's Question:**
    {question}
    """
    response = ai_model.generate_content(prompt)
    return response.text

def analyze_handwritten_image(image, instruction, ai_model):
    """Handles the image analysis logic with a focused task instruction."""
    # --- THIS IS THE CORRECTED LOGIC ---
    # It now uses the single 'gemini-1.5-flash-latest' model passed into this function.
    # It no longer incorrectly calls the 'pro-vision' model.
    prompt = f"""
    You are an expert AI Math Teacher... Your primary goal is to provide a step-by-step solution...
    **CRITICAL RULE:** Provide ONLY the direct answer... DO NOT add any extra summary...
    **User's Instruction:** "{instruction}"
    """
    response = ai_model.generate_content([prompt, image])
    return response.text


# --- Section 5: Streamlit User Interface ---
st.title("🤖 AI Teacher Portal")

# Master safety check to ensure resources loaded before building the UI.
if not model or df_embedded is None or df_embedded.empty:
    st.error("🚨 Application failed to initialize. Please check the container logs or secrets.")
    st.stop() # Halts the script if resources are not available.

with st.sidebar:
    st.header("App Mode")
    app_mode = st.radio(
        "Choose a feature:",
        ("❓ Textbook Q&A", "✍️ Homework Helper")
    )

# --- UI for Textbook Q&A Mode ---
if app_mode == "❓ Textbook Q&A":
    st.header("Ask a Question from the Textbook")
    text_question = st.text_area("Enter your question here:", height=150)
    if st.button("Get Answer"):
        if text_question:
            with st.spinner("The AI Teacher is thinking..."):
                # We pass the single loaded 'model' to the function.
                response = answer_question_from_text(text_question, df_embedded, model)
                st.success("Here is your answer:")
                st.write(response)
        else:
            st.warning("Please enter a question.")

# --- UI for Homework Helper Mode ---
elif app_mode == "✍️ Homework Helper":
    st.header("Get Help with Your Homework")
    uploaded_file = st.file_uploader("Upload an image of your work", type=["png", "jpg", "jpeg"])
    instruction = st.text_input("What should I do with this image?", placeholder="e.g., 'Solve for x' or 'Check my work'")
    if st.button("Analyze Image"):
        if uploaded_file and instruction:
            with st.spinner("The AI Teacher is analyzing your image..."):
                image = Image.open(uploaded_file)
                # We pass the single loaded 'model' to the function.
                response = analyze_handwritten_image(image, instruction, model)
                st.success("Here is the analysis:")
                st.write(response)
        else:
            st.warning("Please upload an image and provide an instruction.")
