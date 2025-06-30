# app.py (The Definitive Version with All Features and Bulletproof Startup)

import os
import pandas as pd
import numpy as np
import google.generativeai as genai
import streamlit as st
from PIL import Image
from gtts import gTTS
from mutagen.mp3 import MP3
import time
import json
import re
from num2words import num2words
from google.api_core.exceptions import TooManyRequests
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

# --- Page Configuration ---
st.set_page_config(page_title="AI Teacher Portal", page_icon="🤖", layout="wide")

# --- Authentication and Resource Loading (Cached for performance) ---
@st.cache_resource
def load_resources():
    """Configures the API and loads all necessary resources."""
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            st.error("GOOGLE_API_KEY secret not found. Please set it in your Space secrets.")
            return None, None
        genai.configure(api_key=api_key)
        print("✅ Google AI API configured successfully.")
    except Exception as e:
        st.error(f"Failed to configure API. Check GOOGLE_API_KEY. Error: {e}")
        return None, None
    
    models = {
        "flash": genai.GenerativeModel('gemini-1.5-flash-latest'),
        "pro_vision": genai.GenerativeModel('gemini-1.5-pro-latest')
    }
    print("✅ Generative models loaded successfully.")

    try:
        dataframe = pd.read_csv("knowledge_base.csv")
        dataframe['embedding'] = dataframe['embedding'].apply(eval)
        print("✅ Knowledge base loaded successfully.")
        return models, dataframe
    except FileNotFoundError:
        st.error("CRITICAL ERROR: 'knowledge_base.csv' not found. App cannot function.")
        return None, None

# Load resources when the app starts.
models, df_embedded = load_resources()

# --- Core Logic Functions ---
@retry(retry=retry_if_exception_type(TooManyRequests), wait=wait_fixed(60), stop=stop_after_attempt(3))
def generate_content_with_retry(model, prompt):
    return model.generate_content(prompt)

def find_relevant_context(query, dataframe, k=3):
    """Finds the top 'k' most relevant text chunks."""
    query_embedding_response = genai.embed_content(model='models/text-embedding-004', content=query, task_type="RETRIEVAL_QUERY")
    query_embedding = query_embedding_response['embedding']
    dot_products = np.dot(np.stack(dataframe['embedding']), query_embedding)
    top_k_indices = np.argsort(dot_products)[-k:][::-1]
    return "\n\n---\n\n".join(dataframe.iloc[top_k_indices]['text_for_search'].tolist())

def answer_question_from_text(question, dataframe):
    """Handles the text-based Q&A logic."""
    context = find_relevant_context(question, dataframe)
    prompt = f"**Persona:** You are a friendly AI Teacher... **Task & Rules:** Answer the user's question based on the 'Source Material'... **Source Material:** {context} --- **User's Question:** {question}"
    response = generate_content_with_retry(models["flash"], prompt)
    return response.text

def analyze_handwritten_image(image, instruction):
    """Handles the image analysis logic."""
    prompt = f"You are an expert AI Math Teacher... **CRITICAL RULE:** Provide ONLY the step-by-step solution... **User's Instruction:** \"{instruction}\""
    response = generate_content_with_retry(models["pro_vision"], [prompt, image])
    return response.text

@st.cache_data
def get_chapter_list(_dataframe):
    """Generates ONLY a list of chapter names."""
    st.write("📚 Analyzing textbook to identify chapters...")
    prompt = "Your task is to act as a data extractor... Your ONLY job is to identify the chapter titles... Output MUST be a single, valid JSON array..."
    full_text_sample = "\n".join(_dataframe['text_for_search'].head(150).tolist())
    response = generate_content_with_retry(models["flash"], [prompt.replace("{full_text_sample}", full_text_sample)])
    try:
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response, strict=False)
    except Exception as e: return [f"Error generating chapters: {e}"]

@st.cache_data
def get_topic_list(chapter_name, _dataframe):
    """Generates a list of topics ONLY for the selected chapter."""
    st.write(f" B Analyzing '{chapter_name}' to find its main topics...")
    source_material = find_relevant_context(chapter_name, _dataframe, k=10)
    prompt = f"You are a curriculum expert... list the main topics for '{chapter_name}'... Output MUST be a single, valid JSON array..."
    response = generate_content_with_retry(models["flash"], [prompt.replace("{source_material}", source_material)])
    try:
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response, strict=False)
    except Exception as e: return [f"Error generating topics: {e}"]

# ... (All other lecture helper functions like verbalize_formula, generate_lecture_script, etc. would go here) ...
# For now, we will just use a placeholder to ensure the UI loads.
def placeholder_lecture_function(topic):
    return f"Lecture generation for '{topic}' is the next step!"


# --- Streamlit User Interface ---
st.title("🤖 AI Teacher Portal")

# Master safety check to ensure resources loaded before building the UI.
if not models or df_embedded is None or df_embedded.empty:
    st.error("🚨 Application failed to initialize. Please check the container logs on Hugging Face for critical errors.")
    st.stop() # Halts the script if resources are not available.

with st.sidebar:
    st.header("App Mode")
    # --- THIS IS THE CORRECTED WIDGET with all three modes ---
    app_mode = st.radio(
        "Choose a feature:",
        ("❓ Textbook Q&A", "✍️ Homework Helper", "👩‍🏫 Teacher Lecture Mode")
    )

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

# --- UI for Teacher Lecture Mode ---
elif app_mode == "👩‍🏫 Teacher Lecture Mode":
    st.header("Generate a Custom Audio Lecture")
    
    # Step 1: Select a Chapter
    chapters = get_chapter_list(df_embedded)
    if chapters and "Error" not in chapters[0]:
        selected_chapter = st.selectbox("Step 1: Choose a Chapter", options=chapters)
        
        if selected_chapter:
            # Step 2: Select a Topic from that Chapter
            topics = get_topic_list(selected_chapter, df_embedded)
            if topics and "Error" not in topics[0]:
                selected_topic = st.selectbox("Step 2: Choose a Topic", options=topics)
                
                if st.button("Generate Lecture", key="generate_lecture"):
                    if selected_topic:
                        with st.spinner(f"Preparing lecture on '{selected_topic}'..."):
                            # This is where we would call the full lecture pipeline.
                            # For now, we'll use a placeholder.
                            response = placeholder_lecture_function(selected_topic)
                            st.success(response)
                    else:
                        st.warning("Please select a topic.")
            else:
                st.error("Could not load topics for this chapter.")
    else:
        st.error("Could not load chapters from the textbook.")
