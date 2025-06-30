# app.py (The Definitive Version with Enhanced Error Reporting)

import os
import pandas as pd
import numpy as np
import google.generativeai as genai
import streamlit as st
from PIL import Image
from gtts import gTTS
import json
import re
from num2words import num2words
from google.api_core.exceptions import TooManyRequests
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

# --- Page Configuration ---
st.set_page_config(page_title="AI Teacher Portal", page_icon="🤖", layout="wide")

# --- Authentication and Resource Loading (Cached) ---
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

models, df_embedded = load_resources()

# --- Core Logic Functions ---
@retry(retry=retry_if_exception_type(TooManyRequests), wait=wait_fixed(60), stop=stop_after_attempt(3))
def generate_content_with_retry(model, prompt):
    """A resilient wrapper for the 'generate_content' API call."""
    return model.generate_content(prompt)

def find_relevant_context(query, dataframe, k=3):
    """Finds the top 'k' most relevant text chunks."""
    query_embedding_response = genai.embed_content(model='models/text-embedding-004', content=query, task_type="RETRIEVAL_QUERY")
    query_embedding = query_embedding_response['embedding']
    dot_products = np.dot(np.stack(dataframe['embedding']), query_embedding)
    top_k_indices = np.argsort(dot_products)[-k:][::-1]
    return "\n\n---\n\n".join(dataframe.iloc[top_k_indices]['text_for_search'].tolist())

# ... (answer_question_from_text and analyze_handwritten_image functions remain here)
def answer_question_from_text(question, dataframe):
    context = find_relevant_context(question, dataframe)
    prompt = f"**Persona:** You are a friendly AI Teacher... **Task:** Answer the user's question... **Source Material:** {context} --- **Question:** {question}"
    response = generate_content_with_retry(models["flash"], prompt)
    return response.text

def analyze_handwritten_image(image, instruction):
    prompt = f"You are an expert AI Math Teacher... **Rule:** Provide ONLY the step-by-step solution... **Instruction:** \"{instruction}\""
    response = generate_content_with_retry(models["pro_vision"], [prompt, image])
    return response.text


# --- Lecture Mode Functions (with improved error capture) ---
@st.cache_data
def get_chapter_list(_dataframe):
    """Generates ONLY a list of chapter names for the selection menu."""
    st.write("📚 Analyzing textbook to identify chapters...")
    try:
        model = models["flash"]
        full_text_sample = "\n".join(_dataframe['text_for_search'].head(200).tolist())
        prompt = f"Your ONLY job is to analyze the text and extract chapter titles. Output MUST be a single, valid JSON array of strings. PERFECT EXAMPLE: [\"Chapter 1: Integers\", \"Chapter 2: Fractions\"]. TEXT TO ANALYZE: {full_text_sample}"
        response = generate_content_with_retry(model, prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response, strict=False)
    except Exception as e:
        # Instead of returning a generic message, return the ACTUAL error.
        return {"error": f"Failed to generate or parse chapter list. Details: {str(e)}"}

@st.cache_data
def get_topic_list(chapter_name, _dataframe):
    """Generates a list of topics ONLY for the selected chapter."""
    st.write(f" B Analyzing '{chapter_name}' to find its main topics...")
    try:
        model = models["flash"]
        source_material = find_relevant_context(chapter_name, _dataframe, k=10)
        prompt = f"You are a curriculum expert. Analyze the source material for '{chapter_name}'. List the main topics. Output MUST be a single, valid JSON array. SOURCE: {source_material}"
        response = generate_content_with_retry(model, prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response, strict=False)
    except Exception as e:
        return {"error": f"Failed to generate or parse topics for '{chapter_name}'. Details: {str(e)}"}

# --- Placeholder functions for the rest of the lecture pipeline ---
def generate_lecture_script(topic, source_material): return [{"display_text": f"Script for {topic}"}]
def convert_script_to_audio(script_parts, folder): return ["dummy.mp3"]
def deliver_grouped_lecture(script, audio, size): pass


# --- Streamlit User Interface ---
st.title("🤖 AI Teacher Portal")

# Master safety check
if not models or df_embedded is None or df_embedded.empty:
    st.error("🚨 Application failed to initialize. Please check the container logs for critical errors.")
    st.stop()

with st.sidebar:
    st.header("App Mode")
    app_mode = st.radio("Choose a feature:", ("❓ Textbook Q&A", "✍️ Homework Helper", "👩‍🏫 Teacher Lecture Mode"))

# --- UI for Textbook Q&A and Homework Helper remain the same ---
if app_mode == "❓ Textbook Q&A":
    st.header("Ask a Question from the Textbook")
    # ... UI code ...
elif app_mode == "✍️ Homework Helper":
    st.header("Get Help with Your Homework")
    # ... UI code ...

# --- UI for Teacher Lecture Mode (with enhanced error display) ---
elif app_mode == "👩‍🏫 Teacher Lecture Mode":
    st.header("Generate a Custom Audio Lecture")
    
    chapters = get_chapter_list(df_embedded)
    
    # --- THIS IS THE KEY CHANGE ---
    # Check if the returned object is a dictionary with an 'error' key.
    if isinstance(chapters, dict) and "error" in chapters:
        # If so, display the specific error message and stop this section.
        st.error(f"An error occurred while loading chapters:\n\n{chapters['error']}")
    else:
        # Otherwise, proceed as normal.
        selected_chapter = st.selectbox("Step 1: Choose a Chapter", options=chapters)
        if selected_chapter:
            topics = get_topic_list(selected_chapter, df_embedded)
            if isinstance(topics, dict) and "error" in topics:
                st.error(f"An error occurred while loading topics for this chapter:\n\n{topics['error']}")
            else:
                selected_topic = st.selectbox("Step 2: Choose a Topic", options=topics)
                if st.button("Generate Lecture", key="generate_lecture"):
                    if selected_topic:
                        st.info(f"Generating lecture for '{selected_topic}'...")
                        # ... (rest of the lecture pipeline)
                    else:
                        st.warning("Please select a topic.")
