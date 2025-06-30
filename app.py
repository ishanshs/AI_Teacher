# app.py (The Definitive Version with Caching Fix)

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
    # ... (This function remains the same)
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("CRITICAL ERROR: GOOGLE_API_KEY secret not found.")
        return None, None
    try:
        genai.configure(api_key=api_key)
        print("✅ Google AI API configured successfully.")
    except Exception as e:
        st.error(f"CRITICAL ERROR during API configuration: {e}")
        return None, None
    
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

models_and_data = load_resources()
if models_and_data:
    model, df_embedded = models_and_data
else:
    model, df_embedded = None, None

# --- Core Logic Functions ---
@retry(retry=retry_if_exception_type(TooManyRequests), wait=wait_fixed(60), stop=stop_after_attempt(3))
def generate_content_with_retry(model, prompt):
    return model.generate_content(prompt)

def find_relevant_context(query, dataframe, k=3):
    query_embedding_response = genai.embed_content(model='models/text-embedding-004', content=query, task_type="RETRIEVAL_QUERY")
    query_embedding = query_embedding_response['embedding']
    dot_products = np.dot(np.stack(dataframe['embedding']), query_embedding)
    top_k_indices = np.argsort(dot_products)[-k:][::-1]
    return "\n\n---\n\n".join(dataframe.iloc[top_k_indices]['text_for_search'].tolist())

def answer_question_from_text(question, dataframe, model):
    context = find_relevant_context(question, dataframe)
    prompt = f"**Persona:** You are a friendly AI Teacher... **Task:** Answer the user's question... **Source Material:** {context} --- **Question:** {question}"
    response = generate_content_with_retry(model, prompt)
    return response.text

def analyze_handwritten_image(image, instruction, model):
    vision_model = genai.GenerativeModel('gemini-1.5-pro-latest') # Using Pro for its superior vision
    prompt = f"You are an expert AI Math Teacher... **Rule:** Provide ONLY the step-by-step solution... **Instruction:** \"{instruction}\""
    response = generate_content_with_retry(vision_model, [prompt, image])
    return response.text

# --- These are the functions with the corrected signatures ---
@st.cache_data
def get_chapter_list(_dataframe, _model): # Use underscore to tell Streamlit to ignore these for caching
    """Generates ONLY a list of chapter names for the selection menu."""
    st.write("📚 Analyzing textbook to identify chapters...")
    full_text_sample = "\n".join(_dataframe['text_for_search'].head(200).tolist())
    prompt = f"Your ONLY job is to analyze the text and extract chapter titles. Output MUST be a single, valid JSON array of strings... TEXT TO ANALYZE: {full_text_sample}"
    try:
        response = generate_content_with_retry(_model, prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response, strict=False)
    except Exception as e:
        return {"error": f"Failed to generate chapters: {str(e)}"}

@st.cache_data
def get_topic_list(chapter_name, _dataframe, _model): # Use underscore here as well
    """Generates a list of topics ONLY for the selected chapter."""
    st.write(f" B Analyzing '{chapter_name}' to find its main topics...")
    source_material = find_relevant_context(chapter_name, _dataframe, k=10)
    prompt = f"You are a curriculum expert. Analyze the source material for '{chapter_name}'. List the main topics. Output MUST be a single, valid JSON array... SOURCE: {source_material}"
    try:
        response = generate_content_with_retry(_model, prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response, strict=False)
    except Exception as e:
        return {"error": f"Failed to generate topics: {str(e)}"}

# ... (All other lecture helper functions like generate_lecture_script, convert_script_to_audio, etc.) ...


# --- Streamlit User Interface ---
st.title("🤖 AI Teacher Portal")

if not model or df_embedded is None or df_embedded.empty:
    st.error("🚨 Application failed to initialize. Please check the container logs or secrets.")
    st.stop()

with st.sidebar:
    st.header("App Mode")
    app_mode = st.radio("Choose a feature:", ("❓ Textbook Q&A", "✍️ Homework Helper", "👩‍🏫 Teacher Lecture Mode"))

# ... (Q&A and Homework Helper UI code remains the same) ...

elif app_mode == "👩‍🏫 Teacher Lecture Mode":
    st.header("Generate a Custom Audio Lecture")
    
    # --- The calls to the functions now match the new signatures ---
    chapters = get_chapter_list(df_embedded, model)
    
    if isinstance(chapters, dict) and "error" in chapters:
        st.error(f"An error occurred while loading chapters:\n\n{chapters['error']}")
    elif not chapters:
        st.warning("The AI could not identify any chapters from the provided text.")
    else:
        selected_chapter = st.selectbox("Step 1: Choose a Chapter", options=chapters)
        if selected_chapter:
            topics = get_topic_list(selected_chapter, df_embedded, model)
            if isinstance(topics, dict) and "error" in topics:
                st.error(f"An error occurred while loading topics:\n\n{topics['error']}")
            elif not topics:
                st.warning(f"The AI could not identify any topics for '{selected_chapter}'.")
            else:
                selected_topic = st.selectbox("Step 2: Choose a Topic", options=topics)
                if st.button("Generate Lecture", key="generate_lecture"):
                    if selected_topic:
                        # ... (Full lecture generation pipeline would be called here) ...
                        st.success(f"Lecture generation for '{selected_topic}' is ready to be built!")

