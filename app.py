# app.py (Version 9: Improved Prompts and Diagnostics)

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

# --- Page Configuration ---
st.set_page_config(page_title="AI Teacher Portal", page_icon="🤖", layout="wide")

# --- Authentication and Resource Loading ---
@st.cache_resource
def load_resources():
    """Configures the API and loads all necessary resources."""
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        print("✅ Google AI API configured successfully.")
    except Exception as e:
        st.error(f"Failed to configure API. Check GOOGLE_API_KEY. Error: {e}")
        return None, None
    
    models = {
        "flash": genai.GenerativeModel('gemini-1.5-flash-latest'),
        "pro_vision": genai.GenerativeModel('gemini-1.5-pro-latest')
    }
    print("✅ Generative models loaded.")

    try:
        dataframe = pd.read_csv("knowledge_base.csv")
        dataframe['embedding'] = dataframe['embedding'].apply(eval)
        print("✅ Knowledge base loaded.")
        return models, dataframe
    except FileNotFoundError:
        st.error("CRITICAL ERROR: 'knowledge_base.csv' not found.")
        return None, None

models, df_embedded = load_resources()

# --- Core Logic Functions (with Upgraded Prompts) ---

def find_relevant_context(query, dataframe, k=5):
    """Finds the top 'k' most relevant text chunks."""
    query_embedding_response = genai.embed_content(model='models/text-embedding-004', content=query, task_type="RETRIEVAL_QUERY")
    query_embedding = query_embedding_response['embedding']
    dot_products = np.dot(np.stack(dataframe['embedding']), query_embedding)
    top_k_indices = np.argsort(dot_products)[-k:][::-1]
    return "\n\n---\n\n".join(dataframe.iloc[top_k_indices]['text_for_search'].tolist())

# --- This function contains our new, ultra-strict prompt for generating chapters ---
@st.cache_data
def get_chapter_list(dataframe):
    """Generates ONLY a list of chapter names for the selection menu."""
    st.write("📚 Analyzing textbook to identify chapters...")
    model = models["flash"]
    full_text_sample = "\n".join(dataframe['text_for_search'].head(200).tolist())
    prompt = f"""
    You are a data extraction expert. Your only task is to analyze the provided text from a book's table of contents and extract the chapter titles.
    - The output MUST be a single, valid JSON array of strings.
    - Do NOT add introductory text, summaries, or any text outside of the JSON array.
    - Do NOT invent or hallucinate chapter names. Only extract what is present in the text.
    
    PERFECT EXAMPLE:
    ["Chapter 1: Integers", "Chapter 2: Fractions and Decimals", "Chapter 3: Data Handling"]

    TEXT SAMPLE TO ANALYZE:
    {full_text_sample}
    """
    try:
        response = model.generate_content(prompt)
        # Store the raw response in the session state for debugging
        st.session_state.raw_chapter_response = response.text
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response, strict=False)
    except Exception as e:
        st.session_state.raw_chapter_response = f"Error during generation: {e}"
        return []

# --- This function contains our new, ultra-strict prompt for generating topics ---
@st.cache_data
def get_topic_list(chapter_name, dataframe):
    """Generates a list of topics ONLY for the selected chapter."""
    st.write(f" B Analyzing '{chapter_name}' to find its main topics...")
    model = models["flash"]
    source_material = find_relevant_context(chapter_name, dataframe, k=10)
    prompt = f"""
    You are a curriculum expert. Analyze the source material for a chapter titled '{chapter_name}'.
    Your only task is to identify and list the main topics covered in this specific chapter.
    - The output MUST be a single, valid JSON array of strings.
    - Do NOT add any conversational text. Your response must begin with `[` and end with `]`.

    PERFECT EXAMPLE for a chapter on Integers:
    ["Introduction to Integers", "Properties of Addition and Subtraction", "Multiplication of Integers"]

    SOURCE MATERIAL:
    {source_material}
    """
    try:
        response = model.generate_content(prompt)
        # Store the raw response for debugging
        st.session_state.raw_topic_response = response.text
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response, strict=False)
    except Exception as e:
        st.session_state.raw_topic_response = f"Error during generation: {e}"
        return []

# ... (The rest of your helper functions like generate_lecture_script, convert_script_to_audio, etc. remain here) ...
def generate_lecture_script(topic, source_material):
    # Placeholder for brevity
    return [{"spoken_text": f"This is part one of the lecture on {topic}.", "display_text": f"Topic: {topic}"}]
def convert_script_to_audio(script_parts, folder):
    # Placeholder for brevity
    return ["dummy_audio.mp3"]
def deliver_grouped_lecture(script, audio, size):
    # Placeholder for brevity
    pass

# --- Streamlit User Interface ---
st.title("🤖 AI Teacher Portal")

with st.sidebar:
    st.header("App Mode")
    app_mode = st.radio("Choose a feature:", ("❓ Textbook Q&A", "✍️ Homework Helper", "👩‍🏫 Teacher Lecture Mode"))

# --- (Q&A and Homework Helper UI code remains here) ---
if app_mode == "❓ Textbook Q&A":
    st.header("Ask a Question from the Textbook")
    # ...
elif app_mode == "✍️ Homework Helper":
    st.header("Get Help with Your Homework")
    # ...

# --- NEW: Teacher Lecture Mode with Diagnostics ---
elif app_mode == "👩‍🏫 Teacher Lecture Mode":
    st.header("Generate a Custom Audio Lecture")

    if models and df_embedded is not None and not df_embedded.empty:
        # Generate and display chapters
        chapters = get_chapter_list(df_embedded)
        if not chapters:
            st.error("Could not generate a list of chapters from the textbook.")
        else:
            selected_chapter = st.selectbox("Step 1: Choose a Chapter", options=chapters)
            
            if selected_chapter:
                # Generate and display topics for the selected chapter
                topics = get_topic_list(selected_chapter, df_embedded)
                if not topics:
                    st.error(f"Could not generate a list of topics for '{selected_chapter}'.")
                else:
                    selected_topic = st.selectbox("Step 2: Choose a Topic", options=topics)
                    
                    if st.button("Generate Lecture", key="generate_lecture"):
                        if selected_topic:
                            st.info(f"Preparing a lecture on: '{selected_topic}'")
                            with st.spinner("Generating lecture script and audio..."):
                                source_material = find_relevant_context(selected_topic, df_embedded)
                                lecture_script = generate_lecture_script(selected_topic, source_material)
                                if lecture_script:
                                    audio_folder = f"Lecture_Assets/{selected_topic.replace(' ', '_')}/Audio/"
                                    audio_files = convert_script_to_audio(lecture_script, audio_folder)
                                    deliver_grouped_lecture(lecture_script, audio_files, 3)
                                else:
                                    st.error("Failed to generate lecture script.")
                        else:
                            st.warning("Please select a topic.")

        # --- Diagnostic Expander ---
        with st.expander("Show AI Raw Output (for debugging)"):
            st.write("--- Chapter Generation Raw Output ---")
            st.code(st.session_state.get('raw_chapter_response', 'No request made yet.'))
            st.write("--- Topic Generation Raw Output ---")
            st.code(st.session_state.get('raw_topic_response', 'No request made yet.'))

    else:
        st.error("Application is not ready. Knowledge base may be missing.")

