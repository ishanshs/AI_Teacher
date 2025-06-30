# app.py (The Definitive Version with a Detailed Progress Indicator)

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

# --- Authentication and Resource Loading (Cached) ---
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
    """A resilient wrapper for the 'generate_content' API call."""
    return model.generate_content(prompt)

def find_relevant_context(query, dataframe, k=3):
    """Finds the top 'k' most relevant text chunks."""
    query_embedding_response = genai.embed_content(model='models/text-embedding-004', content=query, task_type="RETRIEVAL_QUERY")
    query_embedding = query_embedding_response['embedding']
    dot_products = np.dot(np.stack(dataframe['embedding']), query_embedding)
    top_k_indices = np.argsort(dot_products)[-k:][::-1]
    return "\n\n---\n\n".join(dataframe.iloc[top_k_indices]['text_for_search'].tolist())

def answer_question_from_text(question, dataframe):
    """Handles the text-based Q&A logic with our refined persona."""
    context = find_relevant_context(question, dataframe)
    prompt = f"**Persona:** You are a friendly AI Teacher... **Task & Rules:**... Answer the user's question based on the 'Source Material'... **Source Material:** {context} --- **User's Question:** {question}"
    response = generate_content_with_retry(models["flash"], prompt)
    return response.text

def analyze_handwritten_image(image, instruction):
    """Handles the image analysis logic with a focused task instruction."""
    prompt = f"You are an expert AI Math Teacher... **CRITICAL RULE:** Provide ONLY the step-by-step solution... **User's Instruction:** \"{instruction}\""
    response = generate_content_with_retry(models["pro_vision"], [prompt, image])
    return response.text

@st.cache_data
def get_chapter_list(_dataframe):
    """Generates ONLY a list of chapter names for the selection menu."""
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

def verbalize_formula(formula):
    """Programmatically converts a formula string into a verbal explanation."""
    pronunciation_map = {'+': ' plus ', '-': ' minus ', 'x': ' multiplied by ', '*': ' multiplied by ', '/': ' divided by ', '=': ' equals '}
    tokens = re.findall(r'(\d+\.?\d*|[a-zA-Z]+|.)', formula)
    verbalized_tokens = [num2words(int(t)) if t.isdigit() else pronunciation_map.get(t, f' {t} ') for t in tokens]
    return ' '.join(verbalized_tokens).strip().replace('  ', ' ')

def generate_lecture_script(topic, source_material):
    """Uses Gemini to create a lecture script with formula placeholders."""
    prompt = f"You have two jobs... PERSONA: You are a friendly AI Teacher... TASK: Create a lecture script... PERFECT EXAMPLE: [{{'spoken_text': 'Let's solve <FORMULA:2x-3=5>', 'display_text': 'Example: 2x-3=5'}}]... SOURCE MATERIAL: {source_material}"
    response = generate_content_with_retry(models["flash"], [prompt])
    try:
        raw_text = response.text.strip()
        cleaned_response = raw_text.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_response, strict=False)
    except Exception as e:
        print(f"❌ An error occurred during script generation: {e}")
        return None

def convert_script_to_audio(script_parts, lecture_audio_folder):
    """Converts 'spoken_text' into MP3 files, processing formula placeholders."""
    os.makedirs(lecture_audio_folder, exist_ok=True)
    audio_files = []
    for i, part in enumerate(script_parts):
        text_to_speak = part['spoken_text']
        placeholders = re.findall(r'<FORMULA:(.*?)>', text_to_speak)
        for formula in placeholders:
            text_to_speak = text_to_speak.replace(f'<FORMULA:{formula}>', verbalize_formula(formula))
        tts = gTTS(text=text_to_speak, lang='en', tld='co.in', slow=False)
        file_path = os.path.join(lecture_audio_folder, f"part_{i}.mp3")
        tts.save(file_path)
        audio_files.append(file_path)
    return audio_files

def deliver_grouped_lecture(script_parts, audio_files, group_size=3):
    """Delivers the lecture in digestible, manually-playable parts."""
    st.success("Your lecture is ready! Press play on each part to listen.")
    for i in range(0, len(script_parts), group_size):
        script_group = script_parts[i:i + group_size]
        audio_group = audio_files[i:i + group_size]
        st.markdown(f"--- \n### 📖 Lecture Part {i//group_size + 1}")
        for j, part in enumerate(script_group):
            st.write(f"**Note:** {part['display_text']}")
            st.audio(audio_group[j])
    st.markdown("---")
    st.balloons()
    st.success("🎉 End of Lecture! Great job!")

# --- Streamlit User Interface ---
st.title("🤖 AI Teacher Portal")

if not models or df_embedded is None or df_embedded.empty:
    st.error("🚨 Application failed to initialize. Please check logs or secrets.")
    st.stop()

with st.sidebar:
    st.header("App Mode")
    app_mode = st.radio("Choose a feature:", ("❓ Textbook Q&A", "✍️ Homework Helper", "👩‍🏫 Teacher Lecture Mode"))

if app_mode == "❓ Textbook Q&A":
    st.header("Ask a Question from the Textbook")
    # ... UI code ...
elif app_mode == "✍️ Homework Helper":
    st.header("Get Help with Your Homework")
    # ... UI code ...
elif app_mode == "👩‍🏫 Teacher Lecture Mode":
    st.header("Generate a Custom Audio Lecture")
    chapters = get_chapter_list(df_embedded)
    if chapters and isinstance(chapters, list) and "Error" not in chapters[0]:
        selected_chapter = st.selectbox("Step 1: Choose a Chapter", options=chapters, index=None, placeholder="Select a chapter...")
        
        if selected_chapter:
            topics = get_topic_list(selected_chapter, df_embedded)
            if topics and isinstance(topics, list) and "Error" not in topics[0]:
                selected_topic = st.selectbox("Step 2: Choose a Topic", options=topics, index=None, placeholder="Select a topic...")
                
                if st.button("Generate Lecture", key="generate_lecture"):
                    if selected_topic:
                        # --- THIS IS THE NEW, MORE INFORMATIVE PROGRESS INDICATOR ---
                        with st.status(f"Preparing lecture on '{selected_topic}'...", expanded=True) as status:
                            st.write("Gathering relevant material from the textbook...")
                            source_material = find_relevant_context(selected_topic, df_embedded, k=10)
                            
                            st.write("✍️ Writing the lecture script with AI...")
                            lecture_script = generate_lecture_script(selected_topic, source_material)
                            
                            if lecture_script:
                                st.write("🎙️ Creating audio files for the lecture...")
                                lecture_asset_path = f"Lecture_Assets/{selected_topic.replace(' ', '_')}"
                                audio_files = convert_script_to_audio(lecture_script, lecture_asset_path)
                                
                                status.update(label="Lecture ready!", state="complete", expanded=False)
                                
                                # Deliver the final lecture
                                deliver_grouped_lecture(lecture_script, audio_files)
                            else:
                                status.update(label="Error!", state="error", expanded=True)
                                st.error("Could not generate the lecture script. The AI may have returned an unexpected format or timed out.")
                    else:
                        st.warning("Please select a topic.")
            else:
                st.error("Could not load topics for this chapter.")
    else:
        st.error("Could not load chapters from the textbook.")
