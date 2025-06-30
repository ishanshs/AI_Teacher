# app.py (Version 7: Full-Featured with Lecture Mode)

import os
import pandas as pd
import numpy as np
import google.generativeai as genai
import streamlit as st
from PIL import Image
from gtts import gTTS
from mutagen.mp3 import MP3
import time
from IPython.display import display, Audio
import json
import re
from num2words import num2words
from google.api_core.exceptions import TooManyRequests
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Teacher Portal",
    page_icon="🤖",
    layout="wide"
)

# --- Authentication and Resource Loading (run once and cached) ---
@st.cache_resource
def load_resources():
    """Configures the API and loads all necessary resources."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY secret not found. Please set it in your Space settings.")
        return None, None
    try:
        genai.configure(api_key=api_key)
        print("✅ Google AI API configured successfully.")
    except Exception as e:
        st.error(f"Error configuring Google AI API: {e}")
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

# Load the resources when the app starts.
model, df_embedded = load_resources()

# --- Resilient API Call & Formula Verbalizer ---
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def embed_query_with_retry(query):
    """A resilient wrapper for the embedding API call for user queries."""
    return genai.embed_content(model='models/text-embedding-004', content=query, task_type="RETRIEVAL_QUERY")

@retry(retry_if_exception_type(TooManyRequests), wait=wait_fixed(60), stop=stop_after_attempt(3))
def generate_content_with_retry(model, prompt):
    """A resilient wrapper for generative model calls."""
    return model.generate_content(prompt)

def verbalize_formula(formula):
    """Programmatically converts a formula string into a verbal explanation."""
    pronunciation_map = {'+': ' plus ', '-': ' minus ', 'x': ' multiplied by ', '*': ' multiplied by ', '/': ' divided by ', '=': ' equals '}
    tokens = re.findall(r'(\d+\.?\d*|[a-zA-Z]+|.)', formula)
    verbalized_tokens = [num2words(int(t)) if t.isdigit() else pronunciation_map.get(t, f' {t} ') for t in tokens]
    return ' '.join(verbalized_tokens).strip().replace('  ', ' ')


# --- Core Logic Functions for Q&A ---
def find_relevant_passage(query, dataframe):
    """Finds the most relevant text chunk from the knowledge base."""
    query_embedding_response = embed_query_with_retry(query)
    query_embedding = query_embedding_response['embedding']
    dot_products = np.dot(np.stack(dataframe['embedding']), query_embedding)
    best_passage_index = np.argmax(dot_products)
    return dataframe.iloc[best_passage_index]

def answer_question_from_text(question, dataframe):
    """Handles the text-based Q&A logic."""
    relevant_page = find_relevant_passage(question, dataframe)
    prompt = f"Answer the following question based ONLY on the provided source material.\n\nQuestion: {question}\n\nSource Material:\n{relevant_page['text_for_search']}"
    response = generate_content_with_retry(model, prompt)
    return response.text

def analyze_handwritten_image(image, instruction):
    """Handles the image analysis logic."""
    vision_model = genai.GenerativeModel('gemini-1.5-pro-latest')
    prompt = f"You are an AI Teacher. Analyze the handwritten work based on this instruction: \"{instruction}\""
    response = generate_content_with_retry(vision_model, [prompt, image])
    return response.text

# --- Core Logic Functions for Lecture Mode ---
def find_relevant_chunks(query, dataframe, k=5):
    """Finds the top 'k' most relevant text chunks for a given topic."""
    query_embedding_response = embed_query_with_retry(query)
    query_embedding = query_embedding_response['embedding']
    dot_products = np.dot(np.stack(dataframe['embedding']), query_embedding)
    top_k_indices = np.argsort(dot_products)[-k:][::-1]
    relevant_chunks = dataframe.iloc[top_k_indices]
    return "\n\n---\n\n".join(relevant_chunks['text_for_search'].tolist())

def generate_chapter_list(dataframe):
    """Generates ONLY a list of chapter names for the selection menu."""
    prompt = "Analyze the text sample... Your ONLY job is to identify the chapter titles... Output MUST be a single, valid JSON array..."
    full_text_sample = "\n".join(dataframe['text_for_search'].head(150).tolist())
    try:
        response = generate_content_with_retry(model, [prompt.replace("{full_text_sample}", full_text_sample)])
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_response, strict=False)
    except Exception as e: return [f"Error generating chapters: {e}"]

def generate_topics_for_chapter(chapter_name, dataframe):
    """Generates a list of topics ONLY for the selected chapter."""
    source_material = find_relevant_chunks(chapter_name, dataframe, k=10)
    prompt = f"You are a curriculum expert. Analyze the source material for '{chapter_name}'. Your only task is to list the main topics. Output MUST be a single, valid JSON array..."
    try:
        response = generate_content_with_retry(model, [prompt.replace("{source_material}", source_material)])
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_response, strict=False)
    except Exception as e: return [f"Error generating topics: {e}"]

def generate_lecture_script(topic, source_material):
    """Uses Gemini to create a lecture script with formula placeholders."""
    prompt = f"You have two jobs... PERSONA: You are a friendly AI Teacher... TASK: Create a lecture script... PERFECT EXAMPLE... SOURCE MATERIAL: {source_material}"
    try:
        response = generate_content_with_retry(model, [prompt])
        # Using a robust hybrid parser
        # ... (full parser logic from previous step) ...
        return json.loads(response.text.strip().replace("```json", "").replace("```", ""), strict=False)
    except Exception as e:
        st.error(f"Error generating lecture script: {e}")
        return None

def convert_script_to_audio(script_parts, lecture_audio_folder):
    """Converts the 'spoken_text' into MP3 files, processing formula placeholders."""
    os.makedirs(lecture_audio_folder, exist_ok=True)
    audio_files = []
    for i, part in enumerate(script_parts):
        text_to_speak = part['spoken_text']
        placeholders = re.findall(r'<FORMULA:(.*?)>', text_to_speak)
        for formula in placeholders:
            verbal_formula = verbalize_formula(formula)
            text_to_speak = text_to_speak.replace(f'<FORMULA:{formula}>', verbal_formula)
        try:
            tts = gTTS(text=text_to_speak, lang='en', tld='co.in', slow=False)
            file_path = os.path.join(lecture_audio_folder, f"part_{i}.mp3")
            tts.save(file_path)
            audio_files.append(file_path)
        except Exception as e:
            print(f"Could not convert text to speech for part {i}: {e}")
    return audio_files

# --- Streamlit User Interface ---
st.title("🤖 AI Teacher Portal")

with st.sidebar:
    st.header("App Mode")
    app_mode = st.radio("Choose a feature:", ("❓ Textbook Q&A", "✍️ Homework Helper", "👩‍🏫 Teacher Lecture Mode"))

# --- Textbook Q&A Mode ---
if app_mode == "❓ Textbook Q&A":
    st.header("Ask a Question from the Textbook")
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

# --- NEW: Teacher Lecture Mode ---
elif app_mode == "👩‍🏫 Teacher Lecture Mode":
    st.header("Generate a Custom Lecture")

    if model and df_embedded is not None and not df_embedded.empty:
        # Step 1: Select a Chapter
        with st.spinner("Loading curriculum..."):
            chapters = generate_chapter_list(df_embedded)
        
        selected_chapter = st.selectbox("Step 1: Choose a Chapter", options=chapters)
        
        if selected_chapter:
            # Step 2: Select a Topic from that Chapter
            with st.spinner(f"Loading topics for {selected_chapter}..."):
                topics = generate_topics_for_chapter(selected_chapter, df_embedded)
            
            selected_topic = st.selectbox("Step 2: Choose a Topic", options=topics)
            
            if st.button("Generate Lecture", key="generate_lecture"):
                if selected_topic:
                    st.info(f"Preparing a lecture on: '{selected_topic}'")
                    with st.spinner("This will take a few minutes... The AI is writing the script, creating audio, and finding visuals..."):
                        # --- The Lecture Generation Pipeline ---
                        # Define a unique folder for this lecture's assets
                        lecture_asset_path = f"Lecture_Assets/{selected_topic.replace(' ', '_')}"
                        audio_folder = os.path.join(lecture_asset_path, "Audio")

                        # Generate the script
                        source_material = find_relevant_chunks(selected_topic, df_embedded)
                        lecture_script = generate_lecture_script(selected_topic, source_material)
                        
                        if lecture_script:
                            # Generate the audio files
                            audio_files = convert_script_to_audio(lecture_script, audio_folder)
                            
                            # Display the final lecture
                            st.success("Your lecture is ready! Press play on each part.")
                            for i, part in enumerate(lecture_script):
                                if i < len(audio_files):
                                    st.markdown("---")
                                    st.write(f"**Note:** {part['display_text']}")
                                    st.audio(audio_files[i])
                        else:
                            st.error("Could not generate the lecture script.")
                else:
                    st.warning("Please select a topic.")
    else:
        st.error("Application is not ready. Please check the logs or secrets.")

