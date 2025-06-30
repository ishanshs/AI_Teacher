# app.py (The Definitive Version with All Features Implemented)

# --- Section 1: Import All Necessary Libraries ---
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

# --- Section 2: Page Configuration ---
# This sets the title and icon that appear in the browser tab.
st.set_page_config(page_title="AI Teacher Portal", page_icon="🤖", layout="wide")

# --- Section 3: Authentication and Resource Loading (Cached) ---
# The @st.cache_resource decorator ensures this complex setup runs only once.
@st.cache_resource
def load_resources():
    """Configures the API and loads all necessary resources."""
    # Configure Google AI API using the secret key from Hugging Face settings.
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        print("✅ Google AI API configured successfully.")
    except Exception as e:
        st.error(f"Failed to configure API: Please ensure GOOGLE_API_KEY is set correctly in your Space secrets. Error: {e}")
        return None, None

    # Load the generative models we will use.
    models = {
        "flash": genai.GenerativeModel('gemini-1.5-flash-latest'),
        "pro_vision": genai.GenerativeModel('gemini-1.5-pro-latest')
    }
    print("✅ Generative models loaded successfully.")

    # Load our pre-processed knowledge base from the CSV file.
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


# --- Section 4: Core Logic Functions ---
# This section contains all the backend functions that perform the AI tasks.

# This function is used by our other helper functions to make resilient API calls.
@retry(retry=retry_if_exception_type(TooManyRequests), wait=wait_fixed(60), stop=stop_after_attempt(3))
def generate_content_with_retry(model, prompt):
    """A resilient wrapper for the 'generate_content' API call."""
    return model.generate_content(prompt)

def find_relevant_context(query, dataframe, k=3):
    """Finds the top 'k' most relevant text chunks from the knowledge base."""
    query_embedding_response = genai.embed_content(model='models/text-embedding-004', content=query, task_type="RETRIEVAL_QUERY")
    query_embedding = query_embedding_response['embedding']
    dot_products = np.dot(np.stack(dataframe['embedding']), query_embedding)
    top_k_indices = np.argsort(dot_products)[-k:][::-1]
    return "\n\n---\n\n".join(dataframe.iloc[top_k_indices]['text_for_search'].tolist())

def answer_question_from_text(question, dataframe):
    """Handles the text-based Q&A logic with our refined persona."""
    context = find_relevant_context(question, dataframe)
    prompt = f"**Persona:** You are a friendly AI Teacher... **Task & Rules:** Answer the user's question based on the 'Source Material'... **IMPORTANT:** Never refer to the source material directly... **Source Material:** {context} --- **User's Question:** {question}"
    response = generate_content_with_retry(models["flash"], prompt)
    return response.text

def analyze_handwritten_image(image, instruction):
    """Handles the image analysis logic with a focused task instruction."""
    prompt = f"You are an expert AI Math Teacher... **CRITICAL RULE:** Provide ONLY the step-by-step solution... **DO NOT** add any extra summary... **User's Instruction:** \"{instruction}\""
    response = generate_content_with_retry(models["pro_vision"], [prompt, image])
    return response.text

# --- These are the functions specifically for the Lecture Mode ---

@st.cache_data # Cache the list of chapters to avoid re-generating it constantly.
def get_chapter_list(dataframe):
    """Generates ONLY a list of chapter names for the selection menu."""
    print("📚 Generating chapter list...")
    model = models["flash"]
    full_text_sample = "\n".join(dataframe['text_for_search'].head(150).tolist())
    prompt = f"Your task is to act as a data extractor... Your ONLY job is to identify the chapter titles from the text sample... The output MUST be a single, valid JSON array of strings... TEXT SAMPLE TO ANALYZE:\n{full_text_sample}"
    try:
        response = generate_content_with_retry(model, prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response, strict=False)
    except Exception as e: return [f"Error generating chapters: {e}"]

@st.cache_data # Cache the topics for a given chapter.
def get_topic_list(chapter_name, dataframe):
    """Generates a list of topics ONLY for the selected chapter."""
    print(f" B Analyzing '{chapter_name}' to find its main topics...")
    model = models["flash"]
    source_material = find_relevant_context(chapter_name, dataframe, k=10)
    prompt = f"You are a curriculum expert. Analyze the source material for '{chapter_name}'. Your only task is to list the main topics. Output MUST be a single, valid JSON array..."
    try:
        response = generate_content_with_retry(model, [prompt.replace("{source_material}", source_material)])
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
    print(f"✍️ Generating lecture script for '{topic}'...")
    model = models["flash"]
    prompt = f"You have two jobs... PERSONA: You are a friendly AI Teacher... TASK: Create a lecture script... **CRITICAL RULES:** 1. Use <FORMULA:your_formula_here>... SOURCE MATERIAL: {source_material}"
    try:
        response = generate_content_with_retry(model, [prompt])
        raw_text = response.text.strip()
        # Use our robust hybrid parser
        try:
            return json.loads(raw_text.replace("```json", "").replace("```", ""), strict=False)
        except json.JSONDecodeError:
             spoken_texts = re.findall(r'\*\*spoken_text:\*\*\s*(.*?)(?=\n\*\*display_text|\Z)', raw_text, re.DOTALL)
             display_texts = re.findall(r'\*\*display_text:\*\*\s*(.*?)(?=\n\*\*spoken_text|\Z)', raw_text, re.DOTALL)
             if len(spoken_texts) > 0 and len(spoken_texts) == len(display_texts):
                 return [{"spoken_text": s.strip(), "display_text": d.strip()} for s, d in zip(spoken_texts, display_texts)]
             else: return None
    except Exception as e: return None

def convert_script_to_audio(script_parts, lecture_audio_folder):
    """Converts 'spoken_text' into MP3 files, processing formula placeholders."""
    print("🎙️ Converting script to audio files...")
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

# --- Section 5: Streamlit User Interface ---
st.title("🤖 AI Teacher Portal")

with st.sidebar:
    st.header("App Mode")
    app_mode = st.radio("Choose a feature:", ("❓ Textbook Q&A", "✍️ Homework Helper", "👩‍🏫 Teacher Lecture Mode"))

if app_mode == "❓ Textbook Q&A":
    st.header("Ask a Question from the Textbook")
    if models and df_embedded is not None and not df_embedded.empty:
        text_question = st.text_area("Enter your question here:", height=150)
        if st.button("Get Answer"):
            if text_question:
                with st.spinner("The AI Teacher is thinking..."):
                    response = answer_question_from_text(text_question, df_embedded)
                    st.success("Here is your answer:")
                    st.write(response)
            else: st.warning("Please enter a question.")
    else: st.error("Application not ready. Knowledge base may be missing.")

elif app_mode == "✍️ Homework Helper":
    st.header("Get Help with Your Homework")
    if models:
        uploaded_file = st.file_uploader("Upload an image of your work", type=["png", "jpg", "jpeg"])
        instruction = st.text_input("What should I do with this image?", placeholder="e.g., 'Solve for x'")
        if st.button("Analyze Image"):
            if uploaded_file and instruction:
                with st.spinner("The AI Teacher is analyzing your image..."):
                    image = Image.open(uploaded_file)
                    response = analyze_handwritten_image(image, instruction)
                    st.success("Here is the analysis:")
                    st.write(response)
            else: st.warning("Please upload an image and provide an instruction.")
    else: st.error("Application not ready. Check logs or secrets.")

elif app_mode == "👩‍🏫 Teacher Lecture Mode":
    st.header("Generate a Custom Audio Lecture")
    if models and df_embedded is not None and not df_embedded.empty:
        chapters = get_chapter_list(df_embedded)
        if chapters and "Error" not in chapters[0]:
            selected_chapter = st.selectbox("Step 1: Choose a Chapter", options=chapters)
            if selected_chapter:
                topics = get_topic_list(selected_chapter, df_embedded)
                if topics and "Error" not in topics[0]:
                    selected_topic = st.selectbox("Step 2: Choose a Topic", options=topics)
                    if st.button("Generate Lecture", key="generate_lecture"):
                        if selected_topic:
                            st.info(f"Preparing a lecture on: '{selected_topic}'")
                            with st.spinner("This will take a few minutes... The AI is writing the script and creating the audio..."):
                                lecture_asset_path = f"Lecture_Assets/{selected_topic.replace(' ', '_')}"
                                source_material = find_relevant_chunks(selected_topic, df_embedded, k=10)
                                lecture_script = generate_lecture_script(selected_topic, source_material)
                                if lecture_script:
                                    audio_files = convert_script_to_audio(lecture_script, lecture_asset_path)
                                    deliver_grouped_lecture(lecture_script, audio_files)
                                else: st.error("Could not generate the lecture script.")
                        else: st.warning("Please select a topic.")
                else: st.error("Could not load topics for this chapter.")
        else: st.error("Could not load chapters from the textbook.")
    else: st.error("Application not ready. Knowledge base may be missing.")
