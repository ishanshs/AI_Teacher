# app.py (The Definitive Version with All Features and Comments)

# --- Section 1: Import All Necessary Libraries ---
# This section brings in all the tools we need for our application.
import os
import pandas as pd
import numpy as np
import google.generativeai as genai
import streamlit as st # The main library for building our web app interface.
from PIL import Image # For handling image files.
from gtts import gTTS # For converting text to speech.
from mutagen.mp3 import MP3 # For reading audio file metadata (like duration).
import time # For creating pauses in our lecture delivery.
import json # For parsing structured data from the AI.
import re # For our robust backup parser.
from num2words import num2words # For correct number pronunciation.

# --- Section 2: Page Configuration ---
# This sets the title and icon that appear in the browser tab for a professional look.
# This should be the first Streamlit command in your script.
st.set_page_config(
    page_title="AI Teacher Portal",
    page_icon="🤖",
    layout="wide"
)

# --- Section 3: Authentication and Resource Loading ---
# This important function runs only ONCE when the app starts. The @st.cache_resource
# decorator is a powerful Streamlit feature that prevents this slow code from
# re-running every time a user clicks a button.
@st.cache_resource
def load_resources():
    """
    Configures the Google AI API and loads the knowledge base from the CSV file.
    This function is cached so it only runs once per session.
    """
    # Configure the Google AI API using the secret key from Hugging Face settings.
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        # st.error shows a prominent error message in the UI if the key is missing.
        st.error("GOOGLE_API_KEY secret not found. Please set it in your Hugging Face Space secrets.")
        return None, None
    try:
        genai.configure(api_key=api_key)
        print("✅ Google AI API configured successfully.")
    except Exception as e:
        st.error(f"Error configuring Google AI API: {e}")
        return None, None

    # Load the Gemini 1.5 Flash model, which we will use for all generative tasks.
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    print("✅ Gemini 1.5 Flash model loaded successfully.")

    # Load our pre-processed knowledge base from the CSV file.
    try:
        dataframe = pd.read_csv("knowledge_base.csv")
        # The 'embedding' column is read as a string, so we must convert it back to a list of floats.
        dataframe['embedding'] = dataframe['embedding'].apply(eval)
        print("✅ Knowledge base loaded successfully.")
        # Return both the AI model and the dataframe so the rest of the app can use them.
        return model, dataframe
    except FileNotFoundError:
        st.error("CRITICAL ERROR: 'knowledge_base.csv' not found. App cannot function.")
        return None, None

# Load the resources when the app starts. The results are stored in these global variables.
model, df_embedded = load_resources()


# --- Section 4: Core Logic Functions ---
# This section contains all the backend functions that perform the AI tasks.

def find_relevant_context(query, dataframe, k=3):
    """
    Finds the top 'k' most relevant text chunks from the knowledge base
    and combines them into a single block of context for the AI.
    """
    query_embedding_response = genai.embed_content(model='models/text-embedding-004', content=query, task_type="RETRIEVAL_QUERY")
    query_embedding = query_embedding_response['embedding']
    dot_products = np.dot(np.stack(dataframe['embedding']), query_embedding)
    top_k_indices = np.argsort(dot_products)[-k:][::-1]
    return "\n\n---\n\n".join(dataframe.iloc[top_k_indices]['text_for_search'].tolist())

def answer_question_from_text(question, dataframe):
    """Handles the text-based Q&A logic with our refined persona."""
    # Get a rich context by finding the top 3 relevant passages.
    context = find_relevant_context(question, dataframe, k=3)
    # Use our sophisticated prompt to get a natural, non-robotic answer.
    prompt = f"""
    **Your Persona:** You are a friendly, cheerful, and patient AI Teacher...
    **Your Task & Rules:** 1. Answer the user's question clearly... 2. Base your answer on the 'Source Material'... 3. **IMPORTANT:** Never refer to the source material directly...
    ---
    **Source Material:** {context}
    ---
    **User's Question:** {question}
    """
    response = model.generate_content(prompt)
    return response.text

def analyze_handwritten_image(image, instruction):
    """Handles the image analysis logic with a focused task instruction."""
    # Use the more powerful Pro model for its superior vision capabilities.
    vision_model = genai.GenerativeModel('gemini-1.5-pro-latest')
    prompt = f"""
    You are an expert AI Teacher. Your task is to follow the user's instruction precisely...
    **CRITICAL RULE:** Provide ONLY the direct answer... DO NOT add any extra summary or critique...
    **User's Instruction:** "{instruction}"
    """
    response = vision_model.generate_content([prompt, image])
    return response.text

def generate_chapter_list(dataframe):
    """Generates ONLY a list of chapter titles for the selection menu."""
    print("📚 Scanning textbook to identify chapters...")
    prompt = "Analyze this text sample... Your ONLY job is to identify the chapter titles... Output MUST be a single, valid JSON array..."
    full_text_sample = "\n".join(dataframe['text_for_search'].head(150).tolist())
    response = model.generate_content(prompt.replace("{full_text_sample}", full_text_sample))
    try:
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response, strict=False)
    except Exception as e: return [f"Error: {e}"]

def generate_topics_for_chapter(chapter_name, dataframe):
    """Generates a list of topics ONLY for the selected chapter."""
    print(f" B Analyzing '{chapter_name}' to find topics...")
    source_material = find_relevant_context(chapter_name, dataframe, k=10)
    prompt = f"You are a curriculum expert... list the main topics for '{chapter_name}'... Output MUST be a single, valid JSON array..."
    response = model.generate_content(prompt.replace("{source_material}", source_material))
    try:
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response, strict=False)
    except Exception as e: return [f"Error: {e}"]

def verbalize_formula(formula):
    """Programmatically converts a formula string into a verbal explanation."""
    pronunciation_map = {'+': ' plus ', '-': ' minus ', 'x': ' multiplied by ', '*': ' multiplied by ', '/': ' divided by ', '=': ' equals '}
    tokens = re.findall(r'(\d+\.?\d*|[a-zA-Z]+|.)', formula)
    verbalized_tokens = [num2words(int(t)) if t.isdigit() else pronunciation_map.get(t, f' {t} ') for t in tokens]
    return ' '.join(verbalized_tokens).strip().replace('  ', ' ')

def generate_lecture_script(topic, source_material):
    """Uses Gemini to create a lecture script with formula placeholders."""
    print(f"✍️ Generating lecture script for '{topic}'...")
    prompt = f"You have two jobs... PERSONA: You are a friendly AI Teacher... TASK: Create a lecture script... Your entire output MUST be a single, valid JSON array... CRITICAL RULES: 1. Use <FORMULA:your_formula_here>... SOURCE MATERIAL: {source_material}"
    response = model.generate_content(prompt)
    try:
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response, strict=False)
    except Exception: return None

def convert_script_to_audio(script_parts, lecture_audio_folder):
    """Converts 'spoken_text' into MP3 files, processing formula placeholders."""
    print("🎙️ Converting script to audio files...")
    os.makedirs(lecture_audio_folder, exist_ok=True)
    audio_files = []
    for i, part in enumerate(script_parts):
        text_to_speak = part['spoken_text']
        placeholders = re.findall(r'<FORMULA:(.*?)>', text_to_speak)
        for formula in placeholders:
            verbal_formula = verbalize_formula(formula)
            text_to_speak = text_to_speak.replace(f'<FORMULA:{formula}>', verbal_formula)
        tts = gTTS(text=text_to_speak, lang='en', tld='co.in', slow=False)
        file_path = os.path.join(lecture_audio_folder, f"part_{i}.mp3")
        tts.save(file_path)
        audio_files.append(file_path)
    return audio_files

# --- Section 5: Streamlit User Interface ---
# This section creates all the visual elements of your web app.

st.title("🤖 AI Teacher Portal")

# Use the sidebar for navigation. The selected option is stored in the 'app_mode' variable.
with st.sidebar:
    st.header("App Mode")
    # --- THIS IS THE CORRECTED WIDGET ---
    # We now include "Teacher Lecture Mode" as the third option.
    app_mode = st.radio("Choose a feature:", ("❓ Textbook Q&A", "✍️ Homework Helper", "👩‍🏫 Teacher Lecture Mode"))

# --- UI for Textbook Q&A Mode ---
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

# --- UI for Homework Helper Mode ---
elif app_mode == "✍️ Homework Helper":
    st.header("Get Help with Your Homework")
    if model:
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
    else:
        st.error("Application is not ready. Please check the logs or secrets.")

# --- NEW: UI for Teacher Lecture Mode ---
elif app_mode == "👩‍🏫 Teacher Lecture Mode":
    st.header("Generate a Custom Audio Lecture")
    if model and df_embedded is not None and not df_embedded.empty:
        
        # Step 1: Select a Chapter
        # We cache the chapter list to avoid re-generating it on every interaction.
        @st.cache_data
        def get_chapter_list():
            return generate_chapter_list(df_embedded)

        chapters = get_chapter_list()
        selected_chapter = st.selectbox("Step 1: Choose a Chapter", options=chapters)
        
        if selected_chapter:
            # Step 2: Select a Topic from that Chapter
            # We cache the topics for the selected chapter.
            @st.cache_data
            def get_topic_list(chapter):
                return generate_topics_for_chapter(chapter, df_embedded)

            topics = get_topic_list(selected_chapter)
            selected_topic = st.selectbox("Step 2: Choose a Topic", options=topics)
            
            # Step 3: Generate the Lecture
            if st.button("Generate Lecture", key="generate_lecture"):
                if selected_topic:
                    st.info(f"Preparing a lecture on: '{selected_topic}'")
                    with st.spinner("This will take a few minutes... The AI is writing the script and creating the audio..."):
                        # --- The Lecture Generation Pipeline ---
                        lecture_asset_path = f"Lecture_Assets/{selected_topic.replace(' ', '_')}"
                        source_material = find_relevant_chunks(selected_topic, df_embedded)
                        lecture_script = generate_lecture_script(selected_topic, source_material)
                        
                        if lecture_script:
                            audio_files = convert_script_to_audio(lecture_script, lecture_asset_path)
                            
                            # Display the final lecture
                            st.success("Your lecture is ready! Press play on each part.")
                            for i, part in enumerate(lecture_script):
                                if i < len(audio_files):
                                    st.markdown("---")
                                    st.write(f"**Note:** {part['display_text']}")
                                    st.audio(audio_files[i])
                        else:
                            st.error("Could not generate the lecture script. The AI may have returned an unexpected format.")
                else:
                    st.warning("Please select a topic.")
    else:
        st.error("Application is not ready. Please check the logs or secrets.")
