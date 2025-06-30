# app.py (The Definitive, Sharable, Fully-Functional Version)

# --- Section 1: Import All Necessary Libraries ---
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
from google.api_core.exceptions import TooManyRequests, ResourceExhausted
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

# --- Section 2: Page Configuration ---
st.set_page_config(page_title="AI Teacher Portal", page_icon="🤖", layout="wide")

# --- Section 3: Resource Loading (Cached for Performance) ---
# We only load the knowledge base at startup. The AI models are loaded later.
@st.cache_resource
def load_knowledge_base():
    """Loads the knowledge base from the CSV file. Returns None if it fails."""
    try:
        dataframe = pd.read_csv("knowledge_base.csv")
        dataframe['embedding'] = dataframe['embedding'].apply(eval)
        print("✅ Knowledge base loaded successfully.")
        return dataframe
    except FileNotFoundError:
        return None

df_embedded = load_knowledge_base()

# --- Section 4: Core Logic Functions ---
# These functions now accept a 'model' object, ensuring we only use the one we intend to.

# This function makes our API calls resilient to temporary rate limit errors.
@retry(retry=retry_if_exception_type(TooManyRequests), wait=wait_fixed(60), stop=stop_after_attempt(3))
def generate_content_with_retry(model, prompt):
    """A resilient wrapper for the 'generate_content' API call."""
    return model.generate_content(prompt)

def find_relevant_context(query, dataframe, k=5):
    """Finds the top 'k' most relevant text chunks from the knowledge base."""
    query_embedding_response = genai.embed_content(model='models/text-embedding-004', content=query, task_type="RETRIEVAL_QUERY")
    query_embedding = query_embedding_response['embedding']
    dot_products = np.dot(np.stack(dataframe['embedding']), query_embedding)
    top_k_indices = np.argsort(dot_products)[-k:][::-1]
    return "\n\n---\n\n".join(dataframe.iloc[top_k_indices]['text_for_search'].tolist())

def answer_question_from_text(question, dataframe, model):
    """Handles the text-based Q&A logic with our refined persona."""
    context = find_relevant_context(question, dataframe)
    prompt = f"**Persona:** You are a friendly, patient AI Teacher... **Task:** Answer the user's question clearly... **Source Material:** {context} --- **User's Question:** {question}"
    response = generate_content_with_retry(model, prompt)
    return response.text

def analyze_handwritten_image(image, instruction, model):
    """Handles the image analysis logic using the provided Flash model."""
    prompt = f"You are an expert AI Math Teacher... **CRITICAL RULE:** Provide ONLY the step-by-step solution... **Instruction:** \"{instruction}\""
    response = generate_content_with_retry(model, [prompt, image])
    return response.text

# --- These are the functions specifically for the Lecture Mode ---

@st.cache_data
def get_chapter_list(_dataframe, model):
    """Generates ONLY a list of chapter names for the selection menu."""
    st.write("📚 Analyzing textbook to identify chapters...")
    full_text_sample = "\n".join(_dataframe['text_for_search'].head(200).tolist())
    prompt = f"Your ONLY job is to analyze the text and extract chapter titles. Output MUST be a single, valid JSON array of strings. TEXT TO ANALYZE: {full_text_sample}"
    try:
        response = generate_content_with_retry(model, prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response, strict=False)
    except Exception as e:
        return {"error": f"Failed to generate or parse chapter list. Details: {str(e)}"}

@st.cache_data
def get_topic_list(chapter_name, _dataframe, model):
    """Generates a list of topics ONLY for the selected chapter."""
    st.write(f" B Analyzing '{chapter_name}' to find its main topics...")
    source_material = find_relevant_context(chapter_name, _dataframe, k=10)
    prompt = f"You are a curriculum expert. Analyze the source material for '{chapter_name}'. List the main topics. Output MUST be a single, valid JSON array. SOURCE: {source_material}"
    try:
        response = generate_content_with_retry(model, prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response, strict=False)
    except Exception as e:
        return {"error": f"Failed to generate or parse topics. Details: {str(e)}"}

def generate_lecture_script(topic, source_material, model):
    """Uses Gemini to create a lecture script with formula placeholders."""
    prompt = f"You have two jobs... PERSONA: You are a friendly AI Teacher... TASK: Create a lecture script with 'spoken_text' and 'display_text'... PERFECT EXAMPLE: [{{'spoken_text': 'Let's solve <FORMULA:2x-3=5>', 'display_text': 'Example: 2x-3=5'}}]... SOURCE MATERIAL: {source_material}"
    try:
        response = generate_content_with_retry(model, prompt)
        raw_text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(raw_text, strict=False)
    except Exception as e:
        print(f"❌ An error occurred during script generation: {e}")
        return None

def verbalize_formula(formula):
    """Programmatically converts a formula string into a verbal explanation."""
    pronunciation_map = {'+': ' plus ', '-': ' minus ', 'x': ' times ', '*': ' times ', '/': ' divided by ', '=': ' equals '}
    tokens = re.findall(r'(\d+\.?\d*|[a-zA-Z]+|.)', formula)
    verbalized_tokens = [num2words(float(t)) if t.replace('.', '', 1).isdigit() else pronunciation_map.get(t, f' {t} ') for t in tokens]
    return ' '.join(verbalized_tokens).strip().replace('  ', ' ')

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

def deliver_grouped_lecture(script_parts, audio_files):
    """Delivers the lecture in digestible, manually-playable parts."""
    st.success("Your lecture is ready! Press play on each part to listen.")
    for i, part in enumerate(script_parts):
        st.markdown("---")
        st.write(f"**Note:** {part['display_text']}")
        if i < len(audio_files):
            st.audio(audio_files[i])
    st.markdown("---")
    st.balloons()
    st.success("🎉 End of Lecture! Great job!")


# --- Section 5: Streamlit User Interface ---
st.title("🤖 AI Teacher Portal")

with st.sidebar:
    st.header("🔑 API Configuration")
    # This input field is now the source of truth for the API key.
    api_key_input = st.text_input("Enter your Google AI API Key", type="password", help="Get your free API key from Google AI Studio.")
    st.header("App Mode")
    app_mode = st.radio("Choose a feature:", ("❓ Textbook Q&A", "✍️ Homework Helper", "👩‍🏫 Teacher Lecture Mode"))

# Master safety check to ensure resources are available before building the UI.
if df_embedded is None:
    st.error("CRITICAL ERROR: 'knowledge_base.csv' not found. The application cannot start.")
    st.stop()
elif not api_key_input:
    st.info("👋 Welcome! Please enter your Google AI API Key in the sidebar to activate the AI Teacher.")
    st.stop()
else:
    # If we have the dataframe and an API key, initialize the model and build the UI.
    try:
        # --- This is the single point where the model is initialized ---
        genai.configure(api_key=api_key_input)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
    except Exception as e:
        st.error(f"Failed to initialize the AI model. Please check your API key. Error: {e}")
        st.stop()
    
    # --- UI for Textbook Q&A Mode ---
    if app_mode == "❓ Textbook Q&A":
        st.header("Ask a Question from the Textbook")
        text_question = st.text_area("Enter your question here:", height=150)
        if st.button("Get Answer"):
            if text_question:
                with st.spinner("The AI Teacher is thinking..."):
                    response = answer_question_from_text(text_question, df_embedded, model)
                    st.success("Here is your answer:")
                    st.write(response)
            else: st.warning("Please enter a question.")
    
    # --- UI for Homework Helper Mode ---
    elif app_mode == "✍️ Homework Helper":
        st.header("Get Help with Your Homework")
        uploaded_file = st.file_uploader("Upload an image of your work", type=["png", "jpg", "jpeg"])
        instruction = st.text_input("What should I do with this image?", placeholder="e.g., 'Solve for x'")
        if st.button("Analyze Image"):
            if uploaded_file and instruction:
                with st.spinner("The AI Teacher is analyzing your image..."):
                    image = Image.open(uploaded_file)
                    # We pass the same 'model' object to this function.
                    response = analyze_handwritten_image(image, instruction, model)
                    st.success("Here is the analysis:")
                    st.write(response)
            else: st.warning("Please upload an image and provide an instruction.")

    # --- UI for Teacher Lecture Mode ---
    elif app_mode == "👩‍🏫 Teacher Lecture Mode":
        st.header("Generate a Custom Audio Lecture")
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
                            with st.status(f"Preparing lecture on '{selected_topic}'...", expanded=True) as status:
                                status.write("Gathering relevant material...")
                                source_material = find_relevant_context(selected_topic, df_embedded, k=10)
                                status.write("✍️ Writing the lecture script...")
                                lecture_script = generate_lecture_script(selected_topic, source_material, model)
                                if lecture_script:
                                    status.write("🎙️ Creating audio files for the lecture...")
                                    lecture_asset_path = f"Lecture_Assets/{selected_topic.replace(' ', '_')}"
                                    audio_files = convert_script_to_audio(lecture_script, lecture_asset_path)
                                    status.update(label="Lecture ready!", state="complete", expanded=False)
                                    deliver_grouped_lecture(lecture_script, audio_files)
                                else:
                                    status.update(label="Error!", state="error")
                                    st.error("Could not generate the lecture script.")
                        else: st.warning("Please select a topic.")
