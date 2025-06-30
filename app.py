# app.py (The Definitive, Sharable Version with All 3 Modes)

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

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Teacher Portal",
    page_icon="🤖",
    layout="wide"
)

# --- Resource Loading (runs once and is cached) ---
# We only load the knowledge base at startup. The AI models will be initialized
# on-demand using the user's provided API key.
@st.cache_resource
def load_knowledge_base():
    """Loads the knowledge base from the CSV file."""
    try:
        dataframe = pd.read_csv("knowledge_base.csv")
        dataframe['embedding'] = dataframe['embedding'].apply(eval)
        print("✅ Knowledge base loaded successfully.")
        return dataframe
    except FileNotFoundError:
        return None

df_embedded = load_knowledge_base()

# --- Core Logic Functions (Adapted to use a user-provided API key) ---

def configure_genai(api_key):
    """Configures the Google AI SDK with the provided key."""
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Failed to configure API. Please ensure your key is valid. Error: {e}")
        return False

def find_relevant_context(query, dataframe, api_key, k=3):
    """Finds the top 'k' most relevant text chunks."""
    configure_genai(api_key)
    query_embedding_response = genai.embed_content(model='models/text-embedding-004', content=query, task_type="RETRIEVAL_QUERY")
    query_embedding = query_embedding_response['embedding']
    dot_products = np.dot(np.stack(dataframe['embedding']), query_embedding)
    top_k_indices = np.argsort(dot_products)[-k:][::-1]
    return "\n\n---\n\n".join(dataframe.iloc[top_k_indices]['text_for_search'].tolist())

def answer_question_from_text(question, dataframe, api_key):
    """Handles the text-based Q&A logic."""
    context = find_relevant_context(question, dataframe, api_key)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    prompt = f"**Persona:** You are a friendly AI Teacher... **Task:** Answer the user's question... **Source Material:** {context} --- **Question:** {question}"
    response = model.generate_content(prompt)
    return response.text

def analyze_handwritten_image(image, instruction, api_key):
    """Handles the image analysis logic."""
    configure_genai(api_key)
    vision_model = genai.GenerativeModel('gemini-1.5-pro-latest')
    prompt = f"You are an expert AI Math Teacher... **Rule:** Provide ONLY the step-by-step solution... **Instruction:** \"{instruction}\""
    response = vision_model.generate_content([prompt, image])
    return response.text

@st.cache_data
def get_chapter_list(_dataframe, api_key):
    """Generates ONLY a list of chapter names."""
    st.write("📚 Analyzing textbook to identify chapters...")
    configure_genai(api_key)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    full_text_sample = "\n".join(_dataframe['text_for_search'].head(150).tolist())
    prompt = f"Your ONLY job is to analyze the text and extract chapter titles. Output MUST be a single, valid JSON array of strings. TEXT TO ANALYZE: {full_text_sample}"
    try:
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response, strict=False)
    except Exception as e:
        return {"error": f"Failed to generate or parse chapter list. Details: {str(e)}"}

@st.cache_data
def get_topic_list(chapter_name, _dataframe, api_key):
    """Generates a list of topics ONLY for the selected chapter."""
    st.write(f" B Analyzing '{chapter_name}' to find its main topics...")
    configure_genai(api_key)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    source_material = find_relevant_context(chapter_name, _dataframe, api_key, k=10)
    prompt = f"You are a curriculum expert. Analyze the source material for '{chapter_name}'. List the main topics. Output MUST be a single, valid JSON array. SOURCE: {source_material}"
    try:
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response, strict=False)
    except Exception as e:
        return {"error": f"Failed to generate or parse topics. Details: {str(e)}"}

# ... (Placeholders for other lecture functions - can be filled later) ...
def generate_lecture_script(topic, source_material, api_key):
    st.info("Script generation is a placeholder in this version.")
    return [{"display_text": f"This is a placeholder script for the topic: {topic}"}]
def convert_script_to_audio(script_parts, folder):
    st.info("Audio generation is a placeholder in this version.")
    return []
def deliver_grouped_lecture(script, audio, size):
    st.success("Lecture script generated! Audio playback is the next step.")
    for part in script:
        st.markdown("---")
        st.write(f"**Display Note:** {part['display_text']}")


# --- Streamlit User Interface ---
st.title("🤖 AI Teacher Portal")

with st.sidebar:
    st.header("🔑 API Configuration")
    api_key_input = st.text_input(
        "Enter your Google AI API Key", type="password", help="Get your free API key from Google AI Studio."
    )
    st.header("App Mode")
    app_mode = st.radio(
        "Choose a feature:",
        ("❓ Textbook Q&A", "✍️ Homework Helper", "👩‍🏫 Teacher Lecture Mode")
    )

if df_embedded is None:
    st.error("CRITICAL ERROR: 'knowledge_base.csv' not found. Application cannot start.")
    st.stop()
elif not api_key_input:
    st.info("👋 Welcome! Please enter your Google AI API Key in the sidebar to activate the AI Teacher.")
    st.stop()
else:
    # If we have the dataframe and an API key, build the selected UI mode.
    if app_mode == "❓ Textbook Q&A":
        st.header("Ask a Question from the Textbook")
        text_question = st.text_area("Enter your question here:", height=150)
        if st.button("Get Answer"):
            if text_question:
                with st.spinner("The AI Teacher is thinking..."):
                    response = answer_question_from_text(text_question, df_embedded, api_key_input)
                    st.success("Here is your answer:")
                    st.write(response)
            else: st.warning("Please enter a question.")
    
    elif app_mode == "✍️ Homework Helper":
        st.header("Get Help with Your Homework")
        uploaded_file = st.file_uploader("Upload an image of your work", type=["png", "jpg", "jpeg"])
        instruction = st.text_input("What should I do with this image?", placeholder="e.g., 'Solve for x'")
        if st.button("Analyze Image"):
            if uploaded_file and instruction:
                with st.spinner("The AI Teacher is analyzing your image..."):
                    image = Image.open(uploaded_file)
                    response = analyze_handwritten_image(image, instruction, api_key_input)
                    st.success("Here is the analysis:")
                    st.write(response)
            else: st.warning("Please upload an image and provide an instruction.")

    elif app_mode == "👩‍🏫 Teacher Lecture Mode":
        st.header("Generate a Custom Audio Lecture")
        chapters = get_chapter_list(df_embedded, api_key_input)
        
        if isinstance(chapters, dict) and "error" in chapters:
            st.error(f"An error occurred while loading chapters:\n\n{chapters['error']}")
        elif not chapters:
            st.warning("The AI could not identify any chapters from the provided text.")
        else:
            selected_chapter = st.selectbox("Step 1: Choose a Chapter", options=chapters)
            if selected_chapter:
                topics = get_topic_list(selected_chapter, df_embedded, api_key_input)
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
                                source_material = find_relevant_context(selected_topic, df_embedded, api_key_input, k=10)
                                status.write("✍️ Writing the lecture script...")
                                lecture_script = generate_lecture_script(selected_topic, source_material, api_key_input)
                                if lecture_script:
                                    status.update(label="Lecture Ready!", state="complete")
                                    deliver_grouped_lecture(lecture_script, [], 3) # Audio part is placeholder for now
                                else:
                                    status.update(label="Error!", state="error")
                                    st.error("Could not generate the lecture script.")
                        else: st.warning("Please select a topic.")
