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
import tempfile # Import the tempfile library

# --- Section 2: Page Configuration ---
st.set_page_config(
    page_title="AI Teacher Portal",
    page_icon="🤖",
    layout="wide"
)

# --- Section 3: Resource Loading (Cached for Performance) ---
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
# All functions are designed to receive a configured 'model' object.

@retry(retry=retry_if_exception_type((TooManyRequests, ResourceExhausted)), wait=wait_fixed(60), stop=stop_after_attempt(3))
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
    """Handles the text-based Q&A logic with the refined persona prompt."""
    context = find_relevant_context(question, dataframe)
    prompt = f"""
    **Persona:**
    You are a friendly, cheerful, and patient AI Teacher. Imagine you are an older, knowledgeable family member explaining a concept simply and encouragingly.
    **Task & Rules:**
    1. Your primary task is to answer the user's question clearly and conversationally.
    2. You MUST base your answer strictly on the provided 'Source Material'.
    3. **IMPORTANT:** Never refer to the source material directly. Answer naturally as if the knowledge is your own.
    ---
    **Source Material:**
    {context}
    ---
    **User's Question:**
    {question}
    """
    response = generate_content_with_retry(model, prompt)
    return response.text

def analyze_handwritten_image(image, instruction, model):
    """Handles the image analysis logic with the specific step-by-step prompt."""
    prompt = f"""
    You are an expert AI Math Teacher. Your task is to follow the user's instruction precisely based on the provided image.
    **CRITICAL RULES:**
    1.  **Show Your Work:** Your primary goal is to provide a detailed, step-by-step derivation of the solution.
    2.  **Accuracy First:** Double-check your mathematical reasoning.
    3.  **Be Direct:** Provide ONLY the step-by-step solution.
    **PERFECT EXAMPLE OF OUTPUT:**
    Of course! Here is the step-by-step solution:
    Step 1: First, I will rearrange the first equation to solve for y.
    Given: 2x + y = 5
    y = 5 - 2x
    Step 2: Now, I will substitute this expression for y into the second equation.
    Given: 3x - 2y = 4
    3x - 2(5 - 2x) = 4
    Step 3: I will now solve for x.
    3x - 10 + 4x = 4
    7x = 14
    x = 2
    Step 4: Finally, I will substitute the value of x back into the expression for y.
    y = 5 - 2(2)
    y = 1
    **Final Answer:** The solution is x = 2 and y = 1.
    ---
    **User's Instruction:** "{instruction}"
    """
    response = generate_content_with_retry(model, [prompt, image])
    return response.text

# --- These are the functions specifically for the Lecture Mode ---

@st.cache_data
def get_chapter_list(_dataframe, _model):
    """Generates an ordered list of chapter names for the selection menu."""
    st.write("📚 Analyzing textbook to identify chapters...")
    full_text_sample = "\n".join(_dataframe['text_for_search'].head(200).tolist())
    # This prompt now uses the "Contrastive" technique to be more precise.
    prompt = f"""
    You are a data extraction expert. Your only task is to analyze the provided text and extract the high-level chapter titles in the correct sequence.
    - A 'Chapter' is a major section like "Chapter 1: Integers" or "Chapter 5: Lines and Angles".
    - A 'Chapter' is NOT a smaller topic like "Properties of Addition" or "Adjacent Angles".
    - Your output MUST be a single, valid JSON array of strings. Do NOT add any text before or after the JSON array.
    TEXT TO ANALYZE: {full_text_sample}
    """
    try:
        response = generate_content_with_retry(_model, prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response, strict=False)
    except Exception as e:
        return {"error": f"Failed to generate or parse chapter list. Details: {str(e)}"}

@st.cache_data
def get_topic_list(chapter_name, _dataframe, _model):
    """Generates a list of topics ONLY for the selected chapter."""
    st.write(f" B Analyzing '{chapter_name}' to find its main topics...")
    source_material = find_relevant_context(chapter_name, _dataframe, k=10)
    prompt = f"""
    You are a curriculum expert. Analyze the source material for '{chapter_name}'. List the main topics covered in this specific chapter.
    - A 'Topic' is a main idea within a chapter, like 'Properties of Addition' or 'Solving Equations'.
    - It is NOT a tiny sub-topic or a single definition like 'What is an integer?'.
    - Your output MUST be a single, valid JSON array of strings.
    SOURCE: {source_material}
    """
    try:
        response = generate_content_with_retry(_model, prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response, strict=False)
    except Exception as e:
        return {"error": f"Failed to generate or parse topics. Details: {str(e)}"}

def generate_lecture_script(topic, source_material, model):
    """Uses Gemini to create a lecture script with formula placeholders."""
    print(f"✍️ Generating lecture script for '{topic}'...")
    prompt = f"""
    You have two jobs. First, adopt a persona. Second, complete a task.
    **PERSONA:** You are a friendly, enthusiastic AI Teacher...
    **TASK:** Create a lecture script... Your output MUST be a single, valid JSON array...
    **CRITICAL RULES:**
    1.  **Formula Placeholder:** When you need to say a formula, use the format `<FORMULA:your_formula_here>`.
    2.  **Synchronized Examples:** If you use a `<FORMULA:...>` placeholder, the formula inside MUST be written in the 'display_text'.
    SOURCE MATERIAL: {source_material}
    """
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
    """Converts 'spoken_text' into MP3 files, with robust error handling."""
    print("🎙️ Converting script to audio files...")
    # This function now correctly creates the directory in the approved /tmp/ location.
    os.makedirs(lecture_audio_folder, exist_ok=True)
    audio_files = []
    for i, part in enumerate(script_parts):
        try:
            text_to_speak = part['spoken_text']
            placeholders = re.findall(r'<FORMULA:(.*?)>', text_to_speak)
            for formula in placeholders:
                text_to_speak = text_to_speak.replace(f'<FORMULA:{formula}>', verbalize_formula(formula))
            
            if not text_to_speak.strip():
                print(f"⚠️ Skipping audio generation for part {i} because text is empty.")
                audio_files.append(None)
                continue

            tts = gTTS(text=text_to_speak, lang='en', tld='co.in', slow=False)
            file_path = os.path.join(lecture_audio_folder, f"part_{i}.mp3")
            tts.save(file_path)
            audio_files.append(file_path)
        except Exception as e:
            print(f"Could not convert text to speech for part {i}: {e}")
            audio_files.append(None)
    return audio_files

def deliver_grouped_lecture(script_parts, audio_files):
    """Delivers the lecture, handling parts where audio generation might have failed."""
    st.success("Your lecture is ready! Press play on each part to listen.")
    for i, part in enumerate(script_parts):
        st.markdown("---")
        st.write(f"**Note:** {part['display_text']}")
        if i < len(audio_files) and audio_files[i] is not None:
            st.audio(audio_files[i])
        else:
            st.warning("Audio could not be generated for this part.")
    st.markdown("---")
    st.balloons()
    st.success("🎉 End of Lecture! Great job!")


# --- Section 5: Streamlit User Interface ---
st.title("🤖 AI Teacher Portal")

with st.sidebar:
    st.header("🔑 API Configuration")
    api_key_input = st.text_input("Enter your Google AI API Key", type="password", help="Get your free API key from Google AI Studio.")
    st.header("App Mode")
    app_mode = st.radio("Choose a feature:", ("❓ Textbook Q&A", "✍️ Homework Helper", "👩‍🏫 Teacher Lecture Mode"))

if df_embedded is None:
    st.error("CRITICAL ERROR: 'knowledge_base.csv' not found. Application cannot start.")
    st.stop()
elif not api_key_input:
    st.info("👋 Welcome! Please enter your Google AI API Key in the sidebar to activate the AI Teacher.")
    st.stop()
else:
    try:
        genai.configure(api_key=api_key_input)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
    except Exception as e:
        st.error(f"Failed to initialize the AI model. Please check your API key. Error: {e}")
        st.stop()
    
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
    
    elif app_mode == "✍️ Homework Helper":
        st.header("Get Help with Your Homework")
        uploaded_file = st.file_uploader("Upload an image of your work", type=["png", "jpg", "jpeg"])
        instruction = st.text_input("What should I do with this image?", placeholder="e.g., 'Solve these equations for x and y'")
        if st.button("Analyze Image"):
            if uploaded_file and instruction:
                with st.spinner("The AI Teacher is analyzing your image..."):
                    image = Image.open(uploaded_file)
                    response = analyze_handwritten_image(image, instruction, model)
                    st.success("Here is the analysis:")
                    st.write(response)
            else: st.warning("Please upload an image and provide an instruction.")

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
                                    status.write("🎙️ Creating audio files...")
                                    # This is the corrected line that uses a temporary directory.
                                    lecture_asset_path = os.path.join(tempfile.gettempdir(), f"Lecture_Assets_{selected_topic.replace(' ', '_')}")
                                    audio_files = convert_script_to_audio(lecture_script, lecture_asset_path)
                                    status.update(label="Lecture ready!", state="complete", expanded=False)
                                    deliver_grouped_lecture(lecture_script, audio_files)
                                else:
                                    status.update(label="Error!", state="error")
                                    st.error("Could not generate the lecture script.")
                        else: st.warning("Please select a topic.")
