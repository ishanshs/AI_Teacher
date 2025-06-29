# =================================================================
# SECTION 1: ALL LIBRARY IMPORTS
# =================================================================
# Here, we import all the libraries our application needs to function.
# We've combined the imports from all our previous steps.
import os
import io
import json
import re
import time
import numpy as np
import pandas as pd
from gtts import gTTS
from num2words import num2words
import google.generativeai as genai
from pypdf import PdfReader
from PIL import Image
from google.api_core.exceptions import TooManyRequests
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

# =================================================================
# SECTION 2: API KEY CONFIGURATION
# =================================================================
# This is where we configure our application to use the Google Gemini API.
# IMPORTANT: In a real application, this key should be stored securely
# and not written directly in the code. We will handle this later using
# Hugging Face secrets.
try:
    # This line will be adapted in the hosting environment to read a secret key.
    genai.configure(api_key="YOUR_GOOGLE_API_KEY_HERE")
    print("✅ Google AI API configured successfully.")
except Exception as e:
    print(f"❌ Error configuring Google AI API: {e}")


# =================================================================
# SECTION 3: CORE HELPER FUNCTIONS (FROM ALL STEPS)
# =================================================================
# This section contains all the working functions we developed and refined
# in our Colab notebook, from PDF processing to lecture generation.

# --- Resilient API Call Function ---
def print_waiting_message(retry_state):
    """A helper function that prints a message when our app is forced to wait."""
    print(f"Rate limit exceeded. Waiting 60 seconds before retry attempt {retry_state.attempt_number}...")

@retry(retry=retry_if_exception_type(TooManyRequests), wait=wait_fixed(60), stop=stop_after_attempt(3), before_sleep=print_waiting_message)
def generate_content_with_retry(model, prompt_list):
    """This function safely makes API calls to Gemini, protected by our retry logic."""
    return model.generate_content(prompt_list)

# --- PDF Processing Functions (from Step 1) ---
def get_pdf_page_images(pdf_file_path):
    """Converts a PDF file into a list of PIL Image objects."""
    # This function is not used in the final app but is kept for completeness.
    # The final app will use pre-processed data.
    pass

def extract_text_from_page(page):
    """Extracts text from a single PDF page."""
    return page.extract_text()

# --- Embedding and RAG Functions (from Steps 2, 3, 4) ---
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def embed_with_retry(model, content):
    """Resilient wrapper for creating embeddings."""
    return genai.embed_content(
        model=model,
        content=content,
        task_type="RETRIEVAL_DOCUMENT"
    )

def find_relevant_chunks(query, dataframe, k=5):
    """Finds the top 'k' most relevant text chunks for a given query."""
    model = 'models/text-embedding-004'
    query_embedding_response = embed_with_retry(model=model, content=query)
    query_embedding = query_embedding_response['embedding']
    dot_products = np.dot(np.stack(dataframe['embedding']), query_embedding)
    top_k_indices = np.argsort(dot_products)[-k:]
    relevant_chunks = dataframe.iloc[top_k_indices[::-1]]
    # We return the text and the source page number for context.
    context = "\n\n".join(relevant_chunks['text_for_search'].tolist())
    source_pages = ", ".join(str(x) for x in relevant_chunks['page_number'].unique())
    return context, source_pages

def answer_question(question, image_input, dataframe):
    """Answers a question based on text input or an image, using RAG."""
    model_text = genai.GenerativeModel('gemini-1.5-flash-latest')
    model_vision = genai.GenerativeModel('gemini-pro-vision') # For image analysis

    if image_input is not None:
        # Handle image-based questions
        print("Analysing the uploaded image...")
        response = model_vision.generate_content(["Please explain the concepts in this image based on my knowledge base.", image_input])
        return response.text
    else:
        # Handle text-based questions using our RAG pipeline
        print("Finding relevant context for the text question...")
        context, source_pages = find_relevant_chunks(question, dataframe)
        prompt = f"""
        You are the AI Teacher. Answer the following question based ONLY on the provided source material.
        Your answer should be helpful and clear. After your answer, cite the page numbers you used.

        Question: {question}

        Source Material:
        {context}
        """
        response = model_text.generate_content(prompt)
        return f"{response.text}\n\n(Source: Pages {source_pages})"

# --- Lecture Generation Functions (from Step 5) ---
def generate_chapter_list(dataframe):
    """Scans the textbook to get a reliable list of chapter titles."""
    print("📚 Scanning the textbook to identify chapters...")
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    full_text_sample = "\n".join(dataframe['text_for_search'].head(150).tolist())
    prompt = f"Your task is to act as a data extractor... Your output MUST be a single, valid JSON array... TEXT SAMPLE: {full_text_sample}"
    try:
        response = generate_content_with_retry(model, [prompt])
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_response, strict=False)
    except Exception as e:
        print(f"❌ Could not generate or parse the chapter list: {e}")
        return []

def generate_topics_for_chapter(chapter_name, dataframe):
    """Generates a list of topics for a specifically chosen chapter."""
    print(f" B Analyzing '{chapter_name}' to find its main topics...")
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    source_material = find_relevant_chunks(chapter_name, dataframe, k=10)[0] # We only need the text context here
    prompt = f"You are a curriculum expert... Your only task is to identify and list the main topics... Your output MUST be a single, valid JSON array... SOURCE MATERIAL: {source_material}"
    try:
        response = generate_content_with_retry(model, [prompt])
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_response, strict=False)
    except Exception as e:
        print(f"❌ Could not generate topics for '{chapter_name}': {e}")
        return []

def verbalize_formula(formula):
    """Programmatically converts a formula string into a verbal explanation."""
    pronunciation_map = {'+': ' plus ', '-': ' minus ', 'x': ' multiplied by ', '*': ' multiplied by ', '/': ' divided by ', '=': ' equals '}
    tokens = re.findall(r'(\d+\.?\d*|[a-zA-Z]+|.)', formula)
    verbalized_tokens = [num2words(int(t)) if t.isdigit() else pronunciation_map.get(t, f' {t} ') for t in tokens]
    return ' '.join(verbalized_tokens).strip().replace('  ', ' ')

def generate_lecture_script(topic, source_material):
    """Uses Gemini to create a lecture script with formula placeholders."""
    print(f"✍️ Generating lecture script for '{topic}'...")
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    prompt = f"You have two jobs... PERSONA: You are a friendly AI Teacher... TASK: Create a lecture script... Your entire output MUST be a single, valid JSON array... CRITICAL RULES: 1. Use <FORMULA:your_formula_here> for all formulas. 2. Synchronize examples... SOURCE MATERIAL: {source_material}"
    try:
        response = generate_content_with_retry(model, [prompt])
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_response, strict=False)
    except Exception as e:
        print(f"❌ An error occurred during script generation: {e}")
        return []

def convert_script_to_audio(script_parts, lecture_audio_folder):
    """Converts the 'spoken_text' into MP3 files, processing formula placeholders."""
    print("🎙️ Converting script to audio files...")
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


# =================================================================
# SECTION 4: DATA LOADING
# =================================================================
# In a real app, we don't re-process the PDF every time. We load our
# pre-processed and embedded data from a file.
try:
    # This assumes you have saved your df_embedded as a CSV file.
    # You will need to upload this file to your Hugging Face Space.
    print("Loading the knowledge base...")
    df_embedded = pd.read_csv("knowledge_base.csv")
    # The 'embedding' column is read as a string, so we need to convert it back to a list of floats.
    df_embedded['embedding'] = df_embedded['embedding'].apply(eval)
    print("✅ Knowledge base loaded successfully.")
except FileNotFoundError:
    print("❌ CRITICAL ERROR: 'knowledge_base.csv' not found. The app cannot function without it.")
    df_embedded = pd.DataFrame() # Create an empty DataFrame to prevent crashes.

# =================================================================
# SECTION 5: GRADIO UI CODE (This will be our next step)
# =================================================================
# This section is intentionally left blank. We will fill this in next
# to build the actual user interface for the application.

print("\nBackend logic is ready. Ready to build the UI.")
