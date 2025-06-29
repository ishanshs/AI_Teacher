# app.py (Final Corrected Version)

# =================================================================
# SECTION 1: ALL LIBRARY IMPORTS
# =================================================================
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
import gradio as gr
from PIL import Image
from google.api_core.exceptions import TooManyRequests
# --- KEY CHANGE: Added 'wait_random_exponential' to the import list ---
from tenacity import retry, stop_after_attempt, wait_fixed, wait_random_exponential, retry_if_exception_type

# =================================================================
# SECTION 2: API KEY CONFIGURATION
# =================================================================
try:
    # This securely reads the API key from the Hugging Face secret you created.
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        print("✅ Google AI API configured successfully.")
    else:
        print("❌ GOOGLE_API_KEY secret not found. Please set it in your Hugging Face Space settings.")
except Exception as e:
    print(f"❌ Error configuring Google AI API: {e}")

# =================================================================
# SECTION 3: ALL YOUR WORKING HELPER FUNCTIONS
# =================================================================

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def embed_with_retry(model, content, task_type):
    """Resilient wrapper for creating embeddings."""
    return genai.embed_content(
        model=model,
        content=content,
        task_type=task_type
    )

def find_relevant_page(query, dataframe):
    """Finds the single most relevant page from the DataFrame for a given query."""
    model = 'models/text-embedding-004'
    query_embedding_response = embed_with_retry(model=model, content=query, task_type="RETRIEVAL_QUERY")
    query_embedding = query_embedding_response['embedding']
    dot_products = np.dot(np.stack(dataframe['embedding']), query_embedding)
    best_passage_index = np.argmax(dot_products)
    return dataframe.iloc[best_passage_index]

def answer_question(question, dataframe):
    """Answers a text-based question using RAG."""
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    relevant_page_info = find_relevant_page(question, dataframe)
    context = relevant_page_info['text_for_search']
    source_page = relevant_page_info['page_number']
    source_pdf = relevant_page_info['source_pdf']
    
    prompt = f"Answer the following question based ONLY on the provided source material.\n\nQuestion: {question}\n\nSource Material:\n{context}"
    response = model.generate_content(prompt)
    return f"{response.text}\n\n(Source: {source_pdf}, Page {source_page})"

def analyze_handwritten_image(image, instruction):
    """Analyzes an uploaded image based on user instructions."""
    model = genai.GenerativeModel('gemini-pro-vision')
    prompt = f"You are an expert AI Teacher. Carefully analyze the user's handwritten work in the attached image based on their instruction.\n\nUser's Instruction: \"{instruction}\""
    response = model.generate_content([prompt, image])
    return response.text

# =================================================================
# SECTION 4: DATA LOADING
# =================================================================
try:
    print("Loading the knowledge base...")
    df_embedded = pd.read_csv("knowledge_base.csv")
    # The 'embedding' column is read as a string, so we convert it back to a list of floats.
    df_embedded['embedding'] = df_embedded['embedding'].apply(eval)
    print(f"✅ Knowledge base with {len(df_embedded)} chunks loaded successfully.")
    is_knowledge_base_loaded = True
except FileNotFoundError:
    print("❌ CRITICAL ERROR: 'knowledge_base.csv' not found. App cannot function without it.")
    is_knowledge_base_loaded = False
    # Create an empty DataFrame with the correct columns to prevent other errors.
    df_embedded = pd.DataFrame(columns=['text_for_search', 'source_pdf', 'page_number', 'image_path', 'embedding'])


# =================================================================
# SECTION 5: THE GRADIO USER INTERFACE
# =================================================================
def student_interface(text_question, image_upload, handwritten_instruction):
    """Function to power the student Q&A tab."""
    if not is_knowledge_base_loaded:
        return "Error: The knowledge base is not loaded. Please contact the administrator."
    
    # Priority is given to the image upload.
    if image_upload is not None:
        if not handwritten_instruction:
            return "Please provide an instruction for the uploaded image (e.g., 'Solve this problem')."
        return analyze_handwritten_image(image_upload, handwritten_instruction)
    elif text_question:
        return answer_question(text_question, df_embedded)
    else:
        return "Please either type a question or upload an image with an instruction."

# We define the UI using Gradio Blocks for a custom layout.
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🤖 AI Teacher Portal")
    
    with gr.Tabs():
        # --- First Tab: Student Q&A ---
        with gr.TabItem("Student Q&A"):
            gr.Markdown("## Ask a Question")
            gr.Markdown("Type a question OR upload an image of your work and provide an instruction below.")
            
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(label="Type your question here...")
                    image_input = gr.Image(type="pil", label="Upload an image of your homework/question")
                    instruction_input = gr.Textbox(label="If uploading an image, what should I do with it?", placeholder="e.g., 'Solve for x', 'Is my approach correct?'")
                    ask_button = gr.Button("Submit to AI Teacher", variant="primary")
                
                with gr.Column(scale=3):
                    qa_output = gr.Textbox(label="AI Teacher's Answer", lines=20, interactive=False)
            
            ask_button.click(
                fn=student_interface,
                inputs=[text_input, image_input, instruction_input],
                outputs=qa_output
            )

        # --- Second Tab: Teacher Lecture Mode (Placeholder) ---
        with gr.TabItem("Teacher Lecture Mode"):
            gr.Markdown("## Generate an Audio Lecture")
            gr.Markdown("This feature is under construction. Come back soon to generate full audio-visual lectures!")


# Launch the app! This command starts the web server.
app.launch()
