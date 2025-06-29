# app.py (Final Definitive Version)

# =================================================================
# SECTION 1: ALL LIBRARY IMPORTS
# =================================================================
import os
import json
import re
import pandas as pd
import google.generativeai as genai
import gradio as gr
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_random_exponential

# =================================================================
# SECTION 2: GLOBAL VARIABLES & DATA LOADING
# =================================================================
try:
    print("Loading the knowledge base...")
    df_embedded = pd.read_csv("knowledge_base.csv")
    df_embedded['embedding'] = df_embedded['embedding'].apply(eval)
    print(f"✅ Knowledge base with {len(df_embedded)} chunks loaded successfully.")
    is_knowledge_base_loaded = True
except FileNotFoundError:
    print("❌ CRITICAL ERROR: 'knowledge_base.csv' not found. App cannot function without it.")
    is_knowledge_base_loaded = False
    df_embedded = pd.DataFrame()

# =================================================================
# SECTION 3: CORE HELPER FUNCTIONS
# =================================================================

def configure_google_ai():
    """Checks for the API key and configures the genai library."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return False, "ERROR: 'GOOGLE_API_KEY' secret not found in Hugging Face Space settings."
    try:
        genai.configure(api_key=api_key)
        return True, "SUCCESS: Google AI API configured successfully."
    except Exception as e:
        return False, f"ERROR: The provided API Key is invalid or has an issue. Details: {e}"

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def embed_with_retry(model, content, task_type):
    """Resilient wrapper for creating embeddings."""
    return genai.embed_content(model=model, content=content, task_type=task_type)

def find_relevant_page(query, dataframe):
    """Finds the single most relevant page from the DataFrame."""
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
    
    prompt = f"Answer the following question based ONLY on the provided source material.\n\nQuestion: {question}\n\nSource Material:\n{context}"
    response = model.generate_content(prompt)
    return f"{response.text}\n\n(Source: Page {source_page})"

def analyze_handwritten_image(image, instruction):
    """Analyzes an uploaded image based on user instructions."""
    model = genai.GenerativeModel('gemini-pro-vision')
    prompt = f"You are an AI Teacher. Analyze the handwritten work in the image based on this instruction: \"{instruction}\""
    response = model.generate_content([prompt, image])
    return response.text

# =================================================================
# SECTION 4: THE GRADIO USER INTERFACE
# =================================================================

def student_interface(text_question, image_upload, handwritten_instruction):
    """Function to power the student Q&A tab. It now checks for the API key first."""
    is_configured, message = configure_google_ai()
    if not is_configured:
        return message
    
    if not is_knowledge_base_loaded:
        return "ERROR: The knowledge base file ('knowledge_base.csv') is not loaded."
    
    try:
        if image_upload is not None:
            if not handwritten_instruction:
                return "Please provide an instruction for the uploaded image."
            return analyze_handwritten_image(image_upload, handwritten_instruction)
        elif text_question:
            return answer_question(text_question, df_embedded)
        else:
            return "Please type a question or upload an image with an instruction."
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def check_environment():
    """This function checks the environment and returns a diagnostic message."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        key_status = f"SUCCESS: Found the GOOGLE_API_KEY secret."
    else:
        key_status = "FAILURE: The GOOGLE_API_KEY secret was NOT found in the environment."
    is_configured, config_message = configure_google_ai()
    return f"--- System Status ---\n\n1. API Key Secret Check:\n{key_status}\n\n2. Google AI Library Configuration Check:\n{config_message}"

# We define the UI using Gradio Blocks.
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🤖 AI Teacher Portal")
    
    with gr.Tabs():
        # --- Student Q&A Tab ---
        with gr.TabItem("Student Q&A"):
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(label="Type your question here...")
                    image_input = gr.Image(type="pil", label="Or upload an image of your homework")
                    instruction_input = gr.Textbox(label="If uploading an image, what should I do?", placeholder="e.g., 'Solve for x'")
                    ask_button = gr.Button("Submit", variant="primary")
                with gr.Column(scale=3):
                    qa_output = gr.Textbox(label="AI Teacher's Answer", lines=20, interactive=False)
            
            ask_button.click(
                fn=student_interface,
                inputs=[text_input, image_input, instruction_input],
                outputs=qa_output
            )

        # --- Debugging Tab ---
        with gr.TabItem("Debug Info"):
            gr.Markdown("## System Environment Check")
            debug_button = gr.Button("Check Environment")
            debug_output = gr.Textbox(label="System Status", lines=10, interactive=False)
            debug_button.click(fn=check_environment, inputs=[], outputs=debug_output)

# Launch the app!
# --- KEY CHANGE: We remove the unsupported 'server_timeout' argument ---
app.launch()

