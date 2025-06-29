# app.py (Version 3: Advanced UI with Image Analysis)

import os
import pandas as pd
import numpy as np
import google.generativeai as genai
import gradio as gr
import fastapi
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_random_exponential

# --- Data Loading (No changes here) ---
try:
    df_embedded = pd.read_csv("knowledge_base.csv")
    df_embedded['embedding'] = df_embedded['embedding'].apply(eval)
    is_knowledge_base_loaded = True
    print("✅ Knowledge base loaded successfully.")
except FileNotFoundError:
    print("❌ CRITICAL ERROR: 'knowledge_base.csv' not found.")
    is_knowledge_base_loaded = False
    df_embedded = pd.DataFrame()

# --- Core API and Logic Functions (With Updates) ---

def configure_google_ai():
    """Checks for the API key and configures the genai library."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return False, "ERROR: The GOOGLE_API_KEY secret is not set in your Space settings."
    try:
        genai.configure(api_key=api_key)
        return True, "SUCCESS: API configured."
    except Exception as e:
        return False, f"ERROR: The provided API Key is invalid. Details: {e}"

def find_relevant_passage(query, dataframe):
    """Finds the most relevant text chunk from the knowledge base."""
    query_embedding_response = genai.embed_content(model='models/text-embedding-004', content=query, task_type="RETRIEVAL_QUERY")
    query_embedding = query_embedding_response['embedding']
    dot_products = np.dot(np.stack(dataframe['embedding']), query_embedding)
    best_passage_index = np.argmax(dot_products)
    return dataframe.iloc[best_passage_index]

def answer_question_from_text(question, dataframe):
    """Answers a text-based question using RAG."""
    relevant_page = find_relevant_passage(question, dataframe)
    prompt = f"Answer the following question based ONLY on the provided source material.\n\nQuestion: {question}\n\nSource Material:\n{relevant_page['text_for_search']}"
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    response = model.generate_content(prompt)
    # As requested, we are removing the source citation from the final output.
    return response.text

# --- NEW: Function to handle image analysis ---
def analyze_handwritten_image(image, instruction):
    """Analyzes an uploaded image based on user instructions using a vision model."""
    model = genai.GenerativeModel('gemini-pro-vision')
    prompt = f"You are an expert AI Teacher. Carefully analyze the user's handwritten work in the attached image based on their instruction.\n\nUser's Instruction: \"{instruction}\""
    response = model.generate_content([prompt, image])
    return response.text

# --- NEW: A "Router" function to direct traffic ---
def student_interface(text_question, image_upload, handwritten_instruction):
    """
    This function acts as the main entry point for user submissions.
    It checks if an image or text was provided and calls the appropriate function.
    """
    # First, always check for API key configuration
    is_configured, message = configure_google_ai()
    if not is_configured:
        return message

    if not is_knowledge_base_loaded:
        return "ERROR: The knowledge base is not loaded. Please contact the administrator."

    # Priority is given to the image upload.
    if image_upload is not None:
        if not handwritten_instruction:
            return "Please provide an instruction for the uploaded image (e.g., 'Solve this problem')."
        return analyze_handwritten_image(image_upload, handwritten_instruction)
    elif text_question:
        return answer_question_from_text(text_question, df_embedded)
    else:
        # If neither input is provided
        return "Please either type a question or upload an image with an instruction."


# --- Gradio Interface Definition (Upgraded for Advanced Features) ---
with gr.Blocks(theme=gr.themes.Soft()) as gradio_app:
    gr.Markdown("# 🤖 AI Teacher Portal")
    gr.Markdown("Type a question OR upload an image of your work and provide an instruction below.")

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(label="Type your question here...")
            gr.Markdown("--- OR ---") # Visual separator
            image_input = gr.Image(type="pil", label="Upload an image of your homework/question")
            instruction_input = gr.Textbox(label="If uploading an image, what should I do with it?", placeholder="e.g., 'Solve for x', 'Is my approach correct?'")
            submit_button = gr.Button("Submit to AI Teacher", variant="primary")

        with gr.Column(scale=3):
            output_text = gr.Textbox(label="AI Teacher's Answer", lines=20, interactive=False)

    # The button now calls our new "router" function
    submit_button.click(
        fn=student_interface,
        inputs=[text_input, image_input, instruction_input],
        outputs=output_text
    )

# --- Mount the Gradio app on a FastAPI server (No changes here) ---
app = fastapi.FastAPI()
app = gr.mount_gradio_app(app, gradio_app, path="/")
