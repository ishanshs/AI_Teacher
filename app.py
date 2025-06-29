# app.py (Final Version using FastAPI)
import os
import pandas as pd
import google.generativeai as genai
import gradio as gr
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_random_exponential
import fastapi

# --- Data Loading ---
try:
    df_embedded = pd.read_csv("knowledge_base.csv")
    df_embedded['embedding'] = df_embedded['embedding'].apply(eval)
    is_knowledge_base_loaded = True
    print("✅ Knowledge base loaded.")
except FileNotFoundError:
    print("❌ CRITICAL ERROR: 'knowledge_base.csv' not found.")
    is_knowledge_base_loaded = False
    df_embedded = pd.DataFrame()

# --- Core API and Logic Functions ---
def configure_google_ai():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key: return False, "ERROR: GOOGLE_API_KEY secret not found."
    try:
        genai.configure(api_key=api_key)
        return True, "SUCCESS: API configured."
    except Exception as e: return False, f"ERROR: Invalid API Key. {e}"

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def embed_with_retry(content, task_type):
    return genai.embed_content(model='models/text-embedding-004', content=content, task_type=task_type)

def answer_question(question, dataframe):
    # This is a placeholder for your actual RAG logic
    return "This is a placeholder answer for your text question."

def analyze_handwritten_image(image, instruction):
    # This is a placeholder for your actual vision logic
    return "This is a placeholder answer for your image question."

# --- Gradio Interface Definition ---
def student_interface(text_question, image_upload, handwritten_instruction):
    is_configured, message = configure_google_ai()
    if not is_configured: return message
    if not is_knowledge_base_loaded: return "ERROR: Knowledge base not loaded."

    if image_upload is not None:
        if not handwritten_instruction: return "Please provide an instruction for the uploaded image."
        return analyze_handwritten_image(image_upload, handwritten_instruction)
    elif text_question:
        return answer_question(text_question, df_embedded)
    else:
        return "Please type a question or upload an image."

# --- Build the Gradio App (but do not launch it) ---
gradio_app = gr.Blocks(theme=gr.themes.Soft())
with gradio_app:
    gr.Markdown("# 🤖 AI Teacher Portal")
    with gr.Tabs():
        with gr.TabItem("Student Q&A"):
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(label="Type your question here...")
                    image_input = gr.Image(type="pil", label="Or upload homework")
                    instruction_input = gr.Textbox(label="If uploading an image, what to do?", placeholder="e.g., 'Solve for x'")
                    ask_button = gr.Button("Submit", variant="primary")
                with gr.Column(scale=3):
                    qa_output = gr.Textbox(label="AI Teacher's Answer", lines=20, interactive=False)
            ask_button.click(fn=student_interface, inputs=[text_input, image_input, instruction_input], outputs=qa_output)

# --- THE DEFINITIVE FIX: Create a FastAPI app and mount the Gradio app to it. ---
app = fastapi.FastAPI()
app = gr.mount_gradio_app(app, gradio_app, path="/")
