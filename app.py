# app.py (The Definitive Version using gr.Blocks for the UI)

import os
import pandas as pd
import numpy as np
import google.generativeai as genai
import gradio as gr
import fastapi
from PIL import Image

# --- Configuration and Setup ---
def configure_google_ai():
    """A single, reliable function to configure the API key."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise gr.Error("GOOGLE_API_KEY secret not found. Please set it in your Space settings.")
    genai.configure(api_key=api_key)
    print("✅ Google AI API configured successfully.")

# Call configuration at startup.
configure_google_ai()

# --- Data Loading ---
try:
    df_embedded = pd.read_csv("knowledge_base.csv")
    df_embedded['embedding'] = df_embedded['embedding'].apply(eval)
    is_knowledge_base_loaded = True
    print("✅ Knowledge base loaded successfully.")
except FileNotFoundError:
    is_knowledge_base_loaded = False
    print("❌ WARNING: 'knowledge_base.csv' not found. Text Q&A will be disabled.")

# --- Core Logic Functions (No changes needed here) ---
def find_relevant_passage(query, dataframe):
    """Finds the most relevant text chunk from the knowledge base."""
    query_embedding_response = genai.embed_content(model='models/text-embedding-004', content=query, task_type="RETRIEVAL_QUERY")
    query_embedding = query_embedding_response['embedding']
    dot_products = np.dot(np.stack(dataframe['embedding']), query_embedding)
    best_passage_index = np.argmax(dot_products)
    return dataframe.iloc[best_passage_index]

def answer_question_from_text(question):
    """Handles the text-based Q&A logic."""
    if not is_knowledge_base_loaded:
        return "Text Q&A is currently disabled because the knowledge base file was not found."

    print(f"Received text question: {question}")
    relevant_page = find_relevant_passage(question, df_embedded)
    prompt = f"Answer the following question based ONLY on the provided source material.\n\nQuestion: {question}\n\nSource Material:\n{relevant_page['text_for_search']}"
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    response = model.generate_content(prompt)
    return response.text

def analyze_handwritten_image(image, instruction):
    """Handles the image analysis logic."""
    if image is None:
        return "Please upload an image."
    if not instruction:
        return "Please provide an instruction for the image."

    print(f"Received image and instruction: {instruction}")
    model = genai.GenerativeModel('gemini-pro-vision')
    prompt = f"You are an expert AI Teacher. Analyze the handwritten work in the image based on this instruction: \"{instruction}\""
    response = model.generate_content([prompt, image])
    return response.text

# --- Gradio Interface Definition using gr.Blocks for Maximum Stability ---
with gr.Blocks(theme=gr.themes.Soft()) as gradio_app:
    gr.Markdown("# 🤖 AI Teacher Portal")

    with gr.Tabs():
        # --- First Tab: Textbook Q&A ---
        with gr.TabItem("❓ Textbook Q&A"):
            with gr.Column():
                text_question_input = gr.Textbox(lines=5, placeholder="Type your question about the textbook here...", label="Question")
                text_submit_button = gr.Button("Get Answer", variant="primary")
                text_answer_output = gr.Textbox(label="Answer", interactive=False)

        # --- Second Tab: Homework Helper ---
        with gr.TabItem("✍️ Homework Helper"):
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image of Your Homework")
                instruction_input = gr.Textbox(placeholder="e.g., 'Solve for x' or 'Check my work'", label="Instruction")
                image_submit_button = gr.Button("Get Help", variant="primary")
                image_analysis_output = gr.Textbox(label="Analysis", interactive=False)

    # --- Wire the components to the functions ---
    text_submit_button.click(
        fn=answer_question_from_text,
        inputs=[text_question_input],
        outputs=[text_answer_output]
    )

    image_submit_button.click(
        fn=analyze_handwritten_image,
        inputs=[image_input, instruction_input],
        outputs=[image_analysis_output]
    )

# --- Mount the Gradio app on a FastAPI server ---
app = fastapi.FastAPI()
app = gr.mount_gradio_app(app, gradio_app, path="/")
