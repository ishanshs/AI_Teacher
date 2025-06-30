# app.py (The Definitive, Final Version with Flagging Disabled)

import os
import pandas as pd
import numpy as np
import google.generativeai as genai
import gradio as gr
import fastapi
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_random_exponential

# --- Configuration and Setup ---
def configure_google_ai():
    """A single, reliable function to configure the API key."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        # Use gr.Error to display a user-friendly error in the Gradio UI
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

# --- Core Logic Functions ---
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

# --- Gradio Interface Definition ---

# This is the interface for the first tab (Text Q&A)
text_qa_interface = gr.Interface(
    fn=answer_question_from_text,
    inputs=gr.Textbox(lines=5, placeholder="Type your question about the textbook here..."),
    outputs=gr.Textbox(label="Answer"),
    title="❓ Ask a Question from the Textbook",
    description="Get answers based on your Class 7 Math book.",
    allow_flagging=False  # THE FIX: This prevents the app from creating a 'flagged' folder.
)

# This is the interface for the second tab (Image Q&A)
image_qa_interface = gr.Interface(
    fn=analyze_handwritten_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image of Your Homework"),
        gr.Textbox(placeholder="e.g., 'Solve for x' or 'Check my work'", label="Instruction")
    ],
    outputs=gr.Textbox(label="Analysis"),
    title="✍️ Get Help with Your Homework",
    description="Upload a photo of your handwritten work and tell the AI Teacher what to do.",
    allow_flagging=False  # THE FIX: We add it here as well for consistency.
)

# Combine the two interfaces into a single, clean, tabbed application.
gradio_app = gr.TabbedInterface(
    [text_qa_interface, image_qa_interface],
    ["Textbook Q&A", "Homework Helper"]
)

# --- Mount the Gradio app on a FastAPI server ---
app = fastapi.FastAPI()
app = gr.mount_gradio_app(app, gradio_app, path="/")
