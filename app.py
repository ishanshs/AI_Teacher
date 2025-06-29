# app.py (Version 2.1: Typo Corrected)

import os
import pandas as pd
import google.generativeai as genai
import gradio as gr
import fastapi
from tenacity import retry, stop_after_attempt, wait_random_exponential
import numpy as np

# --- Data Loading ---
try:
    df_embedded = pd.read_csv("knowledge_base.csv")
    # Convert the string representation of the embedding back into a list of floats
    df_embedded['embedding'] = df_embedded['embedding'].apply(eval)
    is_knowledge_base_loaded = True
    print("✅ Knowledge base loaded successfully.")
except FileNotFoundError:
    print("❌ CRITICAL ERROR: 'knowledge_base.csv' not found.")
    is_knowledge_base_loaded = False
    df_embedded = pd.DataFrame() # Create an empty dataframe to prevent crashes

# --- Core API and Logic Functions ---
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

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def embed_with_retry(content):
    """A resilient wrapper for the embedding API call."""
    return genai.embed_content(model='models/text-embedding-004', content=content, task_type="RETRIEVAL_DOCUMENT")

def find_relevant_passage(query, dataframe):
    """Finds the most relevant text chunk from the knowledge base."""
    # First, configure the API
    is_configured, message = configure_google_ai()
    if not is_configured:
        return None, message # Return the error message if configuration fails

    # Embed the user's query
    query_embedding_response = genai.embed_content(model='models/text-embedding-004', content=query, task_type="RETRIEVAL_QUERY")
    query_embedding = query_embedding_response['embedding']

    # Find the most similar passage using dot product
    dot_products = np.dot(np.stack(dataframe['embedding']), query_embedding)
    best_passage_index = np.argmax(dot_products)
    return dataframe.iloc[best_passage_index], None


def answer_question(question):
    """The main function to answer a text-based question using RAG."""
    if not is_knowledge_base_loaded:
        return "ERROR: The knowledge base is not loaded. Please check the container logs."

    # THIS IS THE CORRECTED LINE:
    relevant_page, error = find_relevant_passage(question, df_embedded)
    if error:
        return error # Return any error messages from the helper functions

    # Build the prompt for the generative model
    prompt = f"Answer the following question based ONLY on the provided source material.\n\nQuestion: {question}\n\nSource Material:\n{relevant_page['text_for_search']}"

    # Call the generative model
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    response = model.generate_content(prompt)

    return f"{response.text}\n\n(Source: Page {relevant_page['page_number']})"


# --- Gradio Interface Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as gradio_app:
    gr.Markdown("# 🤖 AI Teacher Portal")
    gr.Markdown("Ask a question about your Class 7 Math book.")

    with gr.Row():
        text_input = gr.Textbox(label="Type your question here...", scale=3)
        submit_button = gr.Button("Submit", variant="primary", scale=1)

    output_text = gr.Textbox(label="AI Teacher's Answer", lines=20, interactive=False)

    submit_button.click(
        fn=answer_question,
        inputs=text_input,
        outputs=output_text
    )

# --- Mount the Gradio app on a FastAPI server ---
app = fastapi.FastAPI()
app = gr.mount_gradio_app(app, gradio_app, path="/")

