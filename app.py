# app.py (Version 4: Enhanced Persona and Context)

import os
import pandas as pd
import numpy as np
import google.generativeai as genai
import streamlit as st
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Teacher Portal",
    page_icon="🤖",
    layout="wide"
)

# --- Authentication and Resource Loading (run once and cached) ---
@st.cache_resource
def load_resources():
    """Configures the API and loads all necessary resources."""
    # Configure Google AI API
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY secret not found. Please set it in your Space settings.")
        return None, None
    try:
        genai.configure(api_key=api_key)
        print("✅ Google AI API configured successfully.")
    except Exception as e:
        st.error(f"Error configuring Google AI API: {e}")
        return None, None

    # Load the generative model
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    print("✅ Gemini 1.5 Flash model loaded successfully.")

    # Load the knowledge base
    try:
        dataframe = pd.read_csv("knowledge_base.csv")
        dataframe['embedding'] = dataframe['embedding'].apply(eval)
        print("✅ Knowledge base loaded successfully.")
        return model, dataframe
    except FileNotFoundError:
        st.error("CRITICAL ERROR: 'knowledge_base.csv' not found. App cannot function.")
        return None, None

# Load the resources when the app starts.
model, df_embedded = load_resources()

# --- Core Logic Functions (with Upgrades) ---

# --- NEW: A more powerful function to gather comprehensive context ---
def find_relevant_context(query, dataframe, k=3):
    """
    Finds the top 'k' most relevant text chunks from the knowledge base
    and combines them into a single block of context.
    """
    query_embedding_response = genai.embed_content(model='models/text-embedding-004', content=query, task_type="RETRIEVAL_QUERY")
    query_embedding = query_embedding_response['embedding']
    dot_products = np.dot(np.stack(dataframe['embedding']), query_embedding)
    
    # Get the indices of the top 'k' most similar chunks.
    top_k_indices = np.argsort(dot_products)[-k:][::-1] # Reverse to get highest similarity first
    
    # Combine the text from these chunks into one string.
    relevant_context = "\n\n---\n\n".join(dataframe.iloc[top_k_indices]['text_for_search'].tolist())
    return relevant_context

# --- NEW: The upgraded function with our AI Teacher persona ---
def answer_question_from_text(question, dataframe):
    """
    Handles the text-based Q&A logic with a better persona and more context.
    """
    # Use our new function to get a richer context.
    context = find_relevant_context(question, dataframe)
    
    # This new prompt separates persona from the logical task.
    prompt = f"""
    **Your Persona:**
    You are a friendly, cheerful, and patient AI Teacher. Imagine you are an older, knowledgeable family member explaining a concept simply and encouragingly. Your goal is to make the student feel supported and confident.

    **Your Task & Rules:**
    1.  Your primary task is to answer the user's question.
    2.  You MUST base your answer strictly on the provided 'Source Material'.
    3.  **DO NOT** start your response with phrases like "Based on the provided text..." or "The source material says...". Answer the question directly in your own words, assuming the persona.
    4.  Explain concepts clearly and step-by-step. Make it feel like a helpful conversation.

    ---
    **Source Material:**
    {context}
    ---

    **User's Question:**
    {question}
    """
    
    # Use the pre-loaded Flash model for fast text responses
    response = model.generate_content(prompt)
    return response.text

def analyze_handwritten_image(image, instruction):
    """Handles the image analysis logic."""
    if image is None: return "Please upload an image."
    if not instruction: return "Please provide an instruction for the image."
            
    print(f"Received image and instruction: {instruction}")
    # We create a new instance of the Pro vision model here for this specific task.
    vision_model = genai.GenerativeModel('gemini-1.5-pro-latest')
    prompt = f"You are an expert AI Teacher. Analyze the handwritten work in the image based on this instruction: \"{instruction}\""
    response = vision_model.generate_content([prompt, image])
    return response.text

# --- Streamlit User Interface (No changes needed here) ---
st.title("🤖 AI Teacher Portal")

with st.sidebar:
    st.header("App Mode")
    app_mode = st.radio("Choose a feature:", ("❓ Textbook Q&A", "✍️ Homework Helper"))

if app_mode == "❓ Textbook Q&A":
    st.header("Ask a Question from the Textbook")
    if model and df_embedded is not None and not df_embedded.empty:
        text_question = st.text_area("Enter your question here:", height=150)
        if st.button("Get Answer"):
            if text_question:
                with st.spinner("The AI Teacher is thinking..."):
                    response = answer_question_from_text(text_question, df_embedded)
                    st.success("Here is your answer:")
                    st.write(response)
            else:
                st.warning("Please enter a question.")
    else:
        st.error("Application is not ready. Please check the logs or secrets.")

elif app_mode == "✍️ Homework Helper":
    st.header("Get Help with Your Homework")
    if model:
        uploaded_file = st.file_uploader("Upload an image of your work", type=["png", "jpg", "jpeg"])
        instruction = st.text_input("What should I do with this image?", placeholder="e.g., 'Solve for x' or 'Check my work'")
        if st.button("Analyze Image"):
            if uploaded_file is not None:
                if instruction:
                    with st.spinner("The AI Teacher is analyzing your image..."):
                        image = Image.open(uploaded_file)
                        response = analyze_handwritten_image(image, instruction)
                        st.success("Here is the analysis:")
                        st.write(response)
                else:
                    st.warning("Please provide an instruction for the image.")
            else:
                st.warning("Please upload an image.")
    else:
        st.error("Application is not ready. Please check the logs or secrets.")
