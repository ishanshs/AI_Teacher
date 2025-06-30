# app.py (Version 6: Correct Context Gathering)

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

    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    print("✅ Gemini 1.5 Flash model loaded successfully.")

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

# --- Core Logic Functions (with Corrected Context Function) ---

# --- THIS IS THE CORRECT, UPGRADED FUNCTION ---
def find_relevant_context(query, dataframe, k=3):
    """
    Finds the top 'k' most relevant text chunks and combines them.
    """
    query_embedding_response = genai.embed_content(model='models/text-embedding-004', content=query, task_type="RETRIEVAL_QUERY")
    query_embedding = query_embedding_response['embedding']
    dot_products = np.dot(np.stack(dataframe['embedding']), query_embedding)
    
    # Get the indices of the top 'k' most similar passages.
    top_k_indices = np.argsort(dot_products)[-k:][::-1]
    
    # Combine the text from these chunks into one string.
    relevant_context = "\n\n---\n\n".join(dataframe.iloc[top_k_indices]['text_for_search'].tolist())
    return relevant_context

def answer_question_from_text(question, dataframe):
    """Handles the text-based Q&A logic with a refined persona."""
    # --- THIS IS THE CORRECTED FUNCTION CALL ---
    context = find_relevant_context(question, dataframe, k=3)
    
    prompt = f"""
    **Your Persona:**
    You are a friendly, cheerful, and patient AI Teacher. Imagine you are an older, knowledgeable family member explaining a concept simply and encouragingly. Your goal is to make the student feel supported and confident.

    **Your Task & Rules:**
    1.  Your primary task is to answer the user's question clearly and step-by-step.
    2.  You MUST base your answer strictly on the provided 'Source Material'.
    3.  **IMPORTANT:** Never refer to the source material directly. Do NOT use phrases like "The provided text...", "The source material mentions...", or "According to the text...". Just answer the question naturally as if the knowledge is your own.
    4.  If the source material does not contain the answer, simply say, "That's a great question, but it seems to be outside the scope of this textbook. Let's try another topic!"

    ---
    **Source Material:**
    {context}
    ---

    **User's Question:**
    {question}
    """
    
    response = model.generate_content(prompt)
    return response.text

def analyze_handwritten_image(image, instruction):
    """Handles the image analysis logic with a focused task instruction."""
    if image is None: return "Please upload an image."
    if not instruction: return "Please provide an instruction for the image."
            
    prompt = f"""
    You are an expert AI Teacher. Your task is to follow the user's instruction precisely based on the provided image of their handwritten work.

    **CRITICAL RULE:**
    Provide ONLY the direct answer to the user's instruction. For example, if the user asks for a "step-by-step solution", provide ONLY the steps to solve the problem.
    **DO NOT** add any extra summary, critique, or concluding remarks about the student's work unless the user explicitly asks for it.

    **User's Instruction:** "{instruction}"
    """
    
    # Use the Flash model for vision as well to stay on the free tier
    response = model.generate_content([prompt, image])
    return response.text

# --- Streamlit User Interface (No changes needed here) ---
st.title("🤖 AI Teacher Portal")

with st.sidebar:
    st.header("App Mode")
    app_mode = st.radio("Choose a feature:", ("❓ Textbook Q&A", "✍️ Homework Helper"))

if app_mode == "❓ Textbook Q&A":
    st.header("Ask a Question from the Textbook: Class 7 NCERT Only")
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
    st.header("Get Help with Your Homework: Class 7 NCERT Only")
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
