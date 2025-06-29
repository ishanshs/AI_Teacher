# app.py (Minimum Viable Product for Debugging)
import gradio as gr
import fastapi

def simple_echo(text):
    """A simple function that echoes the input text."""
    return f"The app is running. You typed: {text}"

# --- Build a bare-bones Gradio App ---
gradio_app = gr.Blocks()
with gradio_app:
    gr.Markdown("# 🤖 AI Teacher Portal (Debug Mode)")
    text_input = gr.Textbox(label="Test Input")
    text_output = gr.Textbox(label="Test Output")
    submit_button = gr.Button("Submit")

    submit_button.click(fn=simple_echo, inputs=text_input, outputs=text_output)

# --- Mount it on FastAPI ---
app = fastapi.FastAPI()
app = gr.mount_gradio_app(app, gradio_app, path="/")
