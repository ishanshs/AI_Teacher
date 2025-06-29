# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file into the container at /code
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application code into the container at /code
COPY . .

# --- THE DEFINITIVE FIX ---
# Command to run the application using Gradio's official command-line interface.
# This is the standard way to launch a Gradio app in a Docker container.
CMD ["gradio", "app.py", "--server_name", "0.0.0.0", "--server_port", "7860"]
