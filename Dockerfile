# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# --- THE DEFINITIVE FIX ---
# The official command to run a Streamlit app in a production container.
# --server.headless=true tells Streamlit to not try to open a browser window.
# --browser.gatherUsageStats=false prevents it from trying to write the .streamlit config folder.
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
