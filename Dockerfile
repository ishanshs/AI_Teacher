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

# --- THE DEFINITIVE FIX: Launch the app with the Uvicorn server ---
# This tells Uvicorn to find the 'app' object inside the 'app.py' file.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
