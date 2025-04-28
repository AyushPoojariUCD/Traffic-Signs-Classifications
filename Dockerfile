# Use an official lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy your requirements.txt and app.py
COPY requirements.txt . 
COPY app.py .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
