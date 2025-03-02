# Use official Python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy all files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure dataset and model directories exist
RUN mkdir -p /app/dataset

# Expose port required by Hugging Face
EXPOSE 7860

# Run Flask app
CMD ["python", "app.py"]
