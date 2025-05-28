FROM python:3.10-slim

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV RAG_CONFIG_PATH=/app/config/rag_config.json

# Expose port
EXPOSE 8000

# Run the API
CMD ["python", "-m", "src.deployment.api"]
