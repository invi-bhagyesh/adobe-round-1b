# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables for offline mode
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV SPACY_WARNING_IGNORE=W008
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create model cache directory
RUN mkdir -p /app/cached_models

# Download SpaCy English model during build
RUN python -m spacy download en_core_web_sm

# Download SentenceTransformer model during build
RUN python -c "\
    import os; \
    from sentence_transformers import SentenceTransformer; \
    import shutil; \
    os.environ.pop('HF_HUB_OFFLINE', None); \
    os.environ.pop('TRANSFORMERS_OFFLINE', None); \
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'; \
    cache_dir = '/app/cached_models/sentence-transformers_all-MiniLM-L6-v2'; \
    print(f'Downloading {model_name}...'); \
    model = SentenceTransformer(model_name); \
    model.save(cache_dir); \
    print(f'Model saved to {cache_dir}'); \
    test_embedding = model.encode(['test sentence']); \
    print(f'Model verification successful. Embedding shape: {test_embedding.shape}'); \
    "

# Install additional dependencies that might be needed
RUN pip install --no-cache-dir \
    safetensors \
    torch \
    transformers

# Re-enable offline mode for runtime
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

# Copy application files
COPY app.py .
COPY . .

# Create directories for input and output
RUN mkdir -p /app/input /app/output

# Set permissions
RUN chmod +x app.py

# Verify installation during build
RUN python -c "\
    import spacy; \
    import fitz; \
    from sentence_transformers import SentenceTransformer; \
    import numpy as np; \
    from sklearn.metrics.pairwise import cosine_similarity; \
    print('✅ All imports successful'); \
    nlp = spacy.load('en_core_web_sm'); \
    doc = nlp('Test sentence'); \
    print(f'✅ SpaCy working: {len(doc)} tokens'); \
    model = SentenceTransformer('/app/cached_models/sentence-transformers_all-MiniLM-L6-v2'); \
    embedding = model.encode(['test']); \
    print(f'✅ SentenceTransformer working: {embedding.shape}'); \
    print('🎉 All models verified during build!'); \
    "

# Expose port if needed (optional)
EXPOSE 8000

# Default command to run the application
CMD ["python", "app.py", "--help"]
