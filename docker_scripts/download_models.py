from sentence_transformers import SentenceTransformer
import os

# Download to temporary location
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Move to our cache directory
cache_path = '/app/cached_models/sentence-transformers_all-MiniLM-L6-v2'
os.makedirs(cache_path, exist_ok=True)

# Save model to cache directory
model.save(cache_path)
print(f'Model cached to: {cache_path}')
