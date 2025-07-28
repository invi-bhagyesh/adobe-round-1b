import spacy
import os
from sentence_transformers import SentenceTransformer

# Test SpaCy
try:
    nlp = spacy.load('en_core_web_sm')
    print('✅ SpaCy model verified')
except Exception as e:
    print(f'❌ SpaCy model error: {e}')
    exit(1)

# Test SentenceTransformer
try:
    model_path = '/app/cached_models/sentence-transformers_all-MiniLM-L6-v2'
    if os.path.exists(model_path):
        model = SentenceTransformer(model_path)
        test_emb = model.encode(['test'])
        print(f'✅ SentenceTransformer model verified - shape: {test_emb.shape}')
    else:
        print(f'❌ Model path not found: {model_path}')
        exit(1)
except Exception as e:
    print(f'❌ SentenceTransformer error: {e}')
    exit(1)

print('🎉 All models verified successfully!')
