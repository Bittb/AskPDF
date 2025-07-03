from sentence_transformers import SentenceTransformer

def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(chunks, model):
    return model.encode(chunks, show_progress_bar=True)

def embed_query(query, model):
    return model.encode([query])[0]

