import faiss
import os
import pickle
import numpy as np
from config import FAISS_INDEX_FOLDER

index = None
chunk_store = []
processed_files = set()

index_file = os.path.join(FAISS_INDEX_FOLDER, "index.faiss")
chunks_file = os.path.join(FAISS_INDEX_FOLDER, "chunks.pkl")
processed_file_registry = os.path.join(FAISS_INDEX_FOLDER, "processed_files.pkl")

def initialize_faiss():
    global index, chunk_store, processed_files
    dim = 384  # For MiniLM
    index = faiss.IndexFlatL2(dim)

    os.makedirs(FAISS_INDEX_FOLDER, exist_ok=True)

    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
    if os.path.exists(chunks_file):
        with open(chunks_file, "rb") as f:
            chunk_store.extend(pickle.load(f))
    if os.path.exists(processed_file_registry):
        with open(processed_file_registry, "rb") as f:
            processed_files = pickle.load(f)

def save_to_faiss(embeddings, chunks):
    global index, chunk_store
    index.add(np.array(embeddings).astype('float32'))
    chunk_store.extend(chunks)

    faiss.write_index(index, index_file)
    with open(chunks_file, "wb") as f:
        pickle.dump(chunk_store, f)

def already_processed(filename):
    return filename in processed_files

def mark_as_processed(filename):
    processed_files.add(filename)
    with open(processed_file_registry, "wb") as f:
        pickle.dump(processed_files, f)

def clear_vector_store():
    global index, chunk_store, processed_files
    chunk_store.clear()
    processed_files.clear()
    index = faiss.IndexFlatL2(384)

    if os.path.exists(index_file):
        os.remove(index_file)
    if os.path.exists(chunks_file):
        os.remove(chunks_file)
    if os.path.exists(processed_file_registry):
        os.remove(processed_file_registry)

def search_faiss(query_embedding, original_query):
    global index, chunk_store
    D, I = index.search(np.array([query_embedding]).astype('float32'), k=5)

    # De-duplicate results
    seen = set()
    top_chunks = []
    for i in I[0]:
        chunk = chunk_store[i]
        if chunk not in seen:
            seen.add(chunk)
            top_chunks.append(chunk)

    context = "\n".join(top_chunks)

    if "summarize" in original_query.lower() or "what is the pdf about" in original_query.lower():
        from transformers import pipeline
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        result = summarizer(context, max_length=100, min_length=30, do_sample=False)
        return context, result[0]["summary_text"]

    from transformers import pipeline
    qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    result = qa(question=original_query, context=context)
    return context, result['answer']



