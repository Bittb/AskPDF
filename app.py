from flask import Flask, request, render_template, redirect, url_for, jsonify
import os
from utils.pdf_reader import extract_text_from_pdf
from utils.chunker import chunk_text
from utils.embedder import get_embeddings, embed_query, load_model
from utils.vector_store import (
    save_to_faiss,
    search_faiss,
    initialize_faiss,
    already_processed,
    mark_as_processed,
    clear_vector_store,
)
from config import UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model()
initialize_faiss()

@app.route('/')
def index():
    return render_template('index.html', answer=None, context=None)

@app.route('/upload', methods=['POST'])
def upload():
    print("Incoming request files:", request.files)
    uploaded = []
    total_chunks = 0
    if 'files' not in request.files:
        return jsonify({"error": "No files part in the request."}), 400

    pdfs = request.files.getlist('files')
    if not pdfs:
        return jsonify({"error": "No files uploaded."}), 400

    for pdf in pdfs:
        path = os.path.join(UPLOAD_FOLDER, pdf.filename)
        if not already_processed(pdf.filename):
            pdf.save(path)
            text = extract_text_from_pdf(path)
            chunks = chunk_text(text)
            embeddings = get_embeddings(chunks, model)
            save_to_faiss(embeddings, chunks)
            mark_as_processed(pdf.filename)
            uploaded.append(pdf.filename)
            total_chunks += len(chunks)
    return jsonify({"files": uploaded, "total_chunks": total_chunks})

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    user_query = data.get('query')
    if not user_query:
        return jsonify({"error": "Query cannot be empty."}), 400
    query_embedding = embed_query(user_query, model)
    context, answer = search_faiss(query_embedding, user_query)
    return jsonify({"answer": answer, "context": context})

@app.route('/clear', methods=['POST'])
def clear():
    try:
        clear_vector_store()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

