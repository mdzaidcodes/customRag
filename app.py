# app.py
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from sentence_transformers import SentenceTransformer, util
import pdfplumber
import os
import tempfile
import ollama 

app = Flask(__name__)
socketio = SocketIO(app)

# Load a model for creating embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # A smaller, faster model

# Placeholder for storing document chunks and their embeddings
document_chunks = []
document_embeddings = []

# Preload the Llama model
model_name = 'llama2'
try:
    ollama.pull(model_name)
except Exception as e:
    print(f"Error preloading model '{model_name}': {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_document():
    if 'document' not in request.files:
        return "No file uploaded", 400

    file = request.files['document']
    if file.filename == '':
        return "No file selected", 400

    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        file.save(temp_file.name)
        temp_path = temp_file.name

    # Extract text from the document
    try:
        with pdfplumber.open(temp_path) as pdf:
            full_text = " ".join(page.extract_text() for page in pdf.pages if page.extract_text())

        # Split the text into chunks and create embeddings
        chunk_size = 500  # Define a chunk size (tune as needed)
        chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
        
        global document_chunks, document_embeddings
        document_chunks = chunks
        document_embeddings = embedding_model.encode(chunks, convert_to_tensor=True)

        return "Document uploaded and processed successfully", 200
    except Exception as e:
        return f"Error processing document: {str(e)}", 500
    finally:
        os.remove(temp_path)  # Clean up temp file

@socketio.on('question')
def handle_question(data):
    question = data.get('text', '')
    
    # Check if embeddings are ready
    if document_embeddings is None or len(document_embeddings) == 0:
        emit('answer', {'text': "No document uploaded or processed yet."})
        return

    # Encode the question
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)

    # Find the most similar document chunk to the question
    similarities = util.pytorch_cos_sim(question_embedding, document_embeddings)
    best_match_idx = similarities.argmax().item()
    best_match_text = document_chunks[best_match_idx]

    # Construct the prompt with the best-matching chunk as context
    prompt = (
        f"You are a helpful assistant. Answer questions strictly based on the following text. "
        f"Limit the output to 200 characters unless the user specifically asks it to be more than 200 characters. "
        f"If the question is outside this content, reply with 'I'm sorry, but I can only answer questions related to the provided text.' "
        f"Answer concisely: {best_match_text}"
    )

    # Generate response using Llama3.1
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': question}
            ],
        )

        # Extract the generated response
        answer = response['message']['content']
        emit('answer', {'text': answer})
    except Exception as e:
        emit('answer', {'text': f'Error generating response: {str(e)}'})

if __name__ == '__main__':
    socketio.run(app, debug=True,use_reloader=False)
