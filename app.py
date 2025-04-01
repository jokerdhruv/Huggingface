from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import time

# Flask application
app = Flask(__name__)

# Load the Sentence Transformer model
MODEL_NAME = "all-MiniLM-L6-v2"  # You can change this to another efficient model
model = SentenceTransformer(MODEL_NAME)

def generate_summary(text):
    """Generate an embedding-based summary for a given text"""
    try:
        # Simulate a simple summarization by taking the first few sentences
        sentences = text.split('.')[:2]  # Taking the first two sentences as a basic summary
        summary = '. '.join(sentences).strip()
        return summary
    except Exception as e:
        print(f"Error generating summary: {e}")
        return f"Error: {str(e)}"

# API Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok", "message": "API is running and model is loaded"}), 200

@app.route('/summarize', methods=['POST'])
def summarize_text():
    """
    Endpoint to summarize a single text
    
    Expected JSON payload:
    {
        "text": "Text to summarize..."
    }
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    if 'text' not in data:
        return jsonify({"error": "Missing required field: text"}), 400
    
    text = data['text']
    summary = generate_summary(text)
    
    return jsonify({"summary": summary}), 200

@app.route('/batch-summarize', methods=['POST'])
def batch_summarize():
    """
    Endpoint to summarize multiple texts in one request
    
    Expected JSON payload:
    {
        "items": [
            {"text": "Text 1 to summarize..."},
            {"text": "Text 2 to summarize..."}
        ]
    }
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    if 'items' not in data or not isinstance(data['items'], list):
        return jsonify({"error": "Missing or invalid 'items' field: must be a list"}), 400
    
    results = [{"summary": generate_summary(item['text'])} for item in data['items'] if 'text' in item]
    
    return jsonify({"results": results}), 200

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle exceptions globally"""
    print(f"Unexpected error: {e}")
    return jsonify({"error": "Internal server error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
