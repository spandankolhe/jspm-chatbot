from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from rapidfuzz import process, fuzz
import re
from sentence_transformers import SentenceTransformer, util
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# --- Load college data ---
with open("college_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# --- Prepare key/value lists ---
DATA_KEYS = list(data.keys())
DATA_VALUES = [data[k] for k in DATA_KEYS]

# --- Text cleaning function ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    stopwords = ['tell', 'me', 'about', 'what', 'is', 'the', 'who', 'when', 'can', 'you',
                 'give', 'info', 'information', 'details', 'of', 'our', 'college', 'department',
                 'please', 'show', 'list', 'for']
    words = [w for w in text.split() if w not in stopwords]
    return ' '.join(words)

# --- Load sentence embedding model once ---
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
print(f" Loading model: {MODEL_NAME} ...")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded successfully!")

# --- Encode all data keys into embeddings ---
print(" Generating embeddings for knowledge base...")
key_embeddings = model.encode(DATA_KEYS, convert_to_tensor=True, show_progress_bar=False)
print(f"{len(DATA_KEYS)} entries encoded!")

# --- Semantic search using cosine similarity ---
def semantic_match(user_text, top_k=1):
    if not user_text or user_text.strip() == "":
        return None, 0.0
    user_emb = model.encode(user_text, convert_to_tensor=True)
    cos_scores = util.cos_sim(user_emb, key_embeddings)[0]
    top_results = np.argsort(-cos_scores.cpu().numpy())[:top_k]
    best_idx = int(top_results[0])
    best_score = float(cos_scores[best_idx].cpu().numpy())
    return DATA_KEYS[best_idx], best_score

# --- Fuzzy fallback (RapidFuzz) ---
def fuzzy_match(user_text):
    if not user_text:
        return None, 0
    match, score, _ = process.extractOne(user_text, DATA_KEYS, scorer=fuzz.token_set_ratio)
    return match, score  # score 0–100

# --- API Endpoint ---
@app.route("/api/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    cleaned = clean_text(user_message)

    # 1️⃣ Semantic matching first
    sem_key, sem_score = semantic_match(cleaned)
    # sem_score is cosine similarity (0 to 1)
    sem_score_scaled = (sem_score + 1) / 2

    # 2️⃣ Fuzzy matching as backup
    fuzzy_key, fuzzy_score = fuzzy_match(cleaned)

    # 3️⃣ Decision logic
    if sem_key and sem_score_scaled > 0.55:
        best_key = sem_key
        match_type = "semantic"
    elif fuzzy_key and fuzzy_score > 60:
        best_key = fuzzy_key
        match_type = "fuzzy"
    else:
        best_key = None
        match_type = None

    # 4️⃣ Build reply
    if best_key:
        response = data[best_key]
        reply = f"{response}"
    else:
        reply = "Sorry, I couldn’t understand that. Could you please rephrase?"

    print(f"User: {user_message}")
    print(f"Matched by: {match_type} -> Key: {best_key} | Score: {sem_score_scaled:.2f}")


    return jsonify({"reply": reply})

# --- Run the Flask app ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
