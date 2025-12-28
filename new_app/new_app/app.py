# --- app.py ---
# Flask backend server for the NLP Web Application.
# Handles preprocessing, feature extraction, and spam detection.

import re
import string
import json
import warnings
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# --- NLP & ML Libraries ---
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.data import find
from nltk import download
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- Ensure Required NLTK Resources ---
resources = [
    ('tokenizers/punkt', 'punkt'),
    ('corpora/stopwords', 'stopwords'),
    ('corpora/wordnet', 'wordnet'),
    ('corpora/omw-1.4', 'omw-1.4')
]

for path, name in resources:
    try:
        find(path)
    except LookupError:
        download(name)

# Suppress minor warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Initialize Flask App ---
app = Flask(__name__, template_folder='templates')
CORS(app)  # Enable CORS for development/testing

# --- Initialize NLP Tools ---
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# --- Spam Detection Model Setup ---
data = {
    'Offer': [1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1],
    'Link': [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0],
    'Greeting': [0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1],
    'SenderKnown': [0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1],
    'Label': [1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]  # 1 = Spam, 0 = Not Spam
}
df = pd.DataFrame(data)
X = df[['Offer', 'Link', 'Greeting', 'SenderKnown']]
y = df['Label']

spam_model = MultinomialNB()
spam_model.fit(X, y)

# --- Routes ---

@app.route('/')
def index():
    """Render the main HTML page."""
    return render_template('index.html')


# --- 1. Text Preprocessing Endpoint ---
def perform_preprocessing(text, options):
    """Applies selected preprocessing techniques to the input text."""
    processed_text = text
    tokens = nltk.word_tokenize(processed_text)

    if options.get('remove_chars_numbers'):
        tokens = [re.sub(r'[^a-zA-Z\s]', '', token) for token in tokens]
        tokens = [token for token in tokens if token]

    if options.get('lowercase'):
        tokens = [token.lower() for token in tokens]

    if options.get('stopword_removal'):
        tokens = [token for token in tokens if token not in stop_words]

    if options.get('lemmatization'):
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    if options.get('stemming'):
        tokens = [stemmer.stem(token) for token in tokens]

    final_processed_text = ' '.join(tokens)

    if options.get('tokenization') and not any(options.values()):
        return json.dumps(tokens)

    return final_processed_text


@app.route('/preprocess', methods=['POST'])
def preprocess():
    """API endpoint to handle text preprocessing."""
    try:
        data = request.json
        text = data.get('text', '')
        options = data.get('options', {})

        if not text:
            return jsonify({'error': 'Input text cannot be empty.'}), 400

        processed_text = perform_preprocessing(text, options)

        return jsonify({
            'original_text': text,
            'processed_text': processed_text
        })

    except Exception as e:
        print(f"Preprocessing error: {e}")
        return jsonify({'error': 'An internal server error occurred during preprocessing.'}), 500


# --- 2. Feature Extraction Endpoint ---
@app.route('/feature_extract', methods=['POST'])
def feature_extract():
    """Handles feature extraction requests."""
    try:
        data = request.json
        processed_text = data.get('processed_text', '')
        method = data.get('method', 'bow')

        if not processed_text:
            return jsonify({'error': 'Processed text cannot be empty. Please complete the preprocessing step first.'}), 400

        corpus = [processed_text]
        feature_output = {}

        if method == 'bow':
            vectorizer = CountVectorizer()
            X_counts = vectorizer.fit_transform(corpus)
            feature_output['vectorizer'] = 'Bag of Words (Term Frequency)'
            feature_output['features'] = vectorizer.get_feature_names_out().tolist()
            feature_output['vector'] = X_counts.toarray()[0].tolist()

        elif method == 'tfidf':
            vectorizer = TfidfVectorizer()
            X_tfidf = vectorizer.fit_transform(corpus)
            feature_output['vectorizer'] = 'TF-IDF'
            feature_output['features'] = vectorizer.get_feature_names_out().tolist()
            feature_output['vector'] = np.round(X_tfidf.toarray()[0], 4).tolist()

        elif method == 'word2vec':
            feature_output['vectorizer'] = 'Word2Vec (Conceptual)'
            feature_output['features'] = ['embedding_dim_1', 'embedding_dim_2', '...', 'embedding_dim_N']
            feature_output['vector'] = [
                'Requires a trained model or deep learning library (e.g., Gensim/TensorFlow).',
                'Represents words as dense, continuous vectors in a semantic space.',
                'The vector values are placeholders (3.4, -0.9, ...)'
            ]

        else:
            return jsonify({'error': 'Invalid feature extraction method selected.'}), 400

        return jsonify(feature_output)

    except Exception as e:
        print(f"Feature extraction error: {e}")
        return jsonify({'error': 'An internal server error occurred during feature extraction.'}), 500


# --- 3. Spam Detection Endpoint ---
@app.route('/spam_detect', methods=['POST'])
def spam_detect():
    """Predicts if input is Spam or Not Spam."""
    try:
        data = request.json
        offer = 1 if data.get('offer', 'No') == 'Yes' else 0
        link = 1 if data.get('link', 'No') == 'Yes' else 0
        greeting = 1 if data.get('greeting', 'No') == 'Yes' else 0
        sender_known = 1 if data.get('sender_known', 'No') == 'Yes' else 0

        input_data = np.array([[offer, link, greeting, sender_known]])
        prediction = spam_model.predict(input_data)[0]
        prediction_proba = spam_model.predict_proba(input_data)[0]

        result = 'Spam' if prediction == 1 else 'Not Spam'
        confidence = prediction_proba[prediction] * 100

        return jsonify({
            'prediction': result,
            'confidence': f"{confidence:.2f}%"
        })

    except Exception as e:
        print(f"Spam detection error: {e}")
        return jsonify({'error': 'An internal server error occurred during spam detection.'}), 500


# --- Main Run Block ---
if __name__ == '__main__':
    print("Starting Flask NLP Backend on http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
