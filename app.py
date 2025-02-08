from flask import Flask, request, jsonify
import google.generativeai as genai
from langdetect import detect
from textblob import TextBlob
from concurrent.futures import ThreadPoolExecutor
import json

app = Flask(__name__)

# Configure API Key securely
genai.configure(api_key="your_google_api_key_here")

# Initialize ThreadPoolExecutor for concurrent processing
executor = ThreadPoolExecutor(max_workers=5)

# AI response function
def get_ai_response(prompt, email_content):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt + email_content[:2000])
        return response.text.strip()
    except Exception as e:
        return f"AI Error: {e}"

# Sentiment analysis using TextBlob
def get_sentiment(email_content):
    return TextBlob(email_content).sentiment.polarity

# Readability score (simplified)
def get_readability(email_content):
    return round(TextBlob(email_content).sentiment.subjectivity * 10, 2)

# API endpoint to process email content and return insights
@app.route('/analyze_email', methods=['POST'])
def analyze_email():
    email_content = request.json.get("email_content", "")
    
    if not email_content:
        return jsonify({"error": "Email content is required"}), 400
    
    detected_lang = detect(email_content)
    if detected_lang != "en":
        return jsonify({"error": "Only English language is supported."}), 400
    
    features = {
        "sentiment": True,
        "highlights": True,
        "response": True,
        "tone": True,
        "urgency": True,
        "task_extraction": True,
        "subject_recommendation": True,
        "category": True,
        "politeness": True,
        "emotion": True,
        "spam_check": True,
        "readability": True,
        "root_cause": True,
        "grammar_check": True,
        "clarity": True,
        "best_response_time": True,
        "professionalism": True,
    }
    
    # Initiate all analyses in parallel
    futures = {}
    if features.get("highlights"):
        futures["summary"] = executor.submit(get_ai_response, "Summarize this email concisely:\n\n", email_content)
    if features.get("response"):
        futures["response"] = executor.submit(get_ai_response, "Generate a professional response to this email:\n\n", email_content)
    if features.get("tone"):
        futures["tone"] = executor.submit(get_ai_response, "Detect the tone of this email:\n\n", email_content)
    if features.get("urgency"):
        futures["urgency"] = executor.submit(get_ai_response, "Analyze urgency level:\n\n", email_content)
    if features.get("task_extraction"):
        futures["tasks"] = executor.submit(get_ai_response, "List actionable tasks:\n\n", email_content)
    if features.get("sentiment"):
        sentiment = get_sentiment(email_content)
    if features.get("readability"):
        readability_score = get_readability(email_content)
    
    results = {}
    for key, future in futures.items():
        results[key] = future.result()

    # Add non-AI results to the response
    results.update({
        "sentiment": {"polarity": sentiment, "label": "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"},
        "readability": readability_score,
    })
    
    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
