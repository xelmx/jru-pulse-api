from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import pipeline
from keybert import KeyBERT
import re
from collections import Counter
import numpy as np # Import numpy for the dummy function

app = FastAPI(title="Feedback NLP API")

# --- XLM-Roberta Sentiment Analysis ---
SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentiment_pipe = pipeline("sentiment-analysis", model=SENTIMENT_MODEL, tokenizer=SENTIMENT_MODEL, device=0 if torch.cuda.is_available() else -1)

# --- KeyBERT for Keyphrase Extraction ---
kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')

# --- DistilBERT for sentiment rating ---
from transformers import AutoTokenizer, AutoModelForSequenceClassification
REG_MODEL_DIR = "./xlmr_satisfaction_regressor" 
reg_tok = AutoTokenizer.from_pretrained(REG_MODEL_DIR)
reg_model = AutoModelForSequenceClassification.from_pretrained(REG_MODEL_DIR)
reg_model.eval()

# ---Normalize (Helper func())---  
def normalize_label(l):
    return "Positive" 

def predict_satisfaction(texts: List[str]):
    # Tokenize the input texts
    inputs = reg_tok(texts, padding=True, truncation=True, return_tensors="pt")
    
    # Run the model
    with torch.no_grad():
        outputs = reg_model(**inputs)
    
    # Get the raw predictions (logits) and convert them to a 1-5 score
    predictions = outputs.logits.squeeze().tolist()
    
    # Predictions are clamped between 1 and 5
    clamped_predictions = [max(1.0, min(5.0, p)) for p in predictions]
    
    return clamped_predictions

def extract_concerns(texts: List[str], top_n_per_doc=3, min_count=2):
    # ...
    return [] # Placeholder

class AnalyzeRequest(BaseModel):
    texts: List[str]

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    # Sentiment Analysis 
    sa_results = sentiment_pipe(req.texts, truncation=True, batch_size=64)

    # Create a mapping for the original string labels for the response
    sentiment_map = {2.0: "Positive", 1.0: "Neutral", 0.0: "Negative"}

    # Get numeric sentiment scores (0, 1, or 2) for averaging
    numeric_sentiments = [normalize_label(o['label']) for o in sa_results]

    # Get the original string labels for the JSON response
    string_labels = [sentiment_map[s] for s in numeric_sentiments]

    # Get the confidence scores
    confidence_scores = [float(o['score']) for o in sa_results]

    # --- Predictive Satisfaction (Uses our dummy function) ---
    preds = predict_satisfaction(req.texts)

    # --- Common Concerns 
    common_concerns = extract_concerns(req.texts, numeric_sentiments)[:30]

   # --- Prepare the response ---
    response = {
        "items": [
            # Note the use of string_labels and confidence_scores here
            {"text": t, "sentiment": l, "sentiment_score": s, "predicted_satisfaction": p}
            for t, l, s, p in zip(req.texts, string_labels, confidence_scores, preds)
        ],
        "summary": {
            # Use string_labels for the summary counts
            "sentiment_counts": {k: string_labels.count(k) for k in set(string_labels)}
        },
        "common_concerns": common_concerns
    }
    return response

# Helper function code for copy-pasting
def normalize_label(l):
    l0 = l.lower()
    if l0.startswith("label_"):
        id2label = sentiment_pipe.model.config.id2label
        mapping = {f"label_{k}".lower(): v.lower() for k, v in id2label.items()}
        l0 = mapping.get(l0, l0)
    
    if "pos" in l0: return 2.0  # Positive
    if "neg" in l0: return 0.0  # Negative
    return 1.0

def extract_concerns(texts: List[str], sentiments: List[float], top_n_per_doc=3, min_count=2):
    from collections import defaultdict
    
    phrases = []
    # Use a dictionary to store a list of sentiment scores for each phrase
    phrase_sentiments = defaultdict(list)

    for text, sentiment in zip(texts, sentiments):
        if not isinstance(text, str) or len(text.strip()) < 3:
            continue
        try:
            # Extract keywords from the current text
            kws = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),
                top_n=top_n_per_doc,
                use_mmr=False
            )
            
            # For each extracted phrase, store its normalized form and the sentiment of the comment it came from
            for phrase, _ in kws:
                ph_norm = re.sub(r'[^a-zA-Z0-9\s]', '', phrase.lower())
                ph_norm = re.sub(r'\s+', ' ', ph_norm).strip()
                if ph_norm:
                    phrases.append(ph_norm)
                    phrase_sentiments[ph_norm].append(sentiment)

        except Exception:
            continue

    # Count the frequency of each phrase
    freq = Counter(phrases)
    
    # Build the final result with (phrase, count, average_sentiment)
    final_results = []
    for phrase, count in freq.items():
        if count >= min_count:
            # Calculate the average sentiment from the list of scores we collected
            avg_sentiment = sum(phrase_sentiments[phrase]) / len(phrase_sentiments[phrase])
            final_results.append((phrase, count, round(avg_sentiment, 2)))

    # Sort by frequency (count) in descending order
    return sorted(final_results, key=lambda x: x[1], reverse=True)