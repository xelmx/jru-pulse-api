# JRU-PULSE API

A REST API built with Python and Fast API to serve NLP predictions for sentiment analysis, keyword extraction, and satisfaction scoring.

### TECH Stack: 

* **Key Libraries:** FastAPI, PyTorch, Hugging Face Transformers, KeyBERT, DistilBERT

---
### API Endpoints: 

### `POST /analyze`

Analyzes a list of texts for sentiments, keyword and satisfactions.

* **Request Body:** 
    ```json
    {
        "texts": [
            "Very helpful staff.",
            "Long waiting and queue. Please add some counter",
            "Unprofessional staff. Please train them"
        ]
    }
    ```

* **Response:**
    ``` json
    {
    "items": [
        {
        "text": "Very helpful staff.",
        "sentiment": "Positive",
        "sentiment_score": 0.8723053336143494,
        "predicted_satisfaction": 4.815608501434326
        },
        {
        "text": "Long waiting and queue. Please add some counter",
        "sentiment": "Neutral",
            "sentiment_score": 0.4270479083061218,
            "predicted_satisfaction": 2.012843608856201
            },
            {
            "text": "Unprofessional staff. Please train them",
            "sentiment": "Negative",
            "sentiment_score": 0.8494417071342468,
            "predicted_satisfaction": 2.7077887058258057
            }
        ],
        "summary": {
            "sentiment_counts": {
            "Negative": 1,
            "Neutral": 1,
            "Positive": 1
            }
        },
        "common_concerns": [
            [
            "staff",
            2,
            1
            ]
        ]
    }
    ```


