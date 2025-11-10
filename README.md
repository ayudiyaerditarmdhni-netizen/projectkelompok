Project: Sentiment classification (text) — TensorFlow
Structure:
- data/processed/sentiment_dataset.csv : processed dataset (text,label)
- src/train_text.py : training script (TensorFlow)
- src/serve_fastapi.py : FastAPI server (loads saved model from models/saved_text_model)
- web/index.html : simple web UI with 3 forms
- requirements.txt : python libs