import tensorflow as tf
import json, os

MODEL_DIR = os.path.join("models", "saved_text_model")

from keras.layers import TFSMLayer
model = TFSMLayer(MODEL_DIR, call_endpoint='serving_default')
with open(os.path.join(MODEL_DIR, "label_vocab.json"), "r") as f:
    label_vocab = json.load(f)

texts = ["aku suka banget", "jelek banget aplikasinya", "lumayan sih"]

preds = model(texts)
labels = [label_vocab[int(x)] for x in preds.argmax(axis=1)]

print(list(zip(texts, labels)))
