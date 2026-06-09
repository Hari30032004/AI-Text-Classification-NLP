# Testing the saved MLP model and CNN model on new text data

import joblib
import tensorflow as tf
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 


# loading dataset

df = pd.read_csv("social-media-release.csv")

# loading nltk components

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Text cleaning

def clean_text(text):
    text = text.lower()                         # converts the text to lowercase and process with spacy
    text = re.sub(r"[^a-z\s]"," ",text)         # removes punctuation,numbers and symblos
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words] 
    return " ".join(words)

# Loading all the saved models
mlp = joblib.load("mlp_model.pkl")                 # loading trained mlp model
tfidf = joblib.load("tfidf.pkl")                   # loading saved TF-IDF vectorizer
cnn = tf.keras.models.load_model("cnn_model.h5")   # loading trained CNN model
vocab = joblib.load("vocab.pkl")                   # laoding vocabulary dictionary

# conveting cleaned text into integer sequence for CNN

def text_to_seq(text, vocab):
    return [vocab.get(word, 1) for word in text.split()]     # get index of each word, use 1 for 00V 

# Manual padding (same as preprocessing)
def pad(seq_list, maxlen=100):
    padded = np.zeros((len(seq_list), maxlen), dtype="int32")      # creates zero matrix
    for i, seq in enumerate(seq_list):
        seq = seq[:maxlen]
        padded[i, :len(seq)] = seq          # fill with sequence values
    return padded

# classifying multiple inputs at once

def classify_multiple(pairs):
    for i, (headline, post) in enumerate(pairs, start=1):
        clean_h = clean_text(headline)                    # cleans and preprocess headline data
        clean_p = clean_text(post)                        # cleans post data
        combined = clean_h + " " + clean_p


        # MLP prediction
        tfidf_vect = tfidf.transform([combined])        # converts o TF-IDF vector
        mlp_pred = mlp.predict(tfidf_vect)[0]           # predicts the cass(0(or)1)
        mlp_prob = mlp.predict_proba(tfidf_vect)[0][mlp_pred]     # confidence score
        mlp_label = "TRUE" if mlp_pred == 1 else "FALSE"      # converting to labels

         # CNN prediction

        seq = text_to_seq(combined,vocab)             # converts text to integer sequence
        seq = pad([seq])                              # pad to fixed length
    
        prob = cnn.predict(seq)[0][0]                 # predicted probability        
        cnn_pred = int(prob >= 0.5)                   # CNN output probability - converts to 0/1
 
        cnn_label ="TRUE" if cnn_pred == 1 else "FALSE"    # converts to label
        cnn_conf = prob if cnn_pred == 1 else (1-prob)     # confidence score

        print(f"\n Example {i} :     ")
        print("Headline : ",headline)
        print("Post : ",post)

        # predictions
        print(f"MLP Prediction : {mlp_label} (confidence: {mlp_prob:.2f})")
        print(f"CNN Prediction : {cnn_label} (confidence: {cnn_conf:.2f})")


# sample Input
classify_multiple([
    ("In Afghanistan, over 100 billion dollars spent...",
    "US Deep State Gestapo leaves 100 billion dolla..."),

    ("Heavy rainfall causes flooding in several parts of the city",
     "Many areas in the city are flooded today due to the heavy rainfall reported earlier"),


])