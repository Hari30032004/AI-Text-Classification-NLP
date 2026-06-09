
# Importing Libraries

import pandas as pd                                           # for loading and manupulating the dataset
import re                                                     # used fr text cleaning
from nltk.corpus import stopwords                             # provides a list of common English stopwords to remove
from nltk.stem import PorterStemmer                           # applies stemming to reduce words to their root form
from sklearn.model_selection import train_test_split          # forsplitting the data into train/val/test sets
from sklearn.feature_extraction.text import TfidfVectorizer   # for TF-IDF features (MLP input)
import numpy as np                                            # for numerical operations (padding sequences)
import joblib                                                 # for saving the processed data to files..

# Loading model
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Text Cleaning

def clean_text(text):
    if pd.isna(text):
        return " "
    text = text.lower()                         # converts text to lower case
    text = re.sub(r"[^a-z\s]", " ",text)        # removes punctuation,numbers and symbols
    
    tokens = text.split()                        # splits text into individual words 
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]     # removes stopwords
    return " ".join(tokens)                      # joins processed words back into a string


# Loading and cleaning the data

df = pd.read_csv("social-media-release.csv")


 # cleaning both headline and post

clean_headlines = []               # list to store cleaned headline text
clean_posts = []                   # list to store cleaned post text

print("Cleaning text...")

for i, (h, p) in enumerate(zip(df["news_headline"].astype(str), df["post"].astype(str))):          # iterating through headline and post together, row by row
    
    if i % 1000 == 0 and i != 0:                  # printing progress for every 1000 rows
        print(f"Processed {i} posts")

    clean_headlines.append(clean_text(h))           # cleaning headline and adding it to list
    clean_posts.append(clean_text(p))               # cleaning post and adding it to list

df["clean_headline"] = clean_headlines              # saving cleaned headkines to dataframe
df["clean_post"] = clean_posts                      # saving cleaned posts to dataframes.

                                        
# comnining them

df["combined_text"] = df["clean_headline"] + " " + df["clean_post"]

# encoding the labels as True/False

print(df["class_label"].unique())
df["label"] = df["class_label"].astype(str).str.strip().str.upper().map({"TRUE": 1,"FALSE":0})

X = df["combined_text"]
y = df["label"]

print("Text Clenaing done")
# Splitting the data into training,validation and test data sets..

X_trainval, X_test, y_trainval, y_test = train_test_split( 
    X, y, test_size=0.2, random_state=42                         # training and validation data - 80%, test data - 20%          
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42       # traing data - 60% and 20% test data(from previous split).
)

print("Train size:", len(X_train))
print("Validation size:", len(X_val))
print("Test size:", len(X_test))

# TF-IDF vectors for MLP model

tfidf = TfidfVectorizer(max_features=3000)       # vocabulary limit is set to speed-up the training and for preventing overfitting..

X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)
X_test_tfidf = tfidf.transform(X_test)

# saving TF-IDF data for MLP model..

joblib.dump(
    (X_train_tfidf, X_val_tfidf, X_test_tfidf, y_train, y_val, y_test),
    "mlp_data.pkl"
)
joblib.dump(tfidf, "tfidf.pkl")

print("TF-IDF data saved!")


# Tokeniser and Padded sequences for CNN model.. 

print("Creating tokenised sequences..")

# Building vocabulary from training data..
def build_vocab(texts, vocab_size=20000):
    freq = {}
    for text in texts:
        for word in text.split():
            freq[word] = freq.get(word, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    vocab = {w: i+2 for i, (w, _) in enumerate(sorted_words[:vocab_size])}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab

vocab = build_vocab(X_train.tolist(), vocab_size=20000)

# Converting text data into integer sequences
def text_to_seq(text, vocab):
    return [vocab.get(word, 1) for word in text.split()]  

X_train_seq = [text_to_seq(t, vocab) for t in X_train]
X_val_seq   = [text_to_seq(t, vocab) for t in X_val]
X_test_seq  = [text_to_seq(t, vocab) for t in X_test]

# Padding sequences to fixed length.

def pad(seq_list, maxlen=100):
    padded = np.zeros((len(seq_list), maxlen), dtype="int32")
    for i, seq in enumerate(seq_list):
        seq = seq[:maxlen]
        padded[i, :len(seq)] = seq    # padding shorter ones with zeros
    return padded

X_train_seq = pad(X_train_seq)
X_val_seq   = pad(X_val_seq)
X_test_seq  = pad(X_test_seq)

# Saving CNN data
joblib.dump(
    (X_train_seq, X_val_seq, X_test_seq, y_train, y_val, y_test),
    "cnn_data.pkl"
)
joblib.dump(vocab, "vocab.pkl")

print("CNN sequence data saved!")
print("Preprocessing complete!")