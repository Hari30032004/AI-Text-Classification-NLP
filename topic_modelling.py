# LDA topic modelling

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation     # LDA - topic modelling algorithm
from data_preprocessing import clean_text

# Loading the dataset
df = pd.read_csv("social-media-release.csv")

# Combining headline and post data features..
texts = (df["news_headline"].astype(str).apply(clean_text)+ " " + df["post"].astype(str).apply(clean_text)).tolist()

# LDA with Bag-of-Words

# BoW representation
vect = CountVectorizer(
    stop_words="english",
    max_features = 5000               # max_features keeps the LDA fast and stable
)
bow = vect.fit_transform(texts)


# LDA model on BoW
lda = LatentDirichletAllocation(n_components=5,      # number of topics
                                max_iter = 5,        # number of training iterations
                                random_state=42
                                )    
lda.fit(bow)                                         # training LDA in BoW data

words = vect.get_feature_names_out()                 # List of vocabulary words

# Labelling each Topic
topic_names = {0 :"Government",
               1 : "Elections / Voting",
               2 : "Covid / Public Health",
               3 : "Economy / Tax Policy",
               4 : "Public Policy"
} 

# Printing the BoW topics
for i, topic in enumerate(lda.components_):       # loop through each topic
    label = topic_names.get(i, f"Topic {i+1}")
    print(f"\nTopic {i+1} - {label} :")
    top_words = [words[j] for j in topic.argsort()[-10:]]
    print(top_words)


# LDA with TF-IDF

tfidf_vect = TfidfVectorizer( 
    stop_words ="english",
    max_features = 5000
)

tfidf = tfidf_vect.fit_transform(texts)           # converts text into TF-IDF matrix

# LDA model on TF-IDF

lda_tfidf = LatentDirichletAllocation(n_components=5,max_iter=5,random_state=42)
lda_tfidf.fit(tfidf)                             # Trains LDA on TF-IDF data
tfidf_words = tfidf_vect.get_feature_names_out()

# printing TF-IDF topics

for i, topic in enumerate(lda_tfidf.components_):          # loops through each topic
    label = topic_names.get(i, f"Topic {i+1}")             # reuse same labels
    print(f"\n Topic {i+1} - {label}:")                    # prints topic number and label

    top_words = [tfidf_words[j] for j in topic.argsort()[-10:]]

    print(top_words)

print("Topics generated")
