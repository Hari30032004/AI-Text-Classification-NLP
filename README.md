**AI Text Classification & Topic Discovery**

Overview: This project develops an end-to-end Natural Language Processing (NLP) pipeline to classify social media posts as real or fake. The system applies text preprocessing, feature extraction, neural network modelling, and topic discovery techniques to analyse large-scale textual data.

Dataset: 
The dataet contains
* 89,000+ social media posts
* 947 news headlines
* Binary labels indicating whether content is real or fake
The original dataset is not included in this repository due to file size constraints. 

Methods:

1. Text Preprocessing

* Text cleaning and normalization
* Stop-word removal
* Porter stemming
* Headline and post feature combination

2. Feature Engineering

* TF-IDF vectorisation
* Sequence generation and padding
* Custom vocabulary construction

3. Classification Models

* Multi-Layer Perceptron (MLP)
  - TF-IDF document ca=lassification
  - Hyperparameter tuning
  - Validation and test evaluation
* Convolutional Neural Network
  - Embedding layer
  - Conv1D architecture
  - Global max pooling
  - Binary text classification

4. Topic Discovery

* Latent Dirichlet Allocation (LDA)
* Bag-of-Words representation
* TF-IDF representation
* Topic interpretation and analysis

Evaluation:

Models were evaluated using:

1. Accuracy
2. Precision
3. Recall
4. F1-Score
5. Classification Reports

Technologies:

* Python
* Pandas
* Numpy
* Scikit-learn
* Tensorflow/Keras
* NLTK
* Matplotlib
* Joblib

Future Improvements

* Transformer-based models (BERT)
* Advanced contextual embeddings
* Explainable AI techniques
* Model deployment through a web application
