# CNN Classifier

import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential       # sequential model container 
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense    # converts word indices,etracts local patterns from text,reduces sequences and fully connected layer for classification
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Loading preprocessed padded sequence data
X_train, X_val, X_test, y_train, y_val, y_test = joblib.load("cnn_data.pkl")

vocab = joblib.load("vocab.pkl")      # loads vocabulary dictionary

vocab_size = len(vocab) + 1             # number of unique words in vocabulary

# Building CNN Model
model = Sequential([
    Embedding(vocab_size, 128, input_length=100),    # number of unique words with 128 size of each word vector and each sequence is padded to length 100
    Conv1D(64, 3, activation="relu"),                # 1D convolution matrics containing 64 filters, kernel size = 3 and relu activation function for non-linearity
    GlobalMaxPooling1D(),                               # takes the max_val from each filter
    Dense(64, activation="relu"),                    # fully connected layer with 64 neurons
    Dense(1, activation="sigmoid")                   # output layer
])

model.compile(loss="binary_crossentropy",            # loss function for binary classification
               optimizer="adam",                     # adaptive optimizer
               metrics=["accuracy"]                  # keeps a track on accuracy during training
)
model.summary()

# Training the model 
history = model.fit( X_train, y_train, epochs = 5, batch_size =32, validation_data = (X_val, y_val))

# plotting training and validation accuracy/loss


history_dict = history.history   # training history dictionary

plt.figure(figsize=(12, 5))   

# Accuracy subplot 
plt.subplot(1, 2, 1)             
plt.plot(history_dict["accuracy"], label="Train Accuracy")
plt.plot(history_dict["val_accuracy"], label="Validation Accuracy")
plt.title("CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss subplot
plt.subplot(1, 2, 2)             
plt.plot(history_dict["loss"], label="Train Loss")
plt.plot(history_dict["val_loss"], label="Validation Loss")
plt.title("CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()


# Evaluating the model
test_pred = (model.predict(X_test) >= 0.5).astype(int)
print("Test accuracy:", accuracy_score(y_test, test_pred))
print(classification_report(y_test, test_pred, target_names=["FALSE","TRUE"]))

# Save model
model.save("cnn_model.h5")  
print("CNN model saved!")